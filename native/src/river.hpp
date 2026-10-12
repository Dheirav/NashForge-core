// The river re-solver's CFR+ core, ported from cfr/river.py.
//
// The Python solver runs 400 vector-CFR+ iterations over about 160 nodes and
// 1,081 exact hands in numpy, two to four seconds a river, which made a
// 300-match test against the field panel a three-hour job (22 September
// 2026). The tree is still built in Python from the live chips and handed
// over flat (nodes in creation order, so every child has a higher index than
// its parent and the forward and backward passes are loops, not recursion);
// the hand set's rank order and per-card member lists come over once per
// board. This does the iterations. Same arithmetic as the Python: regret
// matching+, linear averaging of the strategy, showdown values from one
// cumulative sum over the hands in rank order with per-card corrections for
// the blockers, fold values from the compatible reach mass.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

// The passes are dense loops over 1,081 hands with a max and a divide; letting
// GCC reassociate and drop the NaN checks is what gets them vectorised, and
// nothing here depends on IEEE corner cases (regrets are clipped at zero).
#pragma GCC push_options
#pragma GCC optimize ("O3", "fast-math")

namespace pokerbot {

struct RiverTree {
    // Per node.
    std::vector<int> kind;          // 0 decision, 1 fold, 2 showdown
    std::vector<int> player;        // decision: who acts
    std::vector<int> contrib0, contrib1;
    std::vector<int> folder;        // fold: who folded
    std::vector<std::vector<int>> children;   // decision: child node per action
    std::vector<std::vector<int>> actions;    // decision: action code per child
};

class RiverSolver {
public:
    RiverSolver(std::vector<int> pairs_a, std::vector<int> pairs_b, std::vector<int64_t> ranks,
                RiverTree tree, std::vector<double> range0, std::vector<double> range1)
        : a_(std::move(pairs_a)), b_(std::move(pairs_b)), ranks_(std::move(ranks)), tree_(std::move(tree)),
          range_{std::move(range0), std::move(range1)} {
        H_ = static_cast<int>(ranks_.size());
        N_ = static_cast<int>(tree_.kind.size());
        // Rank order and the strictly-lower / at-or-below positions.
        order_.resize(H_);
        for (int i = 0; i < H_; ++i) order_[i] = i;
        std::stable_sort(order_.begin(), order_.end(), [&](int x, int y) { return ranks_[x] < ranks_[y]; });
        std::vector<int64_t> sorted(H_);
        for (int i = 0; i < H_; ++i) sorted[i] = ranks_[order_[i]];
        lo_.resize(H_); hi_.resize(H_);
        for (int h = 0; h < H_; ++h) {
            lo_[h] = static_cast<int>(std::lower_bound(sorted.begin(), sorted.end(), ranks_[h]) - sorted.begin());
            hi_[h] = static_cast<int>(std::upper_bound(sorted.begin(), sorted.end(), ranks_[h]) - sorted.begin());
        }
        // Per card: member hands sorted by rank, with their lower/upper positions within the member list.
        members_.assign(52, {}); member_lo_.assign(52, {}); member_hi_.assign(52, {});
        for (int h : order_) { members_[a_[h]].push_back(h); members_[b_[h]].push_back(h); }
        for (int c = 0; c < 52; ++c) {
            const auto& m = members_[c];
            const int n = static_cast<int>(m.size());
            member_lo_[c].resize(n); member_hi_[c].resize(n);
            for (int i = 0; i < n; ++i) {
                int lo = i; while (lo > 0 && ranks_[m[lo - 1]] == ranks_[m[i]]) --lo;
                int hi = i + 1; while (hi < n && ranks_[m[hi]] == ranks_[m[i]]) ++hi;
                member_lo_[c][i] = lo; member_hi_[c][i] = hi;
            }
        }
        regrets_.resize(N_); strategy_sum_.resize(N_);
        for (int n = 0; n < N_; ++n) {
            if (tree_.kind[n] != 0) continue;
            const size_t width = tree_.actions[n].size() * static_cast<size_t>(H_);
            regrets_[n].assign(width, 0.0);
            strategy_sum_[n].assign(width, 0.0);
        }
        reach_.assign(2, std::vector<std::vector<double>>(N_, std::vector<double>(H_, 0.0)));
        value_.assign(N_, std::vector<double>(H_, 0.0));
        sigma_.resize(N_);
        for (int n = 0; n < N_; ++n) if (tree_.kind[n] == 0) sigma_[n].assign(tree_.actions[n].size() * H_, 0.0);
        scratch_.assign(H_ + 1, 0.0);
        total_.assign(H_, 0.0);
    }

    int iterations() const { return iterations_; }

    int solve(int iterations, double budget_s) {
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ++iterations_;
            pass(0); pass(1);
            const double taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (taken > budget_s) break;
        }
        return iterations_;
    }

    /// The average strategy at the root for one hand, one probability per root action.
    std::vector<double> root_strategy(int hand) const {
        const int width = static_cast<int>(tree_.actions[0].size());
        std::vector<double> out(width, 0.0);
        double total = 0.0;
        for (int a = 0; a < width; ++a) { out[a] = strategy_sum_[0][a * H_ + hand]; total += out[a]; }
        if (total <= 0.0) { for (auto& p : out) p = 1.0 / width; return out; }
        for (auto& p : out) p /= total;
        return out;
    }

private:
    void strategy_of(int n) {
        // Hand-inner loops throughout: the arrays are [action][hand], so the
        // inner loop runs contiguously and the compiler vectorises it. The
        // first version looped hands outer and ran three times slower than
        // this, barely ahead of numpy.
        const int width = static_cast<int>(tree_.actions[n].size());
        const auto& regret = regrets_[n];
        auto& sigma = sigma_[n];
        auto& total = total_;
        std::fill(total.begin(), total.end(), 0.0);
        for (int a = 0; a < width; ++a) {
            const double* r = &regret[static_cast<size_t>(a) * H_];
            for (int h = 0; h < H_; ++h) total[h] += r[h] > 0.0 ? r[h] : 0.0;
        }
        const double uniform = 1.0 / width;
        for (int a = 0; a < width; ++a) {
            const double* r = &regret[static_cast<size_t>(a) * H_];
            double* sg = &sigma[static_cast<size_t>(a) * H_];
            for (int h = 0; h < H_; ++h) sg[h] = total[h] > 0.0 ? (r[h] > 0.0 ? r[h] : 0.0) / total[h] : uniform;
        }
    }

    /// value[h] = contrib_opp * (opponent mass below h - mass above h), compatible hands only.
    void showdown_value(const std::vector<double>& reach, double stake, std::vector<double>& value) {
        auto& cum = scratch_;
        cum[0] = 0.0;
        for (int i = 0; i < H_; ++i) cum[i + 1] = cum[i] + reach[order_[i]];
        const double total = cum[H_];
        for (int h = 0; h < H_; ++h) value[h] = cum[lo_[h]] - (total - cum[hi_[h]]);
        for (int c = 0; c < 52; ++c) {
            const auto& m = members_[c];
            if (m.empty()) continue;
            const int n = static_cast<int>(m.size());
            card_cum_.resize(n + 1); card_cum_[0] = 0.0;
            for (int i = 0; i < n; ++i) card_cum_[i + 1] = card_cum_[i] + reach[m[i]];
            const double ctotal = card_cum_[n];
            for (int i = 0; i < n; ++i)
                value[m[i]] -= card_cum_[member_lo_[c][i]] - (ctotal - card_cum_[member_hi_[c][i]]);
        }
        for (int h = 0; h < H_; ++h) value[h] *= stake;
    }

    /// mass[h] = opponent reach over hands sharing no card with h, times `stake`.
    void fold_value(const std::vector<double>& reach, double stake, std::vector<double>& value) {
        double total = 0.0;
        for (int h = 0; h < H_; ++h) total += reach[h];
        double per_card[52];
        for (int c = 0; c < 52; ++c) { double s = 0.0; for (int h : members_[c]) s += reach[h]; per_card[c] = s; }
        for (int h = 0; h < H_; ++h) value[h] = stake * (total - per_card[a_[h]] - per_card[b_[h]] + reach[h]);
    }

    void pass(int traverser) {
        const int opp = 1 - traverser;
        // Forward: reach at every node, parents before children.
        for (int n = 0; n < N_; ++n) { std::fill(reach_[0][n].begin(), reach_[0][n].end(), 0.0); std::fill(reach_[1][n].begin(), reach_[1][n].end(), 0.0); }
        reach_[0][0] = range_[0]; reach_[1][0] = range_[1];
        for (int n = 0; n < N_; ++n) {
            if (tree_.kind[n] != 0) continue;
            strategy_of(n);
            const int p = tree_.player[n];
            const auto& sigma = sigma_[n];
            const double* own = reach_[p][n].data();
            const double* other = reach_[1 - p][n].data();
            for (size_t a = 0; a < tree_.children[n].size(); ++a) {
                const int child = tree_.children[n][a];
                double* rc = reach_[p][child].data();
                double* ro = reach_[1 - p][child].data();
                const double* sg = &sigma[a * H_];
                for (int h = 0; h < H_; ++h) { rc[h] += own[h] * sg[h]; ro[h] += other[h]; }
            }
        }
        // Terminals: value to the traverser per hand.
        for (int n = 0; n < N_; ++n) {
            if (tree_.kind[n] == 2) {
                const double stake = opp == 0 ? tree_.contrib0[n] : tree_.contrib1[n];
                showdown_value(reach_[opp][n], stake, value_[n]);
            } else if (tree_.kind[n] == 1) {
                if (tree_.folder[n] == traverser) {
                    const double stake = traverser == 0 ? tree_.contrib0[n] : tree_.contrib1[n];
                    fold_value(reach_[opp][n], -stake, value_[n]);
                } else {
                    const double stake = opp == 0 ? tree_.contrib0[n] : tree_.contrib1[n];
                    fold_value(reach_[opp][n], stake, value_[n]);
                }
            }
        }
        // Backward: children before parents.
        const double weight = static_cast<double>(iterations_);
        for (int n = N_ - 1; n >= 0; --n) {
            if (tree_.kind[n] != 0) continue;
            const int width = static_cast<int>(tree_.children[n].size());
            auto& value = value_[n];
            std::fill(value.begin(), value.end(), 0.0);
            if (tree_.player[n] == traverser) {
                const auto& sigma = sigma_[n];
                for (int a = 0; a < width; ++a) {
                    const double* sg = &sigma[static_cast<size_t>(a) * H_];
                    const double* cv = value_[tree_.children[n][a]].data();
                    for (int h = 0; h < H_; ++h) value[h] += sg[h] * cv[h];
                }
                auto& regret = regrets_[n];
                auto& ssum = strategy_sum_[n];
                const double* own = reach_[traverser][n].data();
                for (int a = 0; a < width; ++a) {
                    double* r = &regret[static_cast<size_t>(a) * H_];
                    double* ss = &ssum[static_cast<size_t>(a) * H_];
                    const double* sg = &sigma[static_cast<size_t>(a) * H_];
                    const double* cv = value_[tree_.children[n][a]].data();
                    for (int h = 0; h < H_; ++h) {
                        const double updated = r[h] + cv[h] - value[h];
                        r[h] = updated > 0.0 ? updated : 0.0;                   // regret matching+
                        ss[h] += weight * own[h] * sg[h];
                    }
                }
            } else {
                for (int a = 0; a < width; ++a) {
                    const double* cv = value_[tree_.children[n][a]].data();
                    for (int h = 0; h < H_; ++h) value[h] += cv[h];
                }
            }
        }
    }

    std::vector<int> a_, b_;
    std::vector<int64_t> ranks_;
    RiverTree tree_;
    std::vector<double> range_[2];
    int H_ = 0, N_ = 0, iterations_ = 0;
    std::vector<int> order_, lo_, hi_;
    std::vector<std::vector<int>> members_, member_lo_, member_hi_;
    std::vector<std::vector<double>> regrets_, strategy_sum_, sigma_, value_;
    std::vector<std::vector<std::vector<double>>> reach_;
    std::vector<double> scratch_, card_cum_, total_;
};

}  // namespace pokerbot

#pragma GCC pop_options
