// The no-limit game as MCCFR sees it: betting, dealing, bucketing, payoffs.
//
// `nolimit.hpp` holds the betting, which is pinned against the Python exactly.
// This adds the parts that involve cards, where the port deliberately does NOT
// match bit-for-bit: bucketing runs a Monte Carlo equity estimate seeded from
// the cards, and reproducing numba's generator would buy nothing. What is
// preserved is the structure -- the same abstraction, the same centroids, the
// same tie-breaking -- so the two produce the same distribution of buckets.
#pragma once
#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "nolimit.hpp"
#include "equity.hpp"
#include "bucket_table.hpp"

namespace pokerbot {

/// The fitted card abstraction, handed over from Python.
///
/// Passed as tables rather than refitted here: k-means over sampled equities is
/// cheap and happens once, and refitting in C++ would be a second clustering
/// that could silently disagree with the one every existing strategy was built
/// against.
struct Abstraction {
    /// bucket for each unordered hole pair, indexed [card][card]. 16 bits
    /// because the lossless preflop numbers hands 0 to 168, past int8_t.
    std::vector<int16_t> preflop;           // 52 * 52
    /// Sorted centroids for flop, turn, river.
    std::array<std::vector<double>, 3> centroids;
    int equity_samples = 40;
    /// Fold the board's texture into the postflop bucket. Mirrors
    /// `abstraction.buckets.board_texture` exactly; see that docstring for why.
    bool texture = false;
    /// Histogram mode (abstraction/histogram.py): centroid histograms per
    /// street, [buckets][bins], nearest by earth mover's distance. Empty means
    /// the scalar E[HS] mode above.
    std::array<std::vector<std::vector<double>>, 3> hist_centroids;
    int hist_bins = 20;
    int hist_runouts = 100;
    int hist_opponents = 50;
    bool histogram() const { return !hist_centroids[0].empty(); }
    /// Precomputed buckets for the flop and the turn (bucket_table.hpp); an
    /// empty table means compute the histogram at the lookup.
    std::array<BucketTable, 2> hist_tables;

    static int board_texture(const int* board, int n) {
        if (n == 0) return 0;
        int suits[4] = {0, 0, 0, 0};
        bool ranks[14] = {false};          // index 0 is the ace playing low
        for (int i = 0; i < n; ++i) {
            ++suits[deck_suit(board[i])];
            ranks[deck_rank(board[i]) + 1] = true;
        }
        int most = 0;
        for (int s : suits) most = std::max(most, s);
        const int flush = most <= 2 ? 0 : (most == 3 ? 1 : 2);
        if (ranks[13]) ranks[0] = true;
        int straight = 0;
        for (int low = 0; low <= 9 && !straight; ++low) {
            int count = 0;
            for (int r = low; r <= low + 4; ++r) count += ranks[r] ? 1 : 0;
            if (count >= 4) straight = 1;
        }
        return flush * 2 + straight;
    }

    /// Mirrors `_nearest_centroid`: bisect_left, ties to the lower index.
    static int nearest(const std::vector<double>& sorted, double value) {
        size_t position = 0;
        while (position < sorted.size() && sorted[position] < value) ++position;
        if (position == 0) return 0;
        if (position == sorted.size()) return static_cast<int>(sorted.size()) - 1;
        const double below = sorted[position - 1], above = sorted[position];
        return (above - value) < (value - below)
             ? static_cast<int>(position) : static_cast<int>(position) - 1;
    }
};

//: Entries the hole-and-board bucket memo keeps before clearing. Matches
//: `games.nolimit.BUCKET_CACHE_LIMIT`; see the note at the clear site.
// Measured 15 September: 267 lookups an iteration collapse to 72 distinct
// situations, and across iterations the hit rate is zero, so a memo of a few
// thousand covers every repeat within an iteration and holds no dead weight.
constexpr size_t BUCKET_CACHE_LIMIT = 4096;


class NoLimitGame {
public:
    static constexpr int num_players = 2;

    struct State {
        pokerbot::State bet;                        // betting, from nolimit.hpp
        std::array<std::array<int8_t, 2>, 2> hole{};
    };

    NoLimitGame(Abstraction abstraction, int starting_stack, int small_blind,
                int big_blind, RaiseSchedule schedule)
        : abstraction_(std::move(abstraction)),
          betting_(starting_stack, small_blind, big_blind, std::move(schedule)) {}

    /// See `NoLimitHoldem::set_street_schedules`.
    void set_street_schedules(std::vector<RaiseSchedule> per_street) {
        betting_.set_street_schedules(std::move(per_street));
    }

    State initial_state() const {
        State s;
        s.bet = betting_.initial_state();
        return s;
    }

    bool is_chance(const State& s) const {
        if (!s.bet.dealt) return true;
        return betting_.current_player(s.bet) == CHANCE && !is_terminal(s);
    }

    /// Exact all-in terminals: once both players are all in on the flop or
    /// turn, the hand is scored as the average over every remaining runout
    /// (990 or 44 of them) rather than one sampled board. Unbiased, and it
    /// removes the single-runout noise at exactly the shove nodes. Preflop
    /// all-ins keep the sampled runout (1.7 million boards; a table later).
    /// Off by default until its convergence check and head-to-head.
    void set_exact_terminals(bool on) { exact_terminals_ = on; }

    /// Pluribus's pruning exemption: the river is never pruned, because its
    /// subtrees are cheap and end the hand.
    bool is_final_street(const State& s) const { return s.bet.board_n >= 5; }

    bool is_terminal(const State& s) const {
        if (betting_.is_terminal(s.bet)) return true;
        return exact_terminals_ && s.bet.dealt && betting_.all_in(s.bet) && s.bet.board_n >= 3;
    }
    int current_player(const State& s) const { return betting_.current_player(s.bet); }
    std::vector<int8_t> legal_actions(const State& s) const {
        return betting_.legal_actions(s.bet);
    }

    State next_state(const State& s, int8_t action) const {
        State next = s;
        next.bet = betting_.apply(s.bet, action);
        return next;
    }

    /// Common random numbers: one nine-card deal per iteration, revealed
    /// street by street on every branch, instead of a fresh draw at each
    /// chance node of each branch (about 91 runouts an iteration, so that
    /// `values[i] - value` compared actions on different boards). Each
    /// branch's estimate keeps its expectation; only the variance drops. Off
    /// by default until its head-to-head passes; see `set_common_random_numbers`.
    void set_common_random_numbers(bool on) { crn_ = on; }
    bool common_random_numbers() const { return crn_; }

    void begin_iteration(Rng& rng) {
        if (!crn_) return;
        int8_t deck[52];
        for (int i = 0; i < 52; ++i) deck[i] = static_cast<int8_t>(i);
        for (int i = 0; i < 9; ++i) {
            const int j = i + static_cast<int>(rng.below(static_cast<uint32_t>(52 - i)));
            const int8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        }
        for (int i = 0; i < 9; ++i) deal_[static_cast<size_t>(i)] = deck[i];
    }

    State sample_chance(const State& s, Rng& rng) const {
        State next = s;
        if (crn_) {
            if (!s.bet.dealt) {
                next.hole[0] = {deal_[0], deal_[1]};
                next.hole[1] = {deal_[2], deal_[3]};
                // The betting state carries its own copy of the cards for
                // anything handed only the betting state, the scripted
                // opponent of archetype.hpp above all: without this it read
                // {0, 0} for every hand and the first exploiter rungs
                // (22 September) were best responses to a card-blind bot.
                next.bet.hole = next.hole;
                next.bet.dealt = true;
                return next;
            }
            const int street = s.bet.street + 1;
            const int needed = STREET_BOARD_SIZE[static_cast<size_t>(street)] - s.bet.board_n;
            int8_t board[5];
            for (int i = 0; i < s.bet.board_n; ++i) board[i] = s.bet.board[static_cast<size_t>(i)];
            for (int i = 0; i < needed; ++i)
                board[s.bet.board_n + i] = deal_[static_cast<size_t>(4 + s.bet.board_n + i)];
            next.bet = betting_.advance_street(s.bet, board, s.bet.board_n + needed);
            return next;
        }
        if (!s.bet.dealt) {
            int8_t deck[52];
            for (int i = 0; i < 52; ++i) deck[i] = static_cast<int8_t>(i);
            for (int i = 0; i < 4; ++i) {
                const int j = i + static_cast<int>(rng.below(static_cast<uint32_t>(52 - i)));
                const int8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
            }
            next.hole[0] = {deck[0], deck[1]};
            next.hole[1] = {deck[2], deck[3]};
            next.bet.hole = next.hole;
            next.bet.dealt = true;
            return next;
        }

        // Reveal the next street.
        const int street = s.bet.street + 1;
        const int needed = STREET_BOARD_SIZE[static_cast<size_t>(street)] - s.bet.board_n;
        int8_t board[5];
        for (int i = 0; i < s.bet.board_n; ++i) board[i] = s.bet.board[static_cast<size_t>(i)];

        bool used[52] = {false};
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < 2; ++i) used[s.hole[static_cast<size_t>(p)][static_cast<size_t>(i)]] = true;
        for (int i = 0; i < s.bet.board_n; ++i) used[s.bet.board[static_cast<size_t>(i)]] = true;

        int8_t pool[52];
        int pool_n = 0;
        for (int c = 0; c < 52; ++c) if (!used[c]) pool[pool_n++] = static_cast<int8_t>(c);
        for (int i = 0; i < needed; ++i) {
            const int j = i + static_cast<int>(rng.below(static_cast<uint32_t>(pool_n - i)));
            const int8_t t = pool[i]; pool[i] = pool[j]; pool[j] = t;
            board[s.bet.board_n + i] = pool[i];
        }
        next.bet = betting_.advance_street(s.bet, board, s.bet.board_n + needed);
        return next;
    }

    double utility(const State& s, int player) const {
        const int opponent = 1 - player;
        const size_t fold_at = s.bet.history.find(char('0' + FOLD));
        if (fold_at != std::string::npos) {
            const int folded = who_folded(s.bet, fold_at);
            return folded == player
                 ? -static_cast<double>(s.bet.contributions[static_cast<size_t>(player)])
                 :  static_cast<double>(s.bet.contributions[static_cast<size_t>(opponent)]);
        }

        // Only the matched portion is at risk; anything beyond what the
        // opponent could cover is returned, so it nets to nothing.
        const double at_risk = static_cast<double>(
            std::min(s.bet.contributions[0], s.bet.contributions[1]));
        if (s.bet.board_n < 5)
            return at_risk * allin_edge(s, player);      // exact terminals only reach here

        int8_t mine_r[7], mine_s[7], theirs_r[7], theirs_s[7];
        fill_cards(s, player, mine_r, mine_s);
        fill_cards(s, opponent, theirs_r, theirs_s);
        const int n = 2 + s.bet.board_n;
        const int32_t mine = score_hand_7(mine_r, mine_s, n);
        const int32_t theirs = score_hand_7(theirs_r, theirs_s, n);
        if (mine > theirs) return at_risk;
        if (mine < theirs) return -at_risk;
        return 0.0;
    }

    /// P(win) - P(lose) for `player` over every runout of the board, exactly.
    ///
    /// Any board from none to five cards. It used to handle only one card to
    /// come or two: a preflop board fell into the two-card loop and left three
    /// of the seven slots unset, and a full board wrote one past the end of a
    /// seven-card array. The game never reached either (exact terminals start
    /// at the flop, and the river is scored directly in `utility`), but the
    /// test binding did, and on 25 September it read T7s against AK as 70% and
    /// crashed. Preflop is 1,712,304 runouts.
    double allin_edge(const State& s, int player) const {
        const int opponent = 1 - player;
        if (s.bet.board_n < 0 || s.bet.board_n > 5 || s.bet.board_n == 1 || s.bet.board_n == 2)
            throw std::invalid_argument("allin_edge: a board has 0, 3, 4 or 5 cards, not " +
                                        std::to_string(s.bet.board_n));
        bool used[52] = {false};
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < 2; ++i) used[s.hole[static_cast<size_t>(p)][static_cast<size_t>(i)]] = true;
        for (int i = 0; i < s.bet.board_n; ++i) used[s.bet.board[static_cast<size_t>(i)]] = true;
        int8_t pool[52];
        int pool_n = 0;
        for (int c = 0; c < 52; ++c) if (!used[c]) pool[pool_n++] = static_cast<int8_t>(c);
        const int needed = 5 - s.bet.board_n;
        const int base = 2 + s.bet.board_n;
        int8_t mine_r[7], mine_s[7], theirs_r[7], theirs_s[7];
        fill_cards(s, player, mine_r, mine_s);
        fill_cards(s, opponent, theirs_r, theirs_s);
        long wins = 0, losses = 0, total = 0;
        // Deal the remaining cards as every combination of `needed` from the
        // pool, in increasing index order so each runout is counted once.
        auto deal = [&](auto&& self, int depth, int start) -> void {
            if (depth == needed) {
                const int32_t m = score_hand_7(mine_r, mine_s, 7);
                const int32_t t = score_hand_7(theirs_r, theirs_s, 7);
                wins += m > t; losses += m < t; ++total;
                return;
            }
            for (int i = start; i <= pool_n - (needed - depth); ++i) {
                mine_r[base + depth] = theirs_r[base + depth] = static_cast<int8_t>(deck_rank(pool[i]));
                mine_s[base + depth] = theirs_s[base + depth] = static_cast<int8_t>(deck_suit(pool[i]));
                self(self, depth + 1, i + 1);
            }
        };
        deal(deal, 0, 0);
        return static_cast<double>(wins - losses) / static_cast<double>(total);
    }

    /// The information-set key packed into 64 bits: the bucket in the top
    /// eight, the history in the low 54 as 18 symbols of three bits ('0' to
    /// '5' as 1 to 6, '/' as 7, 0 ending it). A string key was built and
    /// hashed on every visit, about a fifth of an iteration; the packed key
    /// decodes back to the same string for the export, so nothing downstream
    /// changes. The deepest history a cap-2 tree reaches is 16 symbols.
    using Key = uint64_t;
    static constexpr int KEY_SYMBOLS = 18;

    Key information_set(const State& s, int player) const {
        const std::string& h = s.bet.history;
        if (h.size() > static_cast<size_t>(KEY_SYMBOLS))
            throw std::length_error("history too long for a packed key: " + h);
        uint64_t code = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            const uint64_t sym = h[i] == '/' ? 7 : static_cast<uint64_t>(h[i] - '0') + 1;
            code |= sym << (3 * i);
        }
        return (static_cast<uint64_t>(bucket_for(s, player)) << 56) | code;
    }

    static std::string key_to_string(Key key) {
        std::string out = std::to_string(static_cast<int>(key >> 56)) + "|";
        uint64_t code = key & ((uint64_t(1) << 54) - 1);
        for (int i = 0; i < KEY_SYMBOLS; ++i) {
            const uint64_t sym = (code >> (3 * i)) & 7;
            if (sym == 0) break;
            out.push_back(sym == 7 ? '/' : static_cast<char>('0' + sym - 1));
        }
        return out;
    }

    /// The inverse of `key_to_string`, for a warm start from a saved pickle.
    static Key key_from_string(const std::string& text) {
        const size_t bar = text.find('|');
        if (bar == std::string::npos) throw std::invalid_argument("key without '|': " + text);
        const int bucket = std::stoi(text.substr(0, bar));
        const std::string h = text.substr(bar + 1);
        if (h.size() > static_cast<size_t>(KEY_SYMBOLS))
            throw std::length_error("history too long for a packed key: " + h);
        uint64_t code = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            const uint64_t sym = h[i] == '/' ? 7 : static_cast<uint64_t>(h[i] - '0') + 1;
            code |= sym << (3 * i);
        }
        return (static_cast<uint64_t>(bucket) << 56) | code;
    }

    size_t cache_size() const { return bucket_cache_.size(); }

    bool crn_ = false;
    bool exact_terminals_ = false;
    std::array<int8_t, 9> deal_{};

private:
    static int who_folded(const pokerbot::State& s, size_t position) {
        const std::string_view prefix = std::string_view(s.history).substr(0, position);
        int street = 0;
        size_t last_slash = std::string::npos;
        for (size_t i = 0; i < prefix.size(); ++i)
            if (prefix[i] == '/') { ++street; last_slash = i; }
        const size_t before = last_slash == std::string::npos
                            ? prefix.size() : prefix.size() - last_slash - 1;
        const int first = street == 0 ? 0 : 1;
        return (first + static_cast<int>(before)) % 2;
    }

    static void fill_cards(const State& s, int player, int8_t* ranks, int8_t* suits) {
        for (int i = 0; i < 2; ++i) {
            const int card = s.hole[static_cast<size_t>(player)][static_cast<size_t>(i)];
            ranks[i] = static_cast<int8_t>(deck_rank(card));
            suits[i] = static_cast<int8_t>(deck_suit(card));
        }
        for (int i = 0; i < s.bet.board_n; ++i) {
            const int card = s.bet.board[static_cast<size_t>(i)];
            ranks[2 + i] = static_cast<int8_t>(deck_rank(card));
            suits[2 + i] = static_cast<int8_t>(deck_suit(card));
        }
    }

    int bucket_for(const State& s, int player) const {
        const auto& hole = s.hole[static_cast<size_t>(player)];
        if (s.bet.board_n == 0) {
            const int a = hole[0], b = hole[1];
            return abstraction_.preflop[static_cast<size_t>(a) * 52 + static_cast<size_t>(b)];
        }

        // Memoised on the cards, exactly as the Python game does: without it
        // every decision on a street would recompute the same rollout.
        uint64_t key = 0;
        const int lo = std::min(hole[0], hole[1]), hi = std::max(hole[0], hole[1]);
        key = (key << 6) | static_cast<uint64_t>(lo);
        key = (key << 6) | static_cast<uint64_t>(hi);
        for (int i = 0; i < s.bet.board_n; ++i)
            key = (key << 6) | static_cast<uint64_t>(s.bet.board[static_cast<size_t>(i)]);

        auto found = bucket_cache_.find(key);
        if (found != bucket_cache_.end()) return found->second;

        int cards[2] = {hole[0], hole[1]};
        int board[5];
        for (int i = 0; i < s.bet.board_n; ++i) board[i] = s.bet.board[static_cast<size_t>(i)];
        // Seeded from the cards so a situation always buckets the same way
        // within a run. Not the Python's seed: that comes from Python's own
        // tuple hash, which is not reproducible here and need not be.
        const int street_index = s.bet.board_n - 3;
        int bucket;
        if (abstraction_.histogram()) {
            bucket = -1;
            if (street_index < 2 && !abstraction_.hist_tables[static_cast<size_t>(street_index)].empty())
                bucket = abstraction_.hist_tables[static_cast<size_t>(street_index)].lookup(cards, board, s.bet.board_n);
            if (bucket < 0) {
                double hist[64];
                strength_histogram(cards, board, s.bet.board_n, abstraction_.hist_bins,
                                   abstraction_.hist_runouts, abstraction_.hist_opponents,
                                   key * 0x9E3779B97F4A7C15ULL, hist);
                bucket = nearest_emd(abstraction_.hist_centroids[static_cast<size_t>(street_index)],
                                     hist, abstraction_.hist_bins);
            }
        } else {
            const double value = equity_vs_random(cards, 2, board, s.bet.board_n,
                                                  abstraction_.equity_samples, key * 0x9E3779B97F4A7C15ULL);
            bucket = Abstraction::nearest(
                abstraction_.centroids[static_cast<size_t>(street_index)], value);
        }
        if (abstraction_.texture) {
            // The stride is the bucket count for the street, which in histogram
            // mode is the centroid-histogram count, not the scalar list a
            // caller may or may not have passed.
            const size_t classes = abstraction_.histogram()
                ? abstraction_.hist_centroids[static_cast<size_t>(street_index)].size()
                : abstraction_.centroids[static_cast<size_t>(street_index)].size();
            bucket += static_cast<int>(classes) * Abstraction::board_texture(board, s.bet.board_n);
        }
        // Cleared wholesale at the ceiling, as games/nolimit.py does.
        //
        // This was omitted when the game was ported and it cost a run: the
        // [4,2] taper at 6.5M iterations reached 4,960 MB of resident memory on
        // 11 September and spent its time swapping at 50% CPU rather than
        // solving. The Python had the same unbounded memo earlier that day,
        // reaching 2.8 million entries and 491 MB with the information set count
        // flat -- the memo is keyed on (hole, board), a space of roughly 1,326 x
        // 2.1 million, so it never saturates.
        //
        // Safe because `bucket_for` is a pure function of the cards: the seed is
        // derived from them, so dropping the memo costs recomputation, never a
        // different answer.
        if (bucket_cache_.size() >= BUCKET_CACHE_LIMIT) bucket_cache_.clear();
        bucket_cache_.emplace(key, bucket);
        return bucket;
    }

    Abstraction abstraction_;
    NoLimitHoldem betting_;
    mutable std::unordered_map<uint64_t, int> bucket_cache_;
};

}  // namespace pokerbot
