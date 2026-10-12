// Equity against one random opponent, by Monte Carlo.
//
// Matches abstraction/equity.py's `_rollouts` in method, not in stream. The
// Python draws from numba's Mersenne Twister; reproducing that byte for byte in
// C++ would buy nothing, because a rollout is sampling and what has to hold is
// the distribution, not the sequence. The deterministic parts of this port --
// the evaluator, the game's transitions and payoffs -- are pinned exactly
// instead, which is where this project has actually been bitten.
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <array>
#include "hand_eval.hpp"

namespace pokerbot {

//: Deck index is suit * 13 + rank, the same convention as the Python.
inline constexpr int deck_rank(int card) { return card % 13; }
inline constexpr int deck_suit(int card) { return card / 13; }

// xoshiro256++: small, fast, and good enough for rollouts. Seeded through
// splitmix64 so that adjacent seeds -- which is exactly what a key-derived seed
// produces -- do not give correlated streams.
class Rng {
public:
    explicit Rng(uint64_t seed) {
        for (uint64_t& word : s_) {
            seed += 0x9E3779B97F4A7C15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            word = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    /// Uniform in [0, bound). Lemire's method: one multiply, no modulo bias in
    /// practice for the small bounds a deck needs.
    uint32_t below(uint32_t bound) {
        return static_cast<uint32_t>((next() >> 32) * bound >> 32);
    }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
};

/// Probability `hole` beats one random opponent hand, ties counted half.
///
/// `board` holds 0, 3, 4 or 5 cards. The remaining deck is walked in ascending
/// card order and drawn from by partial Fisher-Yates, matching the Python.
inline double equity_vs_random(const int* hole, int hole_n,
                               const int* board, int board_n,
                               int samples, uint64_t seed) {
    int pool[52];
    int pool_n = 0;
    for (int card = 0; card < 52; ++card) {
        bool known = false;
        for (int i = 0; i < hole_n; ++i) if (hole[i] == card) known = true;
        for (int i = 0; i < board_n; ++i) if (board[i] == card) known = true;
        if (!known) pool[pool_n++] = card;
    }

    const int runout = 5 - board_n;
    const int draw = 2 + runout;

    int8_t mine_r[7], mine_s[7], opp_r[7], opp_s[7];
    for (int i = 0; i < hole_n; ++i) {
        mine_r[i] = static_cast<int8_t>(deck_rank(hole[i]));
        mine_s[i] = static_cast<int8_t>(deck_suit(hole[i]));
    }
    for (int i = 0; i < board_n; ++i) {
        mine_r[2 + i] = opp_r[2 + i] = static_cast<int8_t>(deck_rank(board[i]));
        mine_s[2 + i] = opp_s[2 + i] = static_cast<int8_t>(deck_suit(board[i]));
    }

    Rng rng(seed);
    int wins = 0, ties = 0;
    for (int sample = 0; sample < samples; ++sample) {
        for (int k = 0; k < draw; ++k) {
            const int j = k + static_cast<int>(rng.below(static_cast<uint32_t>(pool_n - k)));
            const int tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
        }
        for (int k = 0; k < runout; ++k) {
            const int card = pool[2 + k];
            mine_r[2 + board_n + k] = opp_r[2 + board_n + k] = static_cast<int8_t>(deck_rank(card));
            mine_s[2 + board_n + k] = opp_s[2 + board_n + k] = static_cast<int8_t>(deck_suit(card));
        }
        opp_r[0] = static_cast<int8_t>(deck_rank(pool[0]));
        opp_s[0] = static_cast<int8_t>(deck_suit(pool[0]));
        opp_r[1] = static_cast<int8_t>(deck_rank(pool[1]));
        opp_s[1] = static_cast<int8_t>(deck_suit(pool[1]));

        const int32_t mine = score_hand_7(mine_r, mine_s);
        const int32_t theirs = score_hand_7(opp_r, opp_s);
        if (mine > theirs) ++wins;
        else if (mine == theirs) ++ties;
    }
    return (wins + 0.5 * ties) / samples;
}

/// The distribution of river equity over sampled runouts (abstraction/histogram.py):
/// `runouts` boards completed at random, each scored against `opponents` random
/// hands drawn independently from the rest of the deck, the win rate (ties half)
/// binned into `bins` equal intervals. Same distribution as the Python, not the
/// same stream. `out` receives `bins` probabilities summing to one.
inline void strength_histogram(const int* hole, const int* board, int board_n,
                               int bins, int runouts, int opponents, uint64_t seed, double* out) {
    int pool[52];
    int pool_n = 0;
    for (int card = 0; card < 52; ++card) {
        bool known = card == hole[0] || card == hole[1];
        for (int i = 0; i < board_n; ++i) if (board[i] == card) known = true;
        if (!known) pool[pool_n++] = card;
    }
    const int need = 5 - board_n;
    const int n_runouts = need == 0 ? 1 : runouts;
    for (int b = 0; b < bins; ++b) out[b] = 0.0;

    int8_t mine_r[7], mine_s[7], opp_r[7], opp_s[7];
    mine_r[0] = static_cast<int8_t>(deck_rank(hole[0])); mine_s[0] = static_cast<int8_t>(deck_suit(hole[0]));
    mine_r[1] = static_cast<int8_t>(deck_rank(hole[1])); mine_s[1] = static_cast<int8_t>(deck_suit(hole[1]));
    for (int i = 0; i < board_n; ++i) {
        mine_r[2 + i] = opp_r[2 + i] = static_cast<int8_t>(deck_rank(board[i]));
        mine_s[2 + i] = opp_s[2 + i] = static_cast<int8_t>(deck_suit(board[i]));
    }
    Rng rng(seed);
    for (int r = 0; r < n_runouts; ++r) {
        // A partial shuffle draws the runout; the rest of the pool is the deck
        // the opponents' hands come from.
        for (int k = 0; k < need; ++k) {
            const int j = k + static_cast<int>(rng.below(static_cast<uint32_t>(pool_n - k)));
            const int tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
            mine_r[2 + board_n + k] = opp_r[2 + board_n + k] = static_cast<int8_t>(deck_rank(pool[k]));
            mine_s[2 + board_n + k] = opp_s[2 + board_n + k] = static_cast<int8_t>(deck_suit(pool[k]));
        }
        const int32_t mine = score_hand_7(mine_r, mine_s, 7);
        const int rest_n = pool_n - need;
        double wins = 0.0;
        for (int o = 0; o < opponents; ++o) {
            const int i = static_cast<int>(rng.below(static_cast<uint32_t>(rest_n)));
            int j = static_cast<int>(rng.below(static_cast<uint32_t>(rest_n - 1)));
            if (j >= i) ++j;
            opp_r[0] = static_cast<int8_t>(deck_rank(pool[need + i])); opp_s[0] = static_cast<int8_t>(deck_suit(pool[need + i]));
            opp_r[1] = static_cast<int8_t>(deck_rank(pool[need + j])); opp_s[1] = static_cast<int8_t>(deck_suit(pool[need + j]));
            const int32_t theirs = score_hand_7(opp_r, opp_s, 7);
            wins += mine > theirs ? 1.0 : (mine == theirs ? 0.5 : 0.0);
        }
        const double equity = wins / opponents;
        int bin = static_cast<int>(equity * bins);
        if (bin >= bins) bin = bins - 1;
        out[bin] += 1.0;
    }
    for (int b = 0; b < bins; ++b) out[b] /= n_runouts;
}

/// Index of the centroid nearest under 1-D earth mover's distance (L1 between
/// cumulative sums), as abstraction/histogram.py's `nearest_emd`.
inline int nearest_emd(const std::vector<std::vector<double>>& centroids, const double* hist, int bins) {
    int best = 0;
    double best_d = 1e300;
    for (size_t c = 0; c < centroids.size(); ++c) {
        double ch = 0.0, hh = 0.0, d = 0.0;
        for (int b = 0; b < bins; ++b) {
            ch += centroids[c][static_cast<size_t>(b)];
            hh += hist[b];
            d += std::fabs(ch - hh);
        }
        if (d < best_d) { best_d = d; best = static_cast<int>(c); }
    }
    return best;
}

}  // namespace pokerbot
