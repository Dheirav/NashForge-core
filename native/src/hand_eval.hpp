// Seven-card hand scoring, packed into one comparable integer.
//
// A direct translation of engine/hand_eval_fast.py's score_hand_7_fast, and
// deliberately so: `tests/test_native.py` pins the two against each other over
// every hand class. This is a port, and a port that quietly disagrees with the
// thing it replaced is worse than no port, because it would shift every equity
// estimate rather than fail.
//
// Score layout matches Python exactly: class in bits 20+, then five 4-bit
// tiebreakers. HandEvalResult compares lexicographically on (class, tiebreaks)
// and the tiebreak count is fixed within a class, so integer order reproduces it.
#pragma once
#include <cstdint>
#include <array>

namespace pokerbot {

struct RankTables {
    std::array<int8_t, 8192> straight_high{};
    std::array<int32_t, 8192> top_five{};

    RankTables() {
        for (int mask = 0; mask < 8192; ++mask) {
            int high = -1;
            for (int v = 12; v >= 4; --v) {
                bool run = true;
                for (int s = 0; s < 5; ++s)
                    if (!(mask & (1 << (v - s)))) { run = false; break; }
                if (run) { high = v; break; }
            }
            // The wheel, A-2-3-4-5, which the descending scan cannot find.
            if (high < 0 && (mask & 1) && (mask & 2) && (mask & 4) && (mask & 8)
                && (mask & (1 << 12)))
                high = 3;
            straight_high[mask] = static_cast<int8_t>(high);

            int32_t packed = 0, taken = 0;
            for (int r = 12; r >= 0 && taken < 5; --r)
                if (mask & (1 << r)) { packed = (packed << 4) | r; ++taken; }
            for (; taken < 5; ++taken) packed <<= 4;
            top_five[mask] = packed;
        }
    }
};

inline const RankTables& tables() {
    static const RankTables t;
    return t;
}

inline int32_t score_hand_7(const int8_t* ranks, const int8_t* suits, int n = 7) {
    const RankTables& t = tables();
    int rank_counts[13] = {0};
    int suit_masks[4] = {0};
    int suit_counts[4] = {0};
    int mask = 0;
    for (int i = 0; i < n; ++i) {
        const int r = ranks[i], s = suits[i];
        ++rank_counts[r];
        suit_masks[s] |= 1 << r;
        ++suit_counts[s];
        mask |= 1 << r;
    }

    int flush = -1;
    for (int s = 0; s < 4; ++s)
        if (suit_counts[s] >= 5) { flush = s; break; }

    if (flush >= 0) {
        const int high = t.straight_high[suit_masks[flush]];
        if (high >= 0) return ((high == 12 ? 9 : 8) << 20) | (high << 16);
    }

    int quad = -1, trip_hi = -1, trip_lo = -1, pair_hi = -1, pair_lo = -1;
    for (int v = 12; v >= 0; --v) {
        const int c = rank_counts[v];
        if (c == 4 && quad < 0) quad = v;
        else if (c == 3) { if (trip_hi < 0) trip_hi = v; else if (trip_lo < 0) trip_lo = v; }
        else if (c == 2) { if (pair_hi < 0) pair_hi = v; else if (pair_lo < 0) pair_lo = v; }
    }

    if (quad >= 0)
        return (7 << 20) | (quad << 16) | ((t.top_five[mask & ~(1 << quad)] >> 16) << 12);

    if (trip_hi >= 0) {
        // A second trip plays as the pair and outranks any real pair, because
        // the scan above runs high to low.
        const int pair = trip_lo > pair_hi ? trip_lo : pair_hi;
        if (pair >= 0) return (6 << 20) | (trip_hi << 16) | (pair << 12);
    }

    if (flush >= 0) return (5 << 20) | t.top_five[suit_masks[flush]];

    const int high = t.straight_high[mask];
    if (high >= 0) return (4 << 20) | (high << 16);

    if (trip_hi >= 0) {
        const int k = t.top_five[mask & ~(1 << trip_hi)];
        return (3 << 20) | (trip_hi << 16) | (((k >> 16) & 15) << 12) | (((k >> 12) & 15) << 8);
    }
    if (pair_hi >= 0 && pair_lo >= 0) {
        const int k = t.top_five[mask & ~(1 << pair_hi) & ~(1 << pair_lo)] >> 16;
        return (2 << 20) | (pair_hi << 16) | (pair_lo << 12) | (k << 8);
    }
    if (pair_hi >= 0) {
        const int k = t.top_five[mask & ~(1 << pair_hi)];
        return (1 << 20) | (pair_hi << 16) | (((k >> 16) & 15) << 12)
             | (((k >> 12) & 15) << 8) | (((k >> 8) & 15) << 4);
    }
    return t.top_five[mask];
}

}  // namespace pokerbot
