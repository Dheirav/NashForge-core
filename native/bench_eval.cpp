// A straight translation of engine/hand_eval_fast.py's score_hand_7_fast, to
// measure what the language is worth before committing to a port.
//
// Same algorithm, same packed score, same two rank-mask tables. Nothing clever:
// if this is not much faster than 9.6 M evals/sec, the port's case rests
// entirely on the traversal floor rather than on the card work.
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <array>

static int8_t  STRAIGHT_HIGH[8192];
static int32_t TOP_FIVE[8192];

static void build_tables() {
    for (int mask = 0; mask < 8192; ++mask) {
        int high = -1;
        for (int v = 12; v >= 4; --v) {
            bool run = true;
            for (int s = 0; s < 5; ++s) if (!(mask & (1 << (v - s)))) { run = false; break; }
            if (run) { high = v; break; }
        }
        if (high < 0 && (mask & 1) && (mask & 2) && (mask & 4) && (mask & 8) && (mask & (1 << 12)))
            high = 3;
        STRAIGHT_HIGH[mask] = (int8_t)high;

        int32_t packed = 0, taken = 0;
        for (int r = 12; r >= 0 && taken < 5; --r)
            if (mask & (1 << r)) { packed = (packed << 4) | r; ++taken; }
        for (; taken < 5; ++taken) packed <<= 4;
        TOP_FIVE[mask] = packed;
    }
}

static inline int32_t score_hand_7(const int8_t* ranks, const int8_t* suits) {
    int rank_counts[13] = {0};
    int suit_masks[4] = {0};
    int suit_counts[4] = {0};
    int mask = 0;
    for (int i = 0; i < 7; ++i) {
        int r = ranks[i], s = suits[i];
        ++rank_counts[r]; suit_masks[s] |= 1 << r; ++suit_counts[s]; mask |= 1 << r;
    }
    int flush = -1;
    for (int s = 0; s < 4; ++s) if (suit_counts[s] >= 5) { flush = s; break; }

    if (flush >= 0) {
        int high = STRAIGHT_HIGH[suit_masks[flush]];
        if (high >= 0) return ((high == 12 ? 9 : 8) << 20) | (high << 16);
    }
    int quad = -1, trip_hi = -1, trip_lo = -1, pair_hi = -1, pair_lo = -1;
    for (int v = 12; v >= 0; --v) {
        int c = rank_counts[v];
        if (c == 4 && quad < 0) quad = v;
        else if (c == 3) { if (trip_hi < 0) trip_hi = v; else if (trip_lo < 0) trip_lo = v; }
        else if (c == 2) { if (pair_hi < 0) pair_hi = v; else if (pair_lo < 0) pair_lo = v; }
    }
    if (quad >= 0) return (7 << 20) | (quad << 16) | ((TOP_FIVE[mask & ~(1 << quad)] >> 16) << 12);
    if (trip_hi >= 0) {
        int pair = trip_lo > pair_hi ? trip_lo : pair_hi;
        if (pair >= 0) return (6 << 20) | (trip_hi << 16) | (pair << 12);
    }
    if (flush >= 0) return (5 << 20) | TOP_FIVE[suit_masks[flush]];
    int high = STRAIGHT_HIGH[mask];
    if (high >= 0) return (4 << 20) | (high << 16);
    if (trip_hi >= 0) {
        int k = TOP_FIVE[mask & ~(1 << trip_hi)];
        return (3 << 20) | (trip_hi << 16) | (((k >> 16) & 15) << 12) | (((k >> 12) & 15) << 8);
    }
    if (pair_hi >= 0 && pair_lo >= 0) {
        int k = TOP_FIVE[mask & ~(1 << pair_hi) & ~(1 << pair_lo)] >> 16;
        return (2 << 20) | (pair_hi << 16) | (pair_lo << 12) | (k << 8);
    }
    if (pair_hi >= 0) {
        int k = TOP_FIVE[mask & ~(1 << pair_hi)];
        return (1 << 20) | (pair_hi << 16) | (((k >> 16) & 15) << 12)
             | (((k >> 12) & 15) << 8) | (((k >> 8) & 15) << 4);
    }
    return TOP_FIVE[mask];
}

int main() {
    build_tables();
    const int N = 2000000;
    std::vector<int8_t> ranks(N * 7), suits(N * 7);
    std::mt19937 rng(0);
    std::array<int, 52> deck;
    for (int i = 0; i < 52; ++i) deck[i] = i;
    for (int h = 0; h < N; ++h) {
        for (int i = 0; i < 7; ++i) {
            std::uniform_int_distribution<int> d(i, 51);
            std::swap(deck[i], deck[d(rng)]);
            ranks[h * 7 + i] = (int8_t)(deck[i] % 13);
            suits[h * 7 + i] = (int8_t)(deck[i] / 13);
        }
    }
    auto t0 = std::chrono::steady_clock::now();
    int64_t sink = 0;
    for (int h = 0; h < N; ++h) sink += score_hand_7(&ranks[h * 7], &suits[h * 7]);
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  C++  score_hand_7  %6.1f M evals/sec  (sink %lld)\n", N / dt / 1e6, (long long)sink);
    return 0;
}
