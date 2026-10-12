// Precomputed postflop buckets, keyed by the suit-isomorphic form of the cards.
//
// A histogram bucket costs about 0.4 ms to compute (100 runouts times 50
// opponent hands, each a seven-card evaluation), and the trainer deals new
// cards every iteration, so its memo on the exact cards misses most of the
// time: measured 21 September 2026, a 20-class histogram solve ran at 5.0 ms an
// iteration against 0.30 ms with the histogram made a hundred times cheaper.
// The whole cost was the histogram. Suits are interchangeable for hand
// strength, so the distinct situations up to a suit permutation number 1.29
// million on the flop and 14 million on the turn, small enough to compute
// once per abstraction and look up thereafter; on the river a histogram is one
// runout and costs microseconds, so it needs no table.
//
// The key is the smallest packing of the cards over the 24 suit permutations,
// with the hole pair and the board each sorted, so every situation in an
// isomorphism class reaches the same entry. Tables are sorted key arrays with
// a parallel byte of bucket, searched by bisection: 14 million entries are 126
// MB and a lookup is about 25 comparisons.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "equity.hpp"

namespace pokerbot {

namespace detail {
inline constexpr int kSuitPerms[24][4] = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
    {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}};

inline uint64_t pack_sorted(const int* hole, const int* board, int board_n) {
    int h[2] = {hole[0], hole[1]};
    if (h[0] > h[1]) std::swap(h[0], h[1]);
    int b[5];
    for (int i = 0; i < board_n; ++i) b[i] = board[i];
    std::sort(b, b + board_n);
    uint64_t key = 0;
    key = (key << 6) | static_cast<uint64_t>(h[0]);
    key = (key << 6) | static_cast<uint64_t>(h[1]);
    for (int i = 0; i < board_n; ++i) key = (key << 6) | static_cast<uint64_t>(b[i]);
    return key;
}
}  // namespace detail

/// The smallest packed form of (hole, board) over all suit permutations.
/// Two situations that differ only by a renaming of suits share it.
inline uint64_t canonical_key(const int* hole, const int* board, int board_n) {
    uint64_t best = ~0ULL;
    int h[2], b[5];
    for (const auto& perm : detail::kSuitPerms) {
        for (int i = 0; i < 2; ++i) h[i] = perm[deck_suit(hole[i])] * 13 + deck_rank(hole[i]);
        for (int i = 0; i < board_n; ++i) b[i] = perm[deck_suit(board[i])] * 13 + deck_rank(board[i]);
        best = std::min(best, detail::pack_sorted(h, b, board_n));
    }
    return best;
}

/// Unpack a key produced by `pack_sorted` / `canonical_key`.
inline void unpack_key(uint64_t key, int board_n, int* hole, int* board) {
    for (int i = board_n - 1; i >= 0; --i) { board[i] = static_cast<int>(key & 63); key >>= 6; }
    hole[1] = static_cast<int>(key & 63); key >>= 6;
    hole[0] = static_cast<int>(key & 63);
}

struct BucketTable {
    std::vector<uint64_t> keys;      // sorted canonical keys
    std::vector<uint8_t> buckets;    // bucket per key, texture not included
    bool empty() const { return keys.empty(); }
    /// The bucket for the situation, or -1 when it is not in the table.
    int lookup(const int* hole, const int* board, int board_n) const {
        const uint64_t key = canonical_key(hole, board, board_n);
        const auto it = std::lower_bound(keys.begin(), keys.end(), key);
        if (it == keys.end() || *it != key) return -1;
        return buckets[static_cast<size_t>(it - keys.begin())];
    }
};

/// Every canonical (hole, board) key for a board of `board_n` cards, sorted.
/// Enumerates the boards behind each canonical hole and keeps the
/// combinations that are their own canonical form; the work is split across
/// threads by the hole.
///
/// The hole is the high bits of the key, so the smallest key over the suit
/// permutations always carries the hole in its own smallest form: a suited
/// pair of ranks both in suit 0, or the lower rank in suit 0 and the higher
/// in suit 1. Only those 169 holes can head a canonical key, which cuts the
/// enumeration from 1,326 holes to 169 (39 million turn combinations rather
/// than 359 million) without losing a class.
inline std::vector<uint64_t> canonical_keys(int board_n, int threads) {
    std::vector<std::pair<int, int>> holes;
    for (int r1 = 0; r1 < 13; ++r1)
        for (int r2 = r1; r2 < 13; ++r2) {
            if (r1 < r2) holes.emplace_back(r1, r2);           // suited, both in suit 0
            holes.emplace_back(r1, 13 + r2);                    // offsuit or a pair: suits 0 and 1
        }
    std::vector<std::vector<uint64_t>> parts(static_cast<size_t>(std::max(threads, 1)));
    std::atomic<size_t> next{0};
    auto work = [&](size_t slot) {
        int hole[2], board[5];
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= holes.size()) return;
            hole[0] = holes[i].first; hole[1] = holes[i].second;
            // Board cards in increasing order, none equal to a hole card.
            int idx[5];
            for (int k = 0; k < board_n; ++k) idx[k] = k;
            for (;;) {
                bool ok = true;
                for (int k = 0; k < board_n; ++k) {
                    board[k] = idx[k];
                    if (board[k] == hole[0] || board[k] == hole[1]) ok = false;
                }
                if (ok) {
                    const uint64_t packed = detail::pack_sorted(hole, board, board_n);
                    if (canonical_key(hole, board, board_n) == packed) parts[slot].push_back(packed);
                }
                int k = board_n - 1;
                while (k >= 0 && idx[k] == 52 - board_n + k) --k;
                if (k < 0) break;
                ++idx[k];
                for (int m = k + 1; m < board_n; ++m) idx[m] = idx[m - 1] + 1;
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 0; t < parts.size(); ++t) pool.emplace_back(work, t);
    for (auto& th : pool) th.join();
    std::vector<uint64_t> all;
    for (auto& p : parts) all.insert(all.end(), p.begin(), p.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

/// Build the table for one street: the histogram of each canonical situation,
/// seeded from its key so the table is reproducible, and its nearest centroid
/// by earth mover's distance. `progress`, when given, receives the number done
/// so far from one thread at a time.
inline BucketTable build_bucket_table(int board_n,
                                      const std::vector<std::vector<double>>& centroids,
                                      int bins, int runouts, int opponents, int threads,
                                      std::atomic<size_t>* progress = nullptr) {
    BucketTable table;
    table.keys = canonical_keys(board_n, threads);
    table.buckets.assign(table.keys.size(), 0);
    std::atomic<size_t> next{0};
    auto work = [&]() {
        int hole[2], board[5];
        double hist[64];
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= table.keys.size()) return;
            unpack_key(table.keys[i], board_n, hole, board);
            strength_histogram(hole, board, board_n, bins, runouts, opponents,
                               table.keys[i] * 0x9E3779B97F4A7C15ULL, hist);
            table.buckets[i] = static_cast<uint8_t>(nearest_emd(centroids, hist, bins));
            if (progress && (i & 1023) == 0) progress->store(i);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(threads, 1); ++t) pool.emplace_back(work);
    for (auto& th : pool) th.join();
    if (progress) progress->store(table.keys.size());
    return table;
}

}  // namespace pokerbot
