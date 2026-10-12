# Solver and play-path engineering audit, measured, ranked by gain per hour

Written 15 September 2026 by a read-only audit agent; every figure was measured on this
machine (8 cores, 8 GB). Baselines: native cap-2 solver, 200 equity samples, texture on,
1.432 ms per iteration; 0.551 at 40 samples; 0.252 at 1 sample. Per iteration the cap-2 tree
costs 268 decision nodes, 91 chance nodes, 210 terminals, 567 `next_state`, 1,937
`current_player`, 4,052 `street_actions`, and 267 postflop bucket lookups of which 71.9 are
distinct. Native `equity_vs_random` costs 15.6 µs at 200 samples.

## Tier 1: bit-exact, hours each

1. **`street_actions` returns a fresh `std::string` on every call** (`native/src/nolimit.hpp:94-97`,
   4,052 calls per iteration against a 252 µs traversal budget). Return a `string_view` or cache
   a street-start index in `State`. Same for `utility`'s `find` and `who_folded`'s `substr`
   (`nolimit_game.hpp:152,185`), 210 per iteration. Gain 1.3 to 2x on traversal time (20 to 35
   percent of a 200-sample iteration). 2 hours. Correctness-preserving: fixed-seed equality of
   `average_strategy()`.
2. **The post-recursion refetch is not needed** (`mccfr.hpp:209`): rehashing invalidates
   iterators, not references. Also add `nodes_.reserve()`. 2 to 4 percent plus the rehash
   spikes. Half an hour. Byte-identical.
3. **The bucket memo keeps 500,000 entries and never re-uses one** (`nolimit_game.hpp:72,224-256`):
   across 50 iterations the distinct set equals the misses, hit rate about zero across
   iterations. A memo of a few thousand entries gives the same answers; `cache_size()` is not
   bound to Python, so the rate had never been measured. 24 MB and a few percent. Half an hour.
4. **The export triples peak memory** (`bindings.cpp:131-136` builds a `std::map` then a dict,
   then `train_nolimit.py:180` makes 2.3M numpy arrays): at 655,381 nodes, +253 MB and 1.91 s;
   scaled to the contender about 1.1 GB transient. Stream into a dict or emit flat arrays.
   1.5 hours. Correctness-preserving.
5. **Packed integer info-set keys**: max key 20 chars, bucket up to 168, history codes fit 50
   bits at 3 bits per symbol, so `(bucket << 56) | code` fits a `uint64`. 10 to 15 percent of
   traversal, 30 percent of the node table. 3 hours. Bijective, correctness-preserving.

## Tier 2: the two structural wins

6. **Flat strategy table.** `ladder169l/cap2_100bb.pkl`: 159.8 MB on disk, 2,263,275 entries,
   5.49 s to load, 1,465 MB resident; widths {2: 1,750,913, 5: 257,810, 6: 254,552} = 6.32M
   floats. A sorted `uint64` key array, `int32` row offsets and one contiguous `float32` array is
   52 to 61 MB and loads at the measured 730 MB/s in about 70 ms: 24 to 28x smaller, 78x faster.
   Lookup by `searchsorted` is 2.25 µs against 0.99 today, and the lookup is 9 percent of `act`,
   which is 51 percent of a hand, so about 1 percent net. Ship as a `Mapping` shim with `.get`
   so `cfr_agent`, `chipzen.player.load_solver`, `play_pickles.py`, `cfr/river.py` and the ten
   other `saved["strategy"]` sites are unchanged. 6 to 8 hours. Test: the shim reproduces the
   dict entry for entry, and `benchmark()` returns an identical chip series at a fixed seed.
7. **Common random numbers: one deal per iteration.** `mccfr.hpp:173-174` re-enters
   `sample_chance` at every chance node on every branch: about 91 runouts per iteration. One
   deal per iteration, prefixes revealed, cuts the rollouts to at most 6: 1.43 to about 0.28 ms
   per iteration, 5.1x at 200 samples. Convergence-preserving (each branch's estimator stays
   unbiased) but path-changing: Kuhn −1/18 and Leduc exploitability as the anchor, plus one
   same-tree head-to-head. 4 to 6 hours.

## Parallelism, terminals, play time

8. **Eight cores on one solve.** What breaks: `nodes_` under concurrent insert, the single
   `rng_`, the shared `bucket_cache_`, the read-modify-write in `discount_once`. The GIL is
   already released. Smallest change: the `uint64` key, a pre-sized open-addressed table with
   CAS insert into a node arena, per-thread `Rng` and `NoLimitGame` (the tables are const, 5 KB),
   relaxed atomic accumulation Hogwild-style as Pluribus did. 6 to 7x. 8 to 16 hours. Path-
   changing (interleaving): Kuhn value as the acceptance test. Zero-risk alternative today:
   rungs in parallel processes, but at 1.5 GB per rung only 3 or 4 fit in 8 GB.
9. **Exact terminals.** An all-in ends the hand only at five board cards (`nolimit.hpp:117`), so
   `utility` (`nolimit_game.hpp:150-174`) scores one sampled runout: a single Bernoulli draw of the
   all-in equity, the dominant variance at the short rungs. Exact: turn 44 runouts (about 2 µs),
   flop 990 (about 30 µs), preflop by table only (a suit-aware one has to be built; the existing
   169 x 169 is class-level and sampled). Do preflop by table and turn by enumeration. 4 hours.
   Lower variance, path-changing.
10. **Play time.** `act` is 51 percent of a benchmark hand, `bucket_for` 41 percent, the numba
    rollout 23 percent, and `np.random.default_rng(...)` costs 9.14 µs per construction
    (`benchmark.py:196-198`) inside a 75.8 µs `abstraction.bucket`; `chipzen/player.py:341-347`
    re-derives that bucket up to four more times per decision with no memo. A full ladder loads at
    29 MB/s of pickle (9 to 18 s, 2.4 to 4.9 GB), which item 6 fixes. `warm_up` 0.22 s warm;
    steady-state `decide` 0.05 ms. River: `HandSet.build` is 0.39 s cold and 4 ms warm; the real
    cost is `blueprint_ranges` (0.29 of 1.51 s) bucketing 1,081 hands per street. Memoise
    `HandSet` and the per-board bucket arrays (correctness-preserving); a precomputed incidence
    matmul in `showdown`/`compatible_mass` is worth about 5x on the 1.2 s solve.

## Tests

Existing Python-to-native pins: `tests/test_native.py:24` (evaluator, exact), `:50` (rollout
bias), `:74` (betting tree, four schedules), `:141` (Kuhn analytic). Needed for items 1 to 6:
a native golden test, 2,000 cap-2 iterations at a fixed seed with `average_strategy()`
byte-identical before and after each bit-exact change; and `tests/test_benchmark.py:142`
extended to assert the flat table and the dict produce the same action series.
