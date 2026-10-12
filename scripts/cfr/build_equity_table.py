"""
Precompute equity for every suit-isomorphic postflop situation.

Why
---
`docs/optimisation-plan.md` measured two problems that are one problem. The
solver computes equity *during traversal*, about 43 times per iteration, at 40
Monte Carlo samples. That is 44% of runtime, and the estimate's standard
deviation is 0.0667 against bucket centroids spaced 0.101 to 0.170 apart, so 42%
of situations land in the wrong bucket. We compute the wrong bucket, quickly.

Precomputing fixes both, because a table built once can afford a sample count no
traversal could pay per node. At 1,000 samples the standard deviation falls to
0.0134, a tenth of a bucket width instead of a half.

Why equity and not buckets
--------------------------
A bucket table would be specific to one bucket count and one fitted clustering.
Equity is neither, so the same table serves any abstraction, lets centroids be
refitted without recomputing anything, and makes re-running the bucket sweep
free. `float32` carries far more precision than 1,000 samples justify, and the
whole flop table is 5.1 MB.

Sizes, measured rather than estimated
-------------------------------------
    flop     25,989,600 raw  ->   1,286,792 canonical   5.1 MB
    turn    305,377,800 raw  ->  ~15,100,000 canonical  60.5 MB

1,286,792 is the published count of suit-isomorphic hole-plus-flop combinations,
which is what checks `abstraction.canonical` against the literature.

The river needs no table. With a complete board, equity against a random hand is
an exact enumeration over 990 opponent holdings, which is cheaper than sampling
it and carries no error at all. That is a separate change.

Usage
-----
    venv/bin/python scripts/cfr/build_equity_table.py --street flop
    venv/bin/python scripts/cfr/build_equity_table.py --street flop --samples 200 --limit 20000
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.canonical import canonical_key, unpack_key  # noqa: E402
from abstraction.equity import DECK_RANKS, DECK_SUITS, _rollouts  # noqa: E402
from engine.hand_eval_fast import jit  # noqa: E402

#: Board cards by street name. The river is absent deliberately: see the module
#: docstring.
BOARD_SIZE = {"flop": 3, "turn": 4}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "cfr", "equity_tables")


@jit(nopython=True, cache=True)
def enumerate_keys(board_size, out):
    """Every raw (hole, board) situation's canonical key, with duplicates."""
    hole = np.empty(2, dtype=np.int64)
    board = np.empty(board_size, dtype=np.int64)
    picks = np.empty(board_size, dtype=np.int64)
    n = 0
    for first in range(52):
        for second in range(first + 1, 52):
            hole[0] = first
            hole[1] = second
            # Board combinations in lexicographic order, skipping the hole.
            for i in range(board_size):
                picks[i] = i
            while True:
                ok = True
                for i in range(board_size):
                    if picks[i] == first or picks[i] == second:
                        ok = False
                        break
                if ok:
                    for i in range(board_size):
                        board[i] = picks[i]
                    out[n] = canonical_key(hole, board)
                    n += 1
                # Advance the combination.
                i = board_size - 1
                while i >= 0 and picks[i] == 52 - board_size + i:
                    i -= 1
                if i < 0:
                    break
                picks[i] += 1
                for j in range(i + 1, board_size):
                    picks[j] = picks[j - 1] + 1
    return n


@jit(nopython=True, cache=True)
def equity_for_keys(keys, board_size, samples, seed, out):
    """
    Equity against one random opponent for each canonical key.

    Seeded per situation from its own key rather than from a running stream, so
    a chunk's result does not depend on which chunk it landed in or on how many
    workers ran. Splitting the work must not change the table.
    """
    hole = np.empty(2, dtype=np.int64)
    board = np.empty(board_size, dtype=np.int64)
    available = np.empty(52 - 2 - board_size, dtype=np.int64)
    hole_r = np.empty(2, dtype=np.int32)
    hole_s = np.empty(2, dtype=np.int32)
    board_r = np.empty(board_size, dtype=np.int32)
    board_s = np.empty(board_size, dtype=np.int32)

    for i in range(keys.shape[0]):
        unpack_key(keys[i], hole, board)

        n = 0
        for card in range(52):
            seen = False
            for j in range(2):
                if hole[j] == card:
                    seen = True
            for j in range(board_size):
                if board[j] == card:
                    seen = True
            if not seen:
                available[n] = card
                n += 1

        for j in range(2):
            hole_r[j] = DECK_RANKS[hole[j]]
            hole_s[j] = DECK_SUITS[hole[j]]
        for j in range(board_size):
            board_r[j] = DECK_RANKS[board[j]]
            board_s[j] = DECK_SUITS[board[j]]

        draw = 2 + (5 - board_size)
        out[i] = _rollouts(hole_r, hole_s, board_r, board_s, available,
                           samples, draw,
                           (keys[i] + seed) % np.int64(2 ** 31 - 1),
                           DECK_RANKS, DECK_SUITS)


def _worker(job):
    keys, board_size, samples, seed = job
    out = np.empty(keys.shape[0], dtype=np.float64)
    equity_for_keys(keys, board_size, samples, seed, out)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--street", choices=sorted(BOARD_SIZE), default="flop")
    parser.add_argument("--samples", type=int, default=1000,
                        help="rollouts per situation; 1,000 gives sd 0.0134 "
                             "against bucket spacing of 0.101 to 0.170")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2),
                        help="processes. Half the cores by default, because "
                             "this machine shares with other training jobs")
    parser.add_argument("--limit", type=int,
                        help="build only the first N situations, for checking "
                             "the wiring without paying for the whole table")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    board_size = BOARD_SIZE[args.street]
    raw = {3: 25_989_600, 4: 305_377_800}[board_size]

    print(f"{args.street}: enumerating {raw:,} raw situations", flush=True)
    started = time.perf_counter()
    scratch = np.empty(raw, dtype=np.int64)
    count = enumerate_keys(board_size, scratch)
    keys = np.unique(scratch[:count])
    del scratch
    print(f"  {len(keys):,} canonical ({raw / len(keys):.2f}x collapse) "
          f"in {time.perf_counter() - started:.0f}s", flush=True)

    if args.limit:
        keys = keys[:args.limit]
        print(f"  limited to {len(keys):,} situations")

    chunks = np.array_split(keys, max(args.workers * 4, 1))
    jobs = [(chunk, board_size, args.samples, args.seed) for chunk in chunks]

    print(f"  {args.samples:,} samples each across {args.workers} workers",
          flush=True)
    started = time.perf_counter()
    if args.workers == 1:
        parts = [_worker(job) for job in jobs]
    else:
        with Pool(args.workers) as pool:
            parts = pool.map(_worker, jobs)
    equity = np.concatenate(parts)
    elapsed = time.perf_counter() - started
    print(f"  built in {elapsed / 60:.1f} min "
          f"({len(keys) / elapsed:,.0f} situations/s)", flush=True)

    print(f"  equity: min {equity.min():.3f} mean {equity.mean():.3f} "
          f"max {equity.max():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.street}_{args.samples}")
    np.save(f"{stem}_keys.npy", keys)
    np.save(f"{stem}_equity.npy", equity)
    print(f"wrote {stem}_keys.npy ({keys.nbytes / 1e6:.1f} MB) and "
          f"{stem}_equity.npy ({equity.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
