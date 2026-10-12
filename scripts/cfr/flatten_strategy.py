"""
Write a solver pickle's strategy as flat arrays beside it.

    venv/bin/python scripts/cfr/flatten_strategy.py results/cfr/ladder169l/*.pkl
    venv/bin/python scripts/cfr/flatten_strategy.py --float32 results/cfr/contender/cap2_200bb.pkl

See `cfr/flat.py` for why: the dict-of-arrays pickle is 1.5 GB in a process for
a big rung and the flat pair is under 120 MB, read through the same `.get`.
Every loader prefers the flat pair when it is newer than the pickle, so this is
safe to run on a live set: nothing changes until the pair exists, and the
answers are pinned equal by `tests/test_flat.py`. One pickle is in memory at a
time, so a whole ladder converts within the memory a training run leaves.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from cfr.flat import flat_paths, write_flat  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickles", nargs="+")
    parser.add_argument("--float32", action="store_true", help="halve the values; not bit-identical")
    parser.add_argument("--force", action="store_true", help="rewrite a pair that is already newer than its pickle")
    args = parser.parse_args()
    for path in args.pickles:
        npz, _ = flat_paths(path)
        if not args.force and os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(path):
            print(f"{path}: flat pair is current, skipping")
            continue
        started = time.perf_counter()
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        loaded = time.perf_counter() - started
        out = write_flat(path, saved, np.float32 if args.float32 else np.float64)
        print(f"{path}: {len(saved['strategy']):,} entries, pickle {os.path.getsize(path) / 1e6:.0f} MB "
              f"loaded in {loaded:.1f}s -> {os.path.getsize(out) / 1e6:.0f} MB flat in "
              f"{time.perf_counter() - started - loaded:.1f}s")
        del saved


if __name__ == "__main__":
    main()
