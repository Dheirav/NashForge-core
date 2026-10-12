"""
Write the native solver's golden output, the reference every engineering
change is measured against.

    venv/bin/python scripts/cfr/make_golden.py            # refuses to overwrite
    venv/bin/python scripts/cfr/make_golden.py --force    # after a change that is MEANT to alter the path

A small raise-cap-2 game at 20bb with a seeded texture-aware abstraction,
solved for 2,000 iterations from seed 7, average strategy written as JSON with
Python's exact float repr. `tests/test_native.py` re-runs the same solve and
requires every entry to match to the last bit. A change that only alters the
computation (string copies, packed keys, a reserved table, a streamed export)
leaves the file identical; one that alters the path (one deal per iteration,
exact terminals, threads) does not, and must be re-baselined deliberately with
--force after its own convergence tests have passed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.buckets import CardAbstraction  # noqa: E402
from scripts.cfr.train_nolimit import _native_tables  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN = os.path.join(ROOT, "tests", "golden", "native_cap2_20bb_seed7_2000.json")
SPEC = {"stack": 40, "big_blind": 2, "raise_cap": 2, "buckets": 6, "preflop_buckets": 169,
        "samples": 200, "equity_samples": 40, "texture": True, "iterations": 2000, "seed": 7,
        "rule": "linear"}


def solve(spec=SPEC):
    import pokerbot_native
    abstraction = CardAbstraction(preflop_buckets=spec["preflop_buckets"], postflop_buckets=spec["buckets"],
                                  samples=spec["samples"], equity_samples=spec["equity_samples"],
                                  texture=spec["texture"]).fit(np.random.default_rng(spec["seed"]))
    preflop, flop, turn, river = _native_tables(abstraction)
    solver = pokerbot_native.NoLimitSolver(preflop, flop, turn, river, spec["equity_samples"], spec["stack"],
                                           spec["big_blind"] // 2, spec["big_blind"], [4] * spec["raise_cap"],
                                           spec["seed"], texture=spec["texture"], rule=spec["rule"])
    solver.train(spec["iterations"])
    return {key: [float(p) for p in value] for key, value in solver.average_strategy().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if os.path.exists(GOLDEN) and not args.force:
        sys.exit(f"{GOLDEN} exists; --force to re-baseline after a deliberate path change")
    strategy = solve()
    with open(GOLDEN, "w") as handle:
        json.dump({"spec": SPEC, "entries": len(strategy), "strategy": strategy}, handle, sort_keys=True)
    print(f"wrote {GOLDEN}: {len(strategy):,} information sets")


if __name__ == "__main__":
    main()
