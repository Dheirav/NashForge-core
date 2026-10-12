"""
Solve a rung full-width on the explicit abstract game and write the usual pickle.

    venv/bin/python scripts/cfr/fullwidth_solve.py results/cfr/ladder169l/nolimit_8bb.pkl \\
        --raise-cap 2 --iterations 1000 --output results/cfr/experiments/fw_cap2_8bb.pkl

The rung pickle supplies the abstraction (so the export buckets exactly as the
ladder does) and the stack; the chance tables come from
`results/cfr/chance/<rung>.npz` (`scripts/cfr/chance_tables.py`). The output
is the same dict every other solver writes, so `tools/xtree-gate.sh`, the
replay and the arena player read it unchanged. See `cfr/fullwidth.py` for what
the solver is and the one approximation it adds.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cfr.flat import load_strategy  # noqa: E402
from cfr.fullwidth import ChanceTables, FullWidthSolver  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rung", help="a ladder pickle: its abstraction and stack are used")
    parser.add_argument("--tables", help="chance tables .npz (default results/cfr/chance/<rung>.npz)")
    parser.add_argument("--raise-cap", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    saved = load_strategy(args.rung)
    abstraction, rung_args = saved["abstraction"], saved["args"]
    tables = args.tables or os.path.join(ROOT, "results", "cfr", "chance",
                                         os.path.basename(args.rung).replace(".pkl", ".npz"))
    stack, bb = int(rung_args["stack"]), int(rung_args["big_blind"])
    allin_path = tables.replace(".npz", "_preflop_allin.npy")
    allin = np.load(allin_path) if os.path.exists(allin_path) else None
    print(f"preflop all-in table: {'loaded' if allin is not None else 'none, factorised runout'}", flush=True)
    solver = FullWidthSolver(ChanceTables(tables), abstraction, stack, bb // 2, bb, args.raise_cap,
                             preflop_allin=allin)
    print(f"full-width CFR+ on {os.path.basename(args.rung)}'s abstraction, stack {stack}, "
          f"raise cap {args.raise_cap}, {args.iterations:,} iterations", flush=True)

    def progress(done, total, taken):
        step = max(1, total // 50)
        if done % step == 0 or done == total:
            # ETA from the measured rate, never estimated up front.
            print(f"  {done:>6,}/{total:,}  {taken / done:6.2f} s/it  {len(solver.regret):>9,} infosets  "
                  f"eta {(total - done) * taken / done / 60:6.1f} min", flush=True)

    seconds = solver.train(args.iterations, on_progress=progress)
    strategy = solver.average_strategy()
    out = {"strategy": strategy, "abstraction": abstraction,
           "args": {**{k: v for k, v in rung_args.items() if k in ("stack", "big_blind", "buckets",
                                                                      "preflop_buckets", "texture",
                                                                      "equity_samples")},
                    "raise_cap": args.raise_cap, "iterations": args.iterations,
                    "solver": "fullwidth-cfr+", "tables": os.path.relpath(tables, ROOT),
                    "output": args.output},
           "results": {}, "information_sets_reached": len(strategy), "seconds": seconds}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(out, handle)
    with open(args.output.replace(".pkl", ".json"), "w") as handle:
        import json
        json.dump({k: v for k, v in out.items() if k not in ("strategy", "abstraction")}, handle, indent=1)
    print(f"wrote {args.output}: {len(strategy):,} information sets in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
