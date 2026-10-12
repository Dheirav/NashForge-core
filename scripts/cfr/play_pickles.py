"""
Two solver pickles, head to head, in chips.

    venv/bin/python scripts/cfr/play_pickles.py A.pkl B.pkl --hands 40000 --seeds 0 1 2

Each agent buckets with its own abstraction and both walk the same betting
tree, so this is the clean comparison for a change to the *card* abstraction:
sample count, bucket count, centroids. It is not the comparison for a change
to the betting tree, where the coarser agent has no entries for the richer
one's histories and the result mostly measures its fallback.

Reported as BB/100 to the first pickle with the standard error across seeds,
because a chip count over a few thousand hands is variance, and 40,000 hands
against the same opponent gives roughly ±13 BB/100 here. Seats alternate
inside `benchmark`.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from cfr.flat import load_strategy  # noqa: E402
from evaluation.benchmark import benchmark, cfr_agent  # noqa: E402
from abstraction.betting import StreetSchedule, schedule_from_args  # noqa: E402


def load(path, seed, raise_cap, on_miss="random", purify="none"):
    saved = load_strategy(path)
    args = saved.get("args") or {}
    if not isinstance(args, dict):
        args = vars(args)                # older pickles stored the argparse Namespace
    # The stack and blinds the rung was solved for. Until 19 September every
    # gate played the benchmark's default 200 chips whatever the rung, so a
    # cross-tree gate at 18bb was two 18bb trees playing 100bb poker. A pickle
    # that does not say is refused rather than quietly played at 200/2 again.
    if "stack" not in args or "big_blind" not in args:
        raise SystemExit(f"{path}: no stack/big_blind in its args; refusing to guess the game")
    cap = args.get("raise_cap", raise_cap)
    cap = schedule_from_args(cap)
    misses = [0, 0]
    stack = int(args["stack"])
    bb = int(args["big_blind"])
    return (cfr_agent(saved["strategy"], saved["abstraction"], np.random.default_rng(seed),
                      misses=misses, raise_cap=cap, on_miss=on_miss, stack_cap=True, purify=purify),
            cap, len(saved["strategy"]), misses, (stack, bb))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--hands", type=int, default=40000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--raise-cap", type=int, default=1, help="if the pickle does not say")
    parser.add_argument("--on-miss", default="random", choices=["random", "call"],
                        help="what a lookup miss plays; `call` for a cross-tree gate, where a "
                             "random shove on an unvisited line measures the fallback, not the solve")
    # A set is gated as it plays. Purification changes every mixed node, so a purified set's gate is not the
    # gate of the solve it came from (a purified set once had only the unpurified gate).
    parser.add_argument("--purify", default="none", help="how the FIRST pickle plays (cfr/purify.py modes)")
    parser.add_argument("--output")
    args = parser.parse_args()

    scores = []
    miss_rates = []
    for seed in args.seeds:
        a, cap_a, n_a, miss_a, (stack, bb) = load(args.first, seed, args.raise_cap, args.on_miss, args.purify)
        b, cap_b, n_b, miss_b, (stack_b, bb_b) = load(args.second, seed + 1000, args.raise_cap, args.on_miss)
        if (stack, bb) != (stack_b, bb_b):
            raise SystemExit(f"the pickles were solved for different stacks: {stack}/{bb} and {stack_b}/{bb_b}")
        if seed == args.seeds[0]:
            print(f"{os.path.basename(args.first)}: {n_a:,} infosets, cap {cap_a}; "
                  f"{os.path.basename(args.second)}: {n_b:,} infosets, cap {cap_b}; on miss: {args.on_miss}; "
                  f"first plays purify={args.purify}; "
                  f"stack {stack} at blinds {bb // 2}/{bb}", flush=True)
        started = time.perf_counter()
        # Each side narrowed to its own tree: until 19 September the benchmark's
        # single cap (its default, one raise) applied to both, so a cap-2 solve
        # could never re-raise in the very gate meant to measure its re-raises.
        result = benchmark(a, b, os.path.basename(args.second), hands=args.hands, seed=seed,
                           starting_stack=stack, small_blind=bb // 2, big_blind=bb,
                           raise_caps=(cap_a, cap_b))
        scores.append(result.bb_per_100)
        miss_rates.append((miss_a[0] / max(1, miss_a[1]), miss_b[0] / max(1, miss_b[1])))
        print(f"  seed {seed}: {result.bb_per_100:+8.1f} BB/100 to the first, "
              f"{args.hands:,} hands in {(time.perf_counter() - started) / 60:.1f} min; "
              f"misses {100 * miss_rates[-1][0]:.2f}% / {100 * miss_rates[-1][1]:.2f}% of decisions", flush=True)

    mean = float(np.mean(scores))
    stderr = float(np.std(scores, ddof=1)) / len(scores) ** 0.5 if len(scores) > 1 else float("nan")
    print(f"\n{os.path.basename(args.first)} vs {os.path.basename(args.second)}: "
          f"{mean:+.1f} ± {stderr:.1f} BB/100 (SE across {len(scores)} seeds, {args.hands:,} hands each)")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"first": args.first, "second": args.second, "hands": args.hands,
                       "seeds": args.seeds, "bb_per_100": scores, "mean": mean, "stderr": stderr,
                       "on_miss": args.on_miss, "purify_first": args.purify, "stack": stack, "big_blind": bb, "stack_cap": True,
                       # A per-street schedule is written the way the trainer saves it, so it reads back.
                       "raise_caps": [c.for_saving() if isinstance(c, StreetSchedule) else c for c in (cap_a, cap_b)],
                       "miss_rate": [float(np.mean([m[0] for m in miss_rates])),
                                     float(np.mean([m[1] for m in miss_rates]))],
                       "measured": time.strftime("%Y-%m-%d %H:%M")}, handle, indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
