"""
Milestone 4 — how big is the abstract game, and what fits on a laptop?

The project's proposal treats card-abstraction granularity as the thing to
ablate, and assumes bucketing is what brings no-limit Hold'em down to a
tractable 10^4-10^7 information sets. This measures both axes exactly rather
than assuming either.

Information sets are counted from the betting tree and the bucket counts
directly — no engine, no cards, no sampling. Memory assumes one regret and one
strategy accumulator per action per information set, which is what CFR stores.

Usage
-----
    python scripts/cfr/measure_abstraction.py
    python scripts/cfr/measure_abstraction.py --budget-mb 512
    python scripts/cfr/measure_abstraction.py --fit --buckets 8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.betting import STREETS, count_decision_points, enumerate_street_sequences, measure  # noqa: E402
from abstraction.buckets import CardAbstraction  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--buckets", type=int, nargs="+", default=[4, 8, 20, 50])
    parser.add_argument("--budget-mb", type=float, default=1024.0,
                        help="memory a laptop-class run is allowed")
    parser.add_argument("--fit", action="store_true",
                        help="also fit a card abstraction and show its buckets")
    parser.add_argument("--samples", type=int, default=1500,
                        help="situations sampled per street when fitting")
    parser.add_argument("--output", help="write results as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Betting tree, per street\n")
    print(f"{'raise cap':>10}{'sequences':>12}{'survive':>10}{'decisions':>11}")
    for cap in args.caps:
        sequences = enumerate_street_sequences(cap)
        surviving = sum(1 for _, folded in sequences if not folded)
        decisions = sum(count_decision_points(cap).values())
        print(f"{cap:>10}{len(sequences):>12,}{surviving:>10,}{decisions:>11,}")

    print("\nAbstract game size. * fits the "
          f"{args.budget_mb:.0f} MB budget.\n")
    header = f"{'cap':>4}{'buckets':>9}" + f"{'infosets':>16}{'table':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for cap in args.caps:
        for count in args.buckets:
            size = measure({street: count for street in STREETS}, raise_cap=cap)
            megabytes = size.table_bytes / 1e6
            fits = "*" if megabytes <= args.budget_mb else " "
            print(f"{cap:>4}{count:>9}{size.information_sets:>16,}"
                  f"{megabytes:>10.1f} MB{fits}")
            rows.append({"raise_cap": cap, "buckets": count,
                         "information_sets": size.information_sets,
                         "megabytes": megabytes,
                         "fits_budget": megabytes <= args.budget_mb})
        print()

    # Which axis actually controls the size?
    base = measure({s: args.buckets[0] for s in STREETS}, raise_cap=args.caps[0])
    by_buckets = measure({s: args.buckets[-1] for s in STREETS}, raise_cap=args.caps[0])
    by_cap = measure({s: args.buckets[0] for s in STREETS}, raise_cap=args.caps[-1])

    print(f"Scaling from the smallest configuration:")
    print(f"  buckets {args.buckets[0]} -> {args.buckets[-1]}: "
          f"x{by_buckets.information_sets / base.information_sets:.1f}")
    print(f"  raise cap {args.caps[0]} -> {args.caps[-1]}: "
          f"x{by_cap.information_sets / base.information_sets:.1f}")
    print("\nInformation sets grow linearly in buckets but combinatorially in the\n"
          "raise cap, because betting lines multiply across all four streets.\n"
          "The bet abstraction, not the card abstraction, is the binding constraint.")

    if args.fit:
        print(f"\nFitting a card abstraction ({args.samples} situations/street)...")
        abstraction = CardAbstraction(preflop_buckets=args.buckets[1],
                                      postflop_buckets=args.buckets[1],
                                      samples=args.samples).fit(np.random.default_rng(0))
        print(abstraction.describe())

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"budget_mb": args.budget_mb, "rows": rows}, handle, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
