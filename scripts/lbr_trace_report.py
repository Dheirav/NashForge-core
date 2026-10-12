"""
Where an LBR run makes its money: its sized bets by street, size and hand strength.

    venv/bin/python scripts/lbr_trace_report.py results/cfr/lbr_trace_2026-09-26/*.jsonl

The records come from `scripts/lbr_ladder.py --trace` (see `cfr/lbr.py`): one per
sized bet LBR made, with its equity against the strategy's range at that
moment, the size the strategy perceived after translation, how it answered, and
the hand's result. On 26 Sept every one-size menu lost while a menu of sizes won,
so the question is which size LBR picks for which hand, and what that earns.

A hand with several bets is counted once per bet, with the hand's whole result,
so the columns show where winning hands went, not an exact split of the chips.
"""
import argparse
import json
import math
from collections import defaultdict

STREETS = ("preflop", "flop", "turn", "river")
STRENGTH = ((0.0, 0.3, "air, under 0.3"), (0.3, 0.5, "weak, 0.3 to 0.5"), (0.5, 0.7, "medium, 0.5 to 0.7"),
            (0.7, 0.85, "strong, 0.7 to 0.85"), (0.85, 1.01, "very strong, 0.85 up"))
PERCEIVED = {2: "half pot", 3: "pot", 4: "2x pot", 5: "all in"}


def band(equity):
    return next(label for lo, hi, label in STRENGTH if lo <= equity < hi)


def cell(rows):
    n = len(rows)
    mean = sum(r["result_bb"] for r in rows) / n
    se = math.sqrt(sum((r["result_bb"] - mean) ** 2 for r in rows) / (n - 1) / n) if n > 1 else float("nan")
    folds = sum(r.get("answer") == "fold" for r in rows) / n
    return n, mean, se, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--min", type=int, default=40, help="cells with fewer records are not shown")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.paths for line in open(path)]
    print(f"{len(rows):,} sized bets from {len(args.paths)} file(s)\n")

    print("## By street and size\n")
    print("| street | size | bets | result per bet, bb | our bot folds |")
    print("|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["street"], r["size"])].append(r)
    for (street, size), g in sorted(groups.items()):
        if len(g) < args.min:
            continue
        n, mean, se, folds = cell(g)
        print(f"| {STREETS[street]} | {size:g} | {n} | {mean:+.1f} ± {se:.1f} | {100 * folds:.0f}% |")

    print("\n## By street, size and LBR's hand strength: where the money is\n")
    print("Sorted by total contribution (bets x result). The top rows are what the leak is made of.\n")
    print("| street | size | LBR holds | bets | result per bet, bb | our bot folds | total, bb |")
    print("|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["street"], r["size"], band(r["equity"]))].append(r)
    table = []
    for key, g in groups.items():
        if len(g) < args.min:
            continue
        n, mean, se, folds = cell(g)
        table.append((n * mean, key, n, mean, se, folds))
    for total, (street, size, strength), n, mean, se, folds in sorted(table, reverse=True)[:15]:
        print(f"| {STREETS[street]} | {size:g} | {strength} | {n} | {mean:+.1f} ± {se:.1f} | {100 * folds:.0f}% | {total:+.0f} |")

    print("\n## What our bot perceived each size as\n")
    print("| street | size | " + " | ".join(PERCEIVED.values()) + " |")
    print("|---|---|" + "---|" * len(PERCEIVED))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["street"], r["size"])].append(r.get("perceived"))
    for (street, size), g in sorted(groups.items()):
        if len(g) < args.min:
            continue
        print(f"| {STREETS[street]} | {size:g} | " + " | ".join(
            f"{100 * sum(p == code for p in g) / len(g):.0f}%" for code in PERCEIVED) + " |")


if __name__ == "__main__":
    main()
