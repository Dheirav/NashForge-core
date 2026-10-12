"""
Milestone 3 — which regret update rule to use under external sampling.

CFR+ converges dramatically faster than vanilla CFR when the tree is traversed
exhaustively, and its guarantees are proven in that setting. Under Monte Carlo
sampling the picture is different: flooring cumulative regret at zero discards
sampling noise asymmetrically rather than averaging it away, and Discounted CFR
was introduced partly in response. Which rule actually wins is empirical.

Leduc is the right place to settle it because exploitability is computable
*exactly* there, so the comparison is decisive rather than a matter of folklore.

Every configuration is run across several seeds and reported with its spread.
A single-seed comparison of stochastic solvers is not a measurement — that
lesson is the whole subject of CODEBASE_AUDIT.md.

Usage
-----
    python scripts/cfr/compare_update_rules.py
    python scripts/cfr/compare_update_rules.py --iterations 100000 --seeds 5
    python scripts/cfr/compare_update_rules.py --game kuhn --output results.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cfr import ALL_RULES, MCCFRSolver, exploitability  # noqa: E402
from games import KuhnPoker, LeducHoldem  # noqa: E402

GAMES = {"kuhn": KuhnPoker, "leduc": LeducHoldem}


def checkpoints(total: int) -> list:
    """Roughly logarithmic checkpoints up to ``total``."""
    marks, mark = [], 1000
    while mark < total:
        marks.append(mark)
        mark *= 4
    marks.append(total)
    return marks


def run(game, rule, seed, marks):
    """Exploitability at each checkpoint for one rule and seed."""
    solver = MCCFRSolver(game, rule=rule, seed=seed)
    measured, elapsed = {}, 0.0
    for mark in marks:
        start = time.perf_counter()
        solver.train(mark - solver.iterations)
        elapsed += time.perf_counter() - start
        measured[mark] = exploitability(game, solver.average_strategy())
    return measured, elapsed, len(solver.nodes)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=sorted(GAMES), default="leduc")
    parser.add_argument("--iterations", type=int, default=64000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", help="write results as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    game = GAMES[args.game]()
    marks = checkpoints(args.iterations)

    print(f"External-sampling MCCFR on {args.game}, {args.seeds} seeds")
    print("Exploitability (lower is better); +/- is the spread across seeds.\n")

    header = f"{'rule':<9}" + "".join(f"{m:>16,}" for m in marks) + f"{'sec':>9}"
    print(header)
    print("-" * len(header))

    results = {}
    for rule in ALL_RULES:
        per_seed, seconds, reached = [], [], []
        for seed in range(args.seeds):
            measured, elapsed, nodes = run(game, rule, seed, marks)
            per_seed.append(measured)
            seconds.append(elapsed)
            reached.append(nodes)

        row = f"{rule.name:<9}"
        summary = {}
        for mark in marks:
            values = [m[mark] for m in per_seed]
            mean = statistics.fmean(values)
            spread = (max(values) - min(values)) / 2
            summary[mark] = {"mean": mean, "half_range": spread,
                             "values": values}
            row += f"{mean:>9.4f}+/-{spread:<5.4f}"
        row += f"{statistics.fmean(seconds):>9.1f}"
        print(row)

        results[rule.name] = {
            "exploitability": summary,
            "seconds_mean": statistics.fmean(seconds),
            "infosets_reached": reached,
        }

    best = min(results, key=lambda r: results[r]["exploitability"][marks[-1]]["mean"])
    print(f"\nLowest exploitability at {marks[-1]:,} iterations: {best}")

    spread = results[best]["exploitability"][marks[-1]]["half_range"]
    runners = [r for r in results if r != best and
               results[r]["exploitability"][marks[-1]]["mean"]
               - results[best]["exploitability"][marks[-1]]["mean"] < spread]
    if runners:
        print(f"Within seed spread of {best}: {', '.join(runners)} "
              f"— not separated by this many seeds.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"game": args.game, "seeds": args.seeds,
                       "iterations": args.iterations, "results": results},
                      handle, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
