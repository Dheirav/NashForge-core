"""
Is the cheap postflop strength signal worth using?

Bucketing dominates no-limit training — 86% of a profiled run — and almost all
of that is Monte Carlo equity. A made-hand score costs about one ninetieth as
much, but it cannot see a draw: a flush draw scores as whatever it has already
made. So this is a trade, not an optimisation, and the trade has to be measured.

**The comparison is at matched wall-clock, not matched iterations.** A cheaper
signal buys more iterations in the same time, and iterations are what reduce
regret. Comparing per-iteration would flatter the expensive signal by ignoring
the only advantage the cheap one has.

Usage
-----
    python scripts/cfr/compare_strength_signals.py
    python scripts/cfr/compare_strength_signals.py --seconds 120 --seeds 3
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.buckets import STRENGTH_SIGNALS, CardAbstraction  # noqa: E402
from cfr import MCCFRSolver, VANILLA  # noqa: E402
from cfr.play import always_call_policy, play_hands, strategy_policy, uniform_policy  # noqa: E402
from games.nolimit import NoLimitHoldem  # noqa: E402


def train_for(seconds: float, signal: str, buckets: int, seed: int,
              equity_samples: int, raise_cap: int):
    """Train until the budget is spent; returns the solver and iteration count."""
    rng = np.random.default_rng(seed)
    abstraction = CardAbstraction(preflop_buckets=buckets, postflop_buckets=buckets,
                                  samples=800, equity_samples=equity_samples,
                                  strength=signal).fit(rng)
    game = NoLimitHoldem(abstraction, raise_cap=raise_cap,
                         equity_samples=equity_samples)
    solver = MCCFRSolver(game, rule=VANILLA, seed=seed)

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        solver.train(25)
    return game, solver


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="training wall-clock budget per signal per seed")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=6)
    parser.add_argument("--raise-cap", type=int, default=1)
    parser.add_argument("--equity-samples", type=int, default=40)
    parser.add_argument("--eval-hands", type=int, default=3000)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Equal wall-clock budget: {args.seconds:.0f}s of training per signal, "
          f"{args.seeds} seeds\n")

    header = (f"{'signal':<11}{'iters':>9}{'ms/iter':>9}{'infosets':>10}"
              f"{'vs random':>22}{'vs always-call':>22}")
    print(header)
    print("-" * len(header))

    results = {}
    for signal in STRENGTH_SIGNALS:
        iterations, rates, infosets = [], [], []
        against_random, against_call = [], []

        for seed in range(args.seeds):
            start = time.perf_counter()
            game, solver = train_for(args.seconds, signal, args.buckets, seed,
                                     args.equity_samples, args.raise_cap)
            elapsed = time.perf_counter() - start

            iterations.append(solver.iterations)
            rates.append(elapsed / solver.iterations * 1000)
            infosets.append(len(solver.nodes))

            trained = strategy_policy(solver.average_strategy())
            for policy, bucket in ((uniform_policy(), against_random),
                                   (always_call_policy(), against_call)):
                outcome = play_hands(game, [trained, policy], args.eval_hands,
                                     np.random.default_rng(1000 + seed))
                bucket.append(outcome)

        def summarise(outcomes):
            means = [o.mean for o in outcomes]
            stderr = statistics.fmean(o.stderr for o in outcomes)
            return statistics.fmean(means), stderr

        random_mean, random_err = summarise(against_random)
        call_mean, call_err = summarise(against_call)

        print(f"{signal:<11}{statistics.fmean(iterations):>9,.0f}"
              f"{statistics.fmean(rates):>9.1f}{statistics.fmean(infosets):>10,.0f}"
              f"{random_mean:>14.2f} +/-{random_err:<6.2f}"
              f"{call_mean:>14.2f} +/-{call_err:<6.2f}")

        results[signal] = {
            "iterations": statistics.fmean(iterations),
            "ms_per_iteration": statistics.fmean(rates),
            "infosets": statistics.fmean(infosets),
            "vs_random": {"mean": random_mean, "stderr": random_err},
            "vs_always_call": {"mean": call_mean, "stderr": call_err},
        }

    fast, slow = results["made_hand"], results["equity"]
    print(f"\nmade_hand runs {slow['ms_per_iteration'] / fast['ms_per_iteration']:.1f}x "
          f"faster per iteration, so it completes "
          f"{fast['iterations'] / slow['iterations']:.1f}x more of them in the same time.")

    for label, key in (("vs random", "vs_random"), ("vs always-call", "vs_always_call")):
        gap = fast[key]["mean"] - slow[key]["mean"]
        noise = fast[key]["stderr"] + slow[key]["stderr"]
        verdict = ("indistinguishable at this sample size" if abs(gap) < noise
                   else ("made_hand ahead" if gap > 0 else "equity ahead"))
        print(f"  {label:<16} difference {gap:+.2f} chips/hand — {verdict}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
