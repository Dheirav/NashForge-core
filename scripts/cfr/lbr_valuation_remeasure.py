"""
Does the bound clear zero now that the raise valuation counts the call?

The same two strategies the action-abstraction study trained — 59,050 iterations
at raise_cap 1 and 2 — measured again with the fixed valuation. Nothing is
retrained, so any change is the exploiter, not the opponent.

Before the fix, at cap 1: -0.402 on-tree, -0.415 off-tree. Both slack.
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/dheirav/Code/PokerBot")

import numpy as np

from cfr.lbr import DEFAULT_BET_SIZES, lbr_value

HANDS = 2000
ROLLOUTS = 20
ON_TREE = (0.5, 1.0, 2.0)
BEFORE = {"cap1:on-tree": -0.402, "cap1:off-tree": -0.415,
          "cap2:on-tree": +2.783, "cap2:off-tree": +2.059}

RESULTS = os.path.join(HERE, "remeasure.json")
results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS))

print(f"{HANDS:,} hands per row, {ROLLOUTS} rollout samples\n")
print(f"{'':<8}{'exploiter':<11}{'LBR now':>20}{'was':>9}{'verdict':>22}")
print("-" * 71)

for cap in (1, 2):
    with open(os.path.join(HERE, f"solver_cap{cap}.pkl"), "rb") as handle:
        solver = pickle.load(handle)
    strategy = solver.average_strategy()

    for label, sizes in (("on-tree", ON_TREE), ("off-tree", DEFAULT_BET_SIZES)):
        key = f"cap{cap}:{label}"
        if key not in results:
            started = time.perf_counter()
            outcome = lbr_value(solver.game, strategy, hands=HANDS,
                                rng=np.random.default_rng(7000),
                                rollout_samples=ROLLOUTS, candidates=32,
                                bet_sizes=sizes)
            results[key] = {"mean": outcome.mean, "stderr": outcome.stderr,
                            "ci95": list(outcome.ci95),
                            "proves_exploitable": outcome.proves_exploitable,
                            "iterations": solver.iterations,
                            "information_sets": len(solver.nodes),
                            "seconds": time.perf_counter() - started}
            with open(RESULTS, "w") as handle:
                json.dump(results, handle, indent=2)

        row = results[key]
        verdict = ("PROVES EXPLOITABLE" if row["proves_exploitable"] else "slack")
        print(f"cap {cap}  {label:<11}{row['mean']:>+11.3f} +/-{row['stderr']:<5.3f}"
              f"{BEFORE[key]:>+9.3f}{verdict:>22}", flush=True)
    print()

cap1 = results["cap1:off-tree"]
print("The question this was built to answer: at cap 1, 59,050 iterations —")
if cap1["proves_exploitable"]:
    print(f"  the bound now CLEARS ZERO at {cap1['mean']:+.3f} +/- {cap1['stderr']:.3f}.")
    print("  Exploitability is measurable on a converged strategy, and the")
    print("  crossover experiment can be re-run against a bound that works.")
else:
    low, high = cap1["ci95"]
    print(f"  still slack: {cap1['mean']:+.3f} +/- {cap1['stderr']:.3f} "
          f"[{low:+.3f}, {high:+.3f}].")
    print("  The valuation was a real defect but not the binding one. What")
    print("  remains is the second barrel — the rollout still assumes no")
    print("  betting after the modelled action.")
