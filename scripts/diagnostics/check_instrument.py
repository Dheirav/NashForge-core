"""
Are the two families' Phase 4 rows on the same instrument?

Phase 4 put PPO and the CFR agent in one table, but their numbers were produced
by different scripts. That is the first thing to rule out before treating the
non-transitivity in that table as a fact about poker: PPO draws level with the
solver head to head, yet the solver takes nearly twice as much off both
baselines, and a measurement difference would produce exactly that shape.

The difference found by reading the two paths — since fixed
-----------------------------------------------------------
`build_panel()` used to create **one** `default_rng(4)` and hand it to the random
opponent *and* the CFR panel member, while `phase4_comparison.measure_cfr()` made
a fresh one per matchup. The two paths agreed anyway, and only because "random"
happened to be first in the panel, so nothing had drawn from the shared generator
yet. Advancing it by a single draw moved a 2,000-hand score by 24 BB/100.

That, and the larger version of it found next to it — the policy sampling from
torch's global generator, which nothing reseeded between matchups — are fixed:
the panel hands out factories now, and `panel_scores` builds a fresh opponent and
reseeds torch per matchup. This check is kept because it is what would catch the
same class of thing returning.

The test
--------
One fixed agent — the solver, which is deterministic given its seed — measured
against the same two baselines through both paths at the same width. If the
paths are equivalent the two columns agree within the interval. If they do not,
the Phase 4 table is comparing across instruments and the non-transitivity in it
is not yet evidence of anything.

Usage
-----
    venv/bin/python scripts/diagnostics/check_instrument.py
    venv/bin/python scripts/diagnostics/check_instrument.py --hands 4000
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

from evaluation import always_call_agent, benchmark, cfr_agent, random_agent
from train_ppo import build_panel

EVAL_SEED = 20260820
HANDS = 40_000
STRATEGY = os.path.join(ROOT, "results", "cfr", "nolimit_strategy.pkl")


def solver(rng_seed=4):
    with open(STRATEGY, "rb") as handle:
        saved = pickle.load(handle)
    return cfr_agent(saved["strategy"], saved["abstraction"],
                     np.random.default_rng(rng_seed), raise_cap=1)


def via_phase4(hands):
    """A fresh generator per matchup, as `measure_cfr` does."""
    out = {}
    for name, opponent in (("random", random_agent(np.random.default_rng(4))),
                           ("always-call", always_call_agent())):
        result = benchmark(solver(), opponent, name, hands=hands, seed=EVAL_SEED)
        out[name] = (result.bb_per_100, 1.96 * result.stderr / 2 * 100)
    return out


def via_panel(hands):
    """The panel's own opponents, iterated as `panel_scores` iterates them."""
    panel = build_panel()
    out = {}
    for name, make_opponent in panel:
        if name == "cfr":
            continue                    # the solver against itself is not a row
        result = benchmark(solver(), make_opponent(), name, hands=hands,
                           seed=EVAL_SEED)
        out[name] = (result.bb_per_100, 1.96 * result.stderr / 2 * 100)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=HANDS)
    args = parser.parse_args()

    print(f"the solver against both baselines, {args.hands:,} hands, "
          f"seed {EVAL_SEED}\n")
    print("  measuring through the Phase 4 path (fresh generator)...", flush=True)
    phase4 = via_phase4(args.hands)
    print("  measuring through the panel path (shared generator)...", flush=True)
    panel = via_panel(args.hands)

    print(f"\n{'opponent':<14}{'Phase 4 path':>18}{'panel path':>18}"
          f"{'difference':>16}")
    print("-" * 66)
    verdict = "the two paths agree"
    for name in phase4:
        a, a_ci = phase4[name]
        b, b_ci = panel[name]
        gap = a - b
        interval = (a_ci ** 2 + b_ci ** 2) ** 0.5
        flag = "" if abs(gap) <= interval else "   <-- SEPARATED"
        if flag:
            verdict = "the paths disagree; the Phase 4 table spans two instruments"
        print(f"{name:<14}{a:>+13.1f} ±{a_ci:<4.0f}{b:>+13.1f} ±{b_ci:<4.0f}"
              f"{gap:>+11.1f} ±{interval:<4.0f}{flag}")
    print("-" * 66)
    print(verdict)
    print("\nFor reference, Phase 4 published +377.2 vs random and +722.9 vs")
    print("always-call for this agent, at 40,000 hands.")


if __name__ == "__main__":
    main()
