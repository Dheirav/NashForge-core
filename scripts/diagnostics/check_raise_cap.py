"""
Does the raise cap explain the non-transitivity?

Phase 4 left a table that does not rank. PPO at 2M hands draws level with the
solver head to head, yet the solver takes nearly twice as much off both
baselines — +722.9 against always-call to PPO's +297.9. The instrument was
cleared first (`check_instrument.py`: the two paths agree bit-for-bit), so the
next candidate is the rule the panel plays under.

The hypothesis
--------------
Every matchup is constrained to the solver's tree: **one raise per street**.
Against a station that never folds, the natural way to take its stack is to
raise again and again, and nobody is allowed to. If that cap binds harder on
PPO than on the solver, it would produce exactly the observed shape, and it
would be a fact about the measurement rather than about the agents.

What lifting the cap costs
--------------------------
It takes the solver **off-tree**. Its strategy was fitted for one raise per
street, so at a higher cap it is being asked about nodes it was never solved
for, and `cfr_agent` falls back to choosing among legal actions at random when a
lookup misses. The miss rate is printed for exactly this reason: at cap 1 it is
near zero, and if it climbs at cap 2 then the solver's uncapped column is not a
measurement of the solver so much as of a partly-random agent wearing its name.

So this diagnoses the cap. It does not produce a corrected Phase 4 row, and the
uncapped numbers should not be quoted as either family's strength.

Usage
-----
    venv/bin/python scripts/diagnostics/check_raise_cap.py
    venv/bin/python scripts/diagnostics/check_raise_cap.py --hands 4000 --caps 1 2
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
import torch

torch.set_num_threads(1)

from evaluation import always_call_agent, benchmark, cfr_agent, random_agent
from rl.ppo.agent import PPOAgent
from train_ppo import ppo_as_benchmark_agent

EVAL_SEED = 20260820
HANDS = 40_000
RUNG = 2_000_000                     # the rung that drew level with the solver
STRATEGY = os.path.join(ROOT, "results", "cfr", "nolimit_strategy.pkl")
PPO_SCRATCH = os.path.expanduser("~/pokerbot-scratch/phase3")


def solver(raise_cap, misses):
    with open(STRATEGY, "rb") as handle:
        saved = pickle.load(handle)
    return cfr_agent(saved["strategy"], saved["abstraction"],
                     np.random.default_rng(4), misses=misses,
                     raise_cap=raise_cap)


def ppo(seed):
    path = os.path.join(PPO_SCRATCH, f"seed{seed}", f"ppo_rung{RUNG}.pt")
    return ppo_as_benchmark_agent(PPOAgent.from_checkpoint(path, device="cpu"))


def opponents():
    # A fresh generator per matchup, so a result does not depend on what ran
    # before it. See check_instrument.py.
    return (("random", lambda: random_agent(np.random.default_rng(4))),
            ("always-call", lambda: always_call_agent()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=HANDS)
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    print(f"{args.hands:,} hands, seed {EVAL_SEED}, PPO rung {RUNG:,}")
    print("the solver goes off-tree above cap 1; watch its miss rate\n")

    print(f"{'cap':<5}{'agent':<14}{'vs random':>14}{'vs always-call':>17}"
          f"{'solver miss':>13}")
    print("-" * 63)

    for cap in args.caps:
        row = {}
        misses = [0, 0]
        for name, make in opponents():
            result = benchmark(solver(cap, misses), make(), name,
                               hands=args.hands, seed=EVAL_SEED, raise_cap=cap)
            row[name] = result.bb_per_100
        rate = misses[0] / misses[1] if misses[1] else 0.0
        print(f"{cap:<5}{'CFR solver':<14}{row['random']:>+14.1f}"
              f"{row['always-call']:>+17.1f}{rate:>12.1%}")

        # PPO is not fitted to any tree, so a higher cap only widens what it may
        # legally do -- there is no off-tree penalty to report for it.
        scores = {name: [] for name, _ in opponents()}
        for seed in args.seeds:
            agent = ppo(seed)
            for name, make in opponents():
                result = benchmark(agent, make(), name, hands=args.hands,
                                   seed=EVAL_SEED, raise_cap=cap)
                scores[name].append(result.bb_per_100)
        label = f"PPO ({len(args.seeds)} seeds)"
        print(f"{'':<5}{label:<14}{np.mean(scores['random']):>+14.1f}"
              f"{np.mean(scores['always-call']):>+17.1f}{'--':>13}")

        gap = np.mean(scores['always-call']) - row['always-call']
        print(f"{'':<5}{'gap':<14}{'':<14}{gap:>+17.1f}"
              f"   (PPO minus solver)\n")

    print("-" * 63)
    print("If the gap against always-call closes as the cap rises, the cap was")
    print("suppressing PPO and the non-transitivity is partly an artefact of the")
    print("tree. If it holds, the cap is not the explanation.")


if __name__ == "__main__":
    main()
