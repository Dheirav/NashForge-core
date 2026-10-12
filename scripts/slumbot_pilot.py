"""
A few hundred hands against Slumbot, to shake out protocol bugs.

Step 4 of `docs/EXTERNAL_BENCHMARK.md`, and its instruction is explicit: **do not
read the result**. A win rate over a few hundred hands has an interval wider than
anything it could tell us, and reading one here is how a pilot becomes a finding
that has to be withdrawn later. This prints protocol health and writes the hands
to disk; it does not print a win rate, and the file is there for M1 to be
computed from deliberately rather than glanced at now.

What counts as healthy
----------------------
**Every hand completes.** A protocol error loses a hand to the bridge rather than
to poker, and at any rate makes the hand count a lie.

**The miss rate is low.** It is the fraction of decisions where the strategy had
no entry for the node and the agent chose among legal actions at random. This
project published a result once at a 74.3% miss rate, where the benchmark had
quietly become a second random opponent; the counter exists so that shows up as a
number. Some misses are expected here and are not a bug: Slumbot re-raises and the
solver knows one raise per street, so those nodes are genuinely off-tree.

**Position alternates.** If `client_pos` never changes, every hand is played from
one seat, and the result carries a positional bias that no number of hands will
average away.

Usage
-----
    venv/bin/python scripts/slumbot_pilot.py --hands 300
"""
import argparse
from types import SimpleNamespace
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import numpy as np

from slumbot import SlumbotError, play_hand
from slumbot.api import always_fold, call_station
from slumbot.player import SolverPlayer

STRATEGY = os.path.join(ROOT, "results", "cfr", "nolimit_strategy.pkl")
OUT = os.path.join(ROOT, "results", "slumbot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(OUT, "pilot.json"))
    parser.add_argument("--strategy", default=STRATEGY,
                        help="the pickle to play; its raise schedule is read from the file")
    parser.add_argument("--purify", default="none", choices=["none", "postflop", "all"],
                        help="play the most probable action instead of sampling; off by default")
    parser.add_argument("--policy", default="solver", choices=["solver", "fold", "call"],
                        help="fold or call plays a fixed policy instead of the solver, for calibration")
    parser.add_argument("--no-fallback", action="store_true",
                        help="guess uniformly at random on a lookup miss, as before 15 September")
    args = parser.parse_args()

    if args.policy == "solver":
        player = SolverPlayer(args.strategy, np.random.default_rng(args.seed), purify=args.purify,
                              fallback=not args.no_fallback)
    else:
        # A fixed policy needs no strategy, and the contender's pickle is 3 GB
        # in a process: loading it for a calibration run is what would keep the
        # calibration from running beside anything else.
        from slumbot.player import SessionStats
        player = SimpleNamespace(stats=SessionStats(), raise_cap=None, purify="none", fallback=False,
                                 begin_hand=lambda: None,
                                 hand_record=lambda state: {"w": state.winnings, "pos": state.client_pos,
                                                            "street": state.action.count("/"),
                                                            "showdown": not state.action.rstrip("/").endswith("f")})
    policy = {"solver": player, "fold": always_fold, "call": call_station}[args.policy]
    print(f"strategy {os.path.basename(args.strategy)}, schedule {player.raise_cap}")
    hands, errors, positions = [], [], Counter()
    token = None

    print(f"{args.hands} hands, seed {args.seed} — protocol health only\n")
    for index in range(args.hands):
        try:
            player.begin_hand()
            state = play_hand(policy, token)
            token = state.token
            positions[state.client_pos] += 1
            hands.append({
                **player.hand_record(state),
                "action": state.action,
                "client_pos": state.client_pos,
                "hole_cards": state.hole_cards,
                "board": state.board,
                "winnings": state.winnings,
                "baseline_winnings": state.baseline_winnings,
            })
        except SlumbotError as error:
            errors.append(f"hand {index + 1}: {error}")
        if (index + 1) % 50 == 0:
            print(f"  {index + 1:>4} hands, {len(errors)} errors, "
                  f"miss rate {player.stats.miss_rate:.1%}", flush=True)

    report(player, hands, errors, positions, args)
    if args.policy != "solver" and hands:
        w = np.array([h["w"] for h in hands if h.get("w") is not None], dtype=float)
        b = np.array([h["w"] - h["baseline_winnings"] for h in hands if h.get("baseline_winnings") is not None], dtype=float)
        print(f"\ncalibration ({args.policy}): raw {w.mean() / 100 * 1000:+.0f} ± {1.96 * w.std(ddof=1) / np.sqrt(len(w)) / 100 * 1000:.0f} mbb/hand"
              + (f"; baseline-differenced {b.mean() / 100 * 1000:+.0f} ± {1.96 * b.std(ddof=1) / np.sqrt(len(b)) / 100 * 1000:.0f}" if len(b) > 1 else ""))


def report(player, hands, errors, positions, args):
    stats = player.stats
    print(f"\n{'-' * 58}")
    print(f"completed          {len(hands)}/{args.hands}")
    print(f"protocol errors    {len(errors)}")
    for line in errors[:5]:
        print(f"    {line}")
    print(f"decisions          {stats.decisions}")
    print(f"actions sent       {dict(stats.actions_sent)}")
    print(f"lookup miss rate   {stats.miss_rate:.1%} "
          f"({stats.misses}/{stats.consulted})")
    print(f"misses by raises already on the street  {dict(sorted(stats.miss_depths.items()))}")
    print(f"re-raises the schedule had no size for  {dict(sorted(stats.schedule_misses.items()))}")
    print(f"off-abstraction    {stats.off_abstraction} bets too large to describe")
    print(f"seat distribution  {dict(positions)}")

    if len(positions) < 2:
        print("\n  Position never alternated. Every hand was played from one")
        print("  seat, so any result carries a positional bias that more hands")
        print("  will not average away. This has to be handled before M1.")

    if stats.miss_rate > 0.25:
        print(f"\n  A {stats.miss_rate:.0%} miss rate means the agent chose at")
        print("  random that often. A result measured here is substantially a")
        print("  measurement of a random agent wearing the solver's name.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"hands": hands, "errors": errors,
                   "decisions": stats.decisions,
                   "miss_rate": stats.miss_rate,
                   "strategy": os.path.basename(args.strategy),
                   "schedule": str(player.raise_cap),
                   "purify": player.purify, "policy": args.policy,
                   "fallback": player.fallback, "fallbacks": player.stats.fallbacks,
                   "misses_by_raise_depth": {str(k): v for k, v in sorted(stats.miss_depths.items())},
                   "schedule_misses_by_raise_depth": {str(k): v for k, v in sorted(stats.schedule_misses.items())},
                   "off_abstraction": stats.off_abstraction,
                   "note": "PILOT — protocol shakedown. Not a result. "
                           "The win rate over these hands is not to be quoted."},
                  handle, indent=1)
    print(f"\nwrote {args.out}")
    print("The win rate is deliberately not printed: this is a pilot, and the")
    print("interval over a few hundred hands is wider than anything it says.")


if __name__ == "__main__":
    main()
