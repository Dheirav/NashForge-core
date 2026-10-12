"""
What each bot's short-stack play costs it, priced against the exact solution.

    venv/bin/python scripts/push_fold_errors.py --max-bb 10
    venv/bin/python scripts/push_fold_errors.py --max-bb 12 --out errors.md

Below about twelve big blinds a heads-up hand is fold-or-shove, and that game
is solvable exactly (`scripts/push_fold.py`): at ten blinds the button shoves
59% of hands and the big blind calls 37%. Every improvement this project made
this week was worth six to fifteen big blinds per hundred at deep stacks and
nothing at all below 25, because there is no room between good and perfect once
the game is that small. The room is in the opponent's errors, and here they can
be priced exactly rather than estimated.

The corpus is what makes this measurable: the platform's published 247,946
hands (Kaggle, 23 August 2026) carry each seat's stack and the blinds on every
hand, so a hand can be selected by its true depth, alongside both players'
cards. Our own cache of hands has the cards but not the stacks, and a first
attempt at this measurement without them counted every preflop decision as a
shoving spot and produced the same 0.4 bb/hand "cost" for all ten opponents,
which is the signature of a measurement measuring itself.

A bot's cost here is the value it gives up per short-stack decision by shoving
or calling where the solution folds, or folding where it acts. It is an upper
bound on what an exploiter could win from that bot preflop at those depths, and
a lower bound on nothing: a bot that plays the solution exactly reads zero.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.push_fold import combos_of, hand_labels, solve  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANKS = "23456789TJQKA"


def hand_class(cards: str) -> str:
    """"9s Ts" -> "T9s"; the corpus writes hole cards space-separated."""
    a, b = cards.split()
    (r1, s1), (r2, s2) = (a[0], a[1]), (b[0], b[1])
    high, low = (r1, r2) if RANKS.index(r1) >= RANKS.index(r2) else (r2, r1)
    if r1 == r2:
        return f"{high}{high}"
    return f"{high}{low}{'s' if s1 == s2 else 'o'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=os.path.expanduser("~/pokerbot-scratch/kaggle"))
    parser.add_argument("--table", default=os.path.join(ROOT, "results", "cfr", "chance", "nolimit_12bb_preflop_allin.npy"))
    parser.add_argument("--rung", default=os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_12bb.pkl"))
    parser.add_argument("--max-bb", type=float, default=10.0, help="only hands where the effective stack is at or under this")
    parser.add_argument("--shove-fraction", type=float, default=0.8,
                        help="a raise of at least this much of the raiser's stack counts as a shove")
    parser.add_argument("--out")
    args = parser.parse_args()

    edge = np.load(args.table)
    labels = hand_labels(args.rung)
    combos = combos_of(labels)
    index = {label: i for i, label in enumerate(labels)}

    hands = pd.read_parquet(os.path.join(args.dir, "hands.parquet"))
    hands["effective_bb"] = hands[["stack_seat0", "stack_seat1"]].min(axis=1) / hands["big_blind"]
    short = hands[hands["effective_bb"] <= args.max_bb]
    actions = pd.read_parquet(os.path.join(args.dir, "actions.parquet"))
    actions = actions.merge(short[["match_id", "hand_number", "sb_seat", "big_blind",
                                   "stack_seat0", "stack_seat1", "hole_seat0", "hole_seat1", "effective_bb"]],
                            on=["match_id", "hand_number"], how="inner")
    print(f"{len(short):,} hands at or under {args.max_bb:g} big blinds "
          f"({100 * len(short) / len(hands):.1f}% of {len(hands):,}), {len(actions):,} actions", flush=True)

    # The solution at the median depth of the selected hands.
    depth = float(short["effective_bb"].median())
    shove_ok, call_ok, _, value_shove, value_call = solve(edge, combos, depth)
    weights = np.array(combos, dtype=float)
    print(f"solved at the median depth, {depth:.1f}bb: shove {100 * np.average(shove_ok, weights=weights):.0f}% "
          f"of hands, call {100 * np.average(call_ok, weights=weights):.0f}%\n", flush=True)

    rows = defaultdict(lambda: {"shove_spots": 0, "shoved": 0, "shove_cost": 0.0,
                                "call_spots": 0, "called": 0, "call_cost": 0.0})
    for (match_id, hand_number), group in actions.groupby(["match_id", "hand_number"], sort=False):
        group = group.sort_values("action_idx")
        first = group.iloc[0]
        sb_seat = int(first["sb_seat"]) if pd.notna(first["sb_seat"]) else 0
        stacks = {0: float(first["stack_seat0"]), 1: float(first["stack_seat1"])}
        holes = {0: first["hole_seat0"], 1: first["hole_seat1"]}
        shoved_by = None
        for a in group.itertuples():
            if str(a.action_type).startswith("post"):
                continue
            seat = int(a.seat)
            klass = index.get(hand_class(holes[seat]) if holes[seat] else "")
            if klass is None:
                continue
            row = rows[a.agent]
            if shoved_by is None:
                # This seat is first to act with no shove in front: shove or fold.
                if seat != sb_seat and str(a.action_type) == "check":
                    break                                   # the big blind's option after a limp: not this game
                row["shove_spots"] += 1
                is_shove = str(a.action_type) == "raise" and float(a.amount) >= args.shove_fraction * stacks[seat]
                row["shoved"] += int(is_shove)
                best = max(value_shove[klass], -0.5)
                row["shove_cost"] += best - (value_shove[klass] if is_shove else -0.5)
                if is_shove:
                    shoved_by = seat
                elif str(a.action_type) == "fold":
                    break
                else:
                    break                                   # limped or raised small: outside the push-fold game
            else:
                if seat == shoved_by:
                    break
                row["call_spots"] += 1
                called = str(a.action_type) == "call"
                row["called"] += int(called)
                best = max(value_call[klass], -1.0)
                row["call_cost"] += best - (value_call[klass] if called else -1.0)
                break

    lines = ["| agent | shove spots | shoves | correct | cost (bb) | call spots | calls | correct | cost (bb) | total bb/decision |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    correct_shove = 100 * np.average(shove_ok, weights=weights)
    correct_call = 100 * np.average(call_ok, weights=weights)
    for agent in sorted(rows):
        r = rows[agent]
        spots = r["shove_spots"] + r["call_spots"]
        if spots < 200:
            continue
        lines.append(
            f"| {agent} | {r['shove_spots']:,} | {100 * r['shoved'] / max(r['shove_spots'], 1):.0f}% | "
            f"{correct_shove:.0f}% | {r['shove_cost'] / max(r['shove_spots'], 1):.3f} | "
            f"{r['call_spots']:,} | {100 * r['called'] / max(r['call_spots'], 1):.0f}% | {correct_call:.0f}% | "
            f"{r['call_cost'] / max(r['call_spots'], 1):.3f} | "
            f"**{(r['shove_cost'] + r['call_cost']) / max(spots, 1):.3f}** |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(f"# Short-stack errors at or under {args.max_bb:g} big blinds\n\n"
                         f"{len(short):,} hands from the published corpus, solved at {depth:.1f}bb.\n\n"
                         + table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
