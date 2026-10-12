"""
The push-fold game at short stacks, solved exactly, and what each bot loses to it.

    venv/bin/python scripts/push_fold.py --table results/cfr/chance/nolimit_12bb_preflop_allin.npy
    venv/bin/python scripts/push_fold.py --measure <bot> <bot> --depths 6 8 10 12 15

Below about fifteen big blinds a heads-up hand is nearly always decided before
the flop: the small blind shoves or folds, and the big blind calls or folds.
That game is small enough to solve exactly rather than approximately, which is
the opposite of every other depth in this project, and its solution is a pair
of lists: which of the 169 starting hands to shove, and which to call with.
Both follow from one 169-by-169 matrix of all-in equities
(`scripts/cfr/preflop_allin.py`) by iterating best responses until they stop
moving, which they do in a handful of rounds because each side's list is
monotone in strength.

The measurement this exists for: every set we trained this week gained six to
fifteen big blinds per hundred at deep stacks and nothing below 25, because
there is little room between good and perfect once the game is fold-or-shove.
The room is in the opponent's errors, and here they are exactly priceable: for
a bot's observed shoving and calling lists, the chips it gives away per hand
against correct play, in big blinds. That number says whether a short-stack
exploiter is worth training at all.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.equity import RANKS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TABLE = os.path.join(ROOT, "results", "cfr", "chance", "nolimit_12bb_preflop_allin.npy")


def hand_labels(rung: str):
    """
    The 169 classes in the order the all-in table was written: the lossless
    preflop abstraction's own class numbers, not an invented ordering. Read
    from the rung the table was built from, since that is what indexed it.
    """
    from cfr.flat import load_strategy
    preflop = load_strategy(rung)["abstraction"]._preflop
    labels = [""] * (max(preflop.values()) + 1)
    for (high, low, suited), index in preflop.items():
        labels[index] = f"{high}{high}" if high == low else f"{high}{low}{'s' if suited else 'o'}"
    return labels


def solve(edge: np.ndarray, combos, stack_bb: float, iterations: int = 400):
    """
    The exact equilibrium of the shove-or-fold game at `stack_bb` blinds.

    Seat 0 (small blind, the button heads-up) shoves or folds; seat 1 calls or
    folds. `edge[i, j]` is P(win) - P(lose) for hand i against hand j all in.
    Chips are in big blinds, the small blind is half. Returns (shove, call) as
    boolean masks over the 169 classes, and the small blind's value per hand.
    """
    n = edge.shape[0]
    # Combinations, not classes: a pair is 6 of the 1,326 deals, a suited hand
    # 4, an offsuit one 12. Weighting by class made every pocket pair as likely
    # as every offsuit hand and put 73o in a ten-blind shoving range.
    weight = np.array(combos, dtype=float)
    weight = weight / weight.sum()
    # Fictitious play rather than alternating best responses: each side moves a
    # fraction of the way to its best response, so the two do not chase each
    # other around a cycle. Straight best-response iteration oscillated between
    # "shove everything" and "call everything" and returned whichever side the
    # loop happened to end on.
    shove_p = np.full(n, 0.5)
    call_p = np.full(n, 0.5)
    value_shove = np.zeros(n)
    value_call = np.zeros(n)
    for step in range(iterations):
        rate = 1.0 / (step + 2)
        # The big blind's value of calling a shove, against the current shoving
        # range. Folding costs the blind it posted, 1; calling risks its stack.
        w = weight * shove_p
        w = w / max(w.sum(), 1e-12)
        equity = 0.5 * (1.0 + edge @ w)
        value_call = equity * (2 * stack_bb) - stack_bb
        call_p = (1 - rate) * call_p + rate * (value_call > -1.0)
        # The small blind's value of shoving against the current calling range:
        # it takes the blinds when called off, otherwise it is all in.
        w = weight * call_p
        p_call = float(w.sum())
        w = w / max(w.sum(), 1e-12)
        equity_vs_call = 0.5 * (1.0 + edge @ w)
        value_shove = (1 - p_call) * 1.0 + p_call * (equity_vs_call * (2 * stack_bb) - stack_bb)
        shove_p = (1 - rate) * shove_p + rate * (value_shove > -0.5)
    shove = value_shove > -0.5
    call = value_call > -1.0
    value = float(np.average(np.where(shove, value_shove, -0.5), weights=weight))
    return shove, call, value, value_shove, value_call


def cost_of(observed: np.ndarray, correct_value: np.ndarray, fold_value: float) -> float:
    """
    Big blinds a hand given away by playing `observed` instead of the best
    response, where `correct_value[i]` is the value of acting and `fold_value`
    the value of folding. Errors in both directions count.
    """
    best = np.maximum(correct_value, fold_value)
    taken = np.where(observed, correct_value, fold_value)
    return float(np.mean(best - taken))


def combos_of(labels) -> list:
    """Combinations of each class: 6 for a pair, 4 suited, 12 offsuit."""
    return [6 if len(l) == 2 else (4 if l.endswith("s") else 12) for l in labels]


def describe(mask: np.ndarray, labels) -> str:
    chosen = [labels[i] for i in np.flatnonzero(mask)]
    return f"{len(chosen)} classes: " + " ".join(chosen[:20]) + (" …" if len(chosen) > 20 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=DEFAULT_TABLE, help="169x169 all-in edge matrix")
    parser.add_argument("--rung", default=os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_12bb.pkl"),
                        help="the solve whose preflop classes index the table")
    parser.add_argument("--depths", type=float, nargs="+", default=[6, 8, 10, 12, 15, 20])
    args = parser.parse_args()

    edge = np.load(args.table)
    labels = hand_labels(args.rung)
    combos = combos_of(labels)
    print(f"{args.table}: {edge.shape[0]} classes, indexed by {args.rung}\n")
    print("| stack (bb) | shove | call | SB value (bb/hand) |")
    print("|---|---|---|---|")
    for stack in args.depths:
        shove, call, value, _, _ = solve(edge, combos, stack)
        w = np.array(combos, dtype=float); w = w / w.sum()
        print(f"| {stack:g} | {100 * float(w[shove].sum()):.0f}% of hands | "
              f"{100 * float(w[call].sum()):.0f}% | {value:+.3f} |")
    print()
    for stack in (10.0,):
        shove, call, _, _, _ = solve(edge, combos, stack)
        print(f"at {stack:g} big blinds")
        print(f"  shove: {describe(shove, labels)}")
        print(f"  call:  {describe(call, labels)}")


if __name__ == "__main__":
    main()
