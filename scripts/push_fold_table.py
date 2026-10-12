"""
Precompute the short-stack solution the arena player consults in a match.

    venv/bin/python scripts/push_fold_table.py

`push_fold.py` solves the shove-or-fold game exactly at one depth in about a
second, which is a second the bot does not have when a hand is on a 30-second
clock and the ladder is already loaded. This writes the answer once, for every
half blind from 2 to 20, as the two ranges a decision needs: what the small
blind shoves first in, and what the big blind calls a shove with. The all-in
edge matrix rides along in the same file so the player needs nothing else: at
a node the solution does not cover exactly, our equity against the relevant
range is one matrix row against a weighted range, which is microseconds.

Why the player wants it at all: below fifteen blinds a miss on the betting
tree falls to `fallback_choice`, and that rule folds to any bet of the pot or
more because it has no read on the bettor. Facing an all-in after we have
already raised, a third of the stack is in and the price is far better than
the rule assumes; at 8bb over our own half-pot raise it agreed with the exact
answer on 88 of the 169 classes and folded 45% of the hands that should call,
worth 0.69 big blinds each time it happened.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.push_fold import combos_of, hand_labels, solve  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TABLE = os.path.join(ROOT, "results", "cfr", "chance", "nolimit_12bb_preflop_allin.npy")
DEFAULT_RUNG = os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_12bb.pkl")
DEFAULT_OUT = os.path.join(ROOT, "results", "cfr", "chance", "push_fold_ranges.npz")


def build(edge, labels, depths):
    combos = combos_of(labels)
    shove = np.zeros((len(depths), len(labels)), dtype=bool)
    call = np.zeros_like(shove)
    for row, depth in enumerate(depths):
        shove[row], call[row], _, _, _ = solve(edge, combos, float(depth))
    return shove, call, np.array(combos, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--rung", default=DEFAULT_RUNG, help="the solve whose preflop classes index the table")
    parser.add_argument("--min", type=float, default=2.0)
    parser.add_argument("--max", type=float, default=20.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()

    edge = np.load(args.table)
    labels = hand_labels(args.rung)
    depths = np.arange(args.min, args.max + 1e-9, args.step)
    shove, call, combos = build(edge, labels, depths)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, labels=np.array(labels), depths=depths,
                        shove=shove, call=call, combos=combos, edge=edge.astype(np.float32))
    weights = combos / combos.sum()
    print(f"wrote {args.output}: {len(depths)} depths, {len(labels)} classes, "
          f"{os.path.getsize(args.output) / 1e6:.1f} MB")
    print("| stack (bb) | shove | call |")
    print("|---|---|---|")
    for row, depth in enumerate(depths):
        if abs(depth - round(depth)) > 1e-9 or int(depth) % 4:
            continue
        print(f"| {depth:g} | {100 * float(weights[shove[row]].sum()):.0f}% | "
              f"{100 * float(weights[call[row]].sum()):.0f}% |")


if __name__ == "__main__":
    main()
