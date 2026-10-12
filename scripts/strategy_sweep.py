"""
Read a solve's strategy for implausible actions, before it plays.

    venv/bin/python scripts/strategy_sweep.py results/cfr/ladder_a
    venv/bin/python scripts/strategy_sweep.py results/cfr/experiments/cap2_70bb_t421_20m_warm.pkl --threshold 0.2

The cross-tree gate and the replay judge a set by what it scores over tens of
thousands of hands. Neither can see one node that is wrong a quarter of the
time in a spot that arises once a match. Such a node
once put a quarter of a weak suited hand on an all-in facing a small open, in
one rung of one solve. Other solves on the same tree held almost none there, so
it was that solve's convergence, not the design, and a minute of reading the
table would have shown it.

Two checks, both cheap and both over the whole table rather than a sample:

* **Heavy shoves.** Nodes where a sized raise is legal and the all-in still
  carries at least `--threshold` of the weighted mass. Preflop nodes are
  listed separately because that is where a shove is most often wrong and
  most expensive.
* **Uniform entries.** Nodes whose strategy is within a hair of uniform, which
  means the solve never really visited them; the fallback would have been as
  good, and a lot of them at shallow histories says the rung is undertrained.

Nothing here is automatically a defect. Shoving the river after three bets is
ordinary. The list is what to read before a set plays.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.betting import ALL_IN, legal_actions, schedule_from_args  # noqa: E402
from cfr.flat import load_strategy  # noqa: E402
from scripts.push_fold import combos_of, hand_labels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIZE_NAMES = {0: "fold", 1: "call", 2: "raise ½pot", 3: "raise pot", 4: "raise 2×pot", 5: "all-in"}


def node_shape(history: str, cap):
    """The legal actions at a history, as the trainer defined them."""
    street = history.split("/")[-1]
    raises = sum(1 for ch in street if ch >= "2")
    facing = bool(street) and street[-1] >= "2"
    if not street and history.count("/") == 0:
        facing = True                      # the small blind opens against the big blind
    return legal_actions(raises, facing, cap, street=history.count("/")), raises, history.count("/")


def sweep(path: str, threshold: float, labels, weights):
    saved = load_strategy(path)
    strategy = saved["strategy"]
    args = saved["args"]
    cap = args["raise_cap"]
    cap = schedule_from_args(cap)
    depth = args["stack"] / args["big_blind"]

    by_history = {}
    for key, probabilities in strategy.items():
        bucket, _, history = key.partition("|")
        try:
            index = int(bucket)
        except ValueError:
            continue
        by_history.setdefault(history, {})[index] = np.asarray(probabilities, dtype=float)

    heavy, uniform_nodes, preflop = [], 0, []
    for history, entries in by_history.items():
        actions, raises, street = node_shape(history, cap)
        width = len(actions)
        sized = [a for a in actions if 2 <= a < ALL_IN]
        for index, probabilities in entries.items():
            if len(probabilities) != width or index >= len(weights):
                continue
            if abs(probabilities.max() - 1.0 / width) < 0.01:
                uniform_nodes += 1
        if ALL_IN not in actions or not sized:
            continue
        slot = actions.index(ALL_IN)
        mass = total = 0.0
        worst = []
        for index, probabilities in entries.items():
            if len(probabilities) != width or index >= len(weights):
                continue
            share = float(probabilities[slot])
            mass += weights[index] * share
            total += weights[index]
            if share >= threshold:
                worst.append((share, labels[index]))
        if total <= 0:
            continue
        worst.sort(reverse=True)
        if worst:
            row = (mass / total, history, street, raises, len(worst), worst[:4])
            heavy.append(row)
            if street == 0:
                preflop.append(row)

    heavy.sort(key=lambda r: -r[4])
    preflop.sort(key=lambda r: -r[4])
    return {"depth": depth, "cap": cap, "nodes": len(by_history), "entries": len(strategy),
            "heavy": heavy, "preflop": preflop, "uniform": uniform_nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="a ladder directory or a single pickle")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="a hand class counts as a heavy shove at this share or above")
    parser.add_argument("--top", type=int, default=5, help="nodes to print per rung")
    args = parser.parse_args()

    labels = hand_labels(os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_12bb.pkl"))
    weights = np.array(combos_of(labels), dtype=float)
    weights = weights / weights.sum()

    if os.path.isdir(args.target):
        paths = sorted(p for p in glob.glob(os.path.join(args.target, "*.pkl")) if ".flat." not in p)
    else:
        paths = [args.target]

    flagged = 0
    for path in paths:
        try:
            report = sweep(path, args.threshold, labels, weights)
        except Exception as error:                      # a rung with no args, say
            print(f"{os.path.basename(path)}: skipped ({error})")
            continue
        name = os.path.basename(path)
        print(f"\n=== {name}  ({report['depth']:.0f}bb, cap {report['cap']}, "
              f"{report['entries']:,} entries, {report['nodes']:,} histories)")
        print(f"  nodes with a hand class shoving ≥{100 * args.threshold:.0f}% where a sized raise exists: "
              f"{len(report['heavy']):,}  (preflop: {len(report['preflop']):,})")
        print(f"  entries still uniform (never really visited): {report['uniform']:,}")
        for title, rows in (("preflop", report["preflop"]), ("all streets", report["heavy"])):
            if not rows:
                continue
            print(f"  worst {title}:")
            for share, history, street, raises, count, worst in rows[: args.top]:
                hands = " ".join(f"{label} {100 * value:.0f}%" for value, label in worst)
                print(f"    '{history or '(root)'}'  street {street}, {count} hand classes over the line; {hands}")
        flagged += len(report["preflop"])
    if flagged:
        print(f"\n{flagged} preflop nodes flagged across {len(paths)} rung(s). Read them before the set plays; "
              f"a shove is not wrong by itself, an unexplainable one is.")


if __name__ == "__main__":
    main()
