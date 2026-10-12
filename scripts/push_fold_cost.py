"""
What our short rungs' small raises cost against exact push-fold play.

    venv/bin/python scripts/push_fold_cost.py --rungs 5bb 8bb 12bb --hands 40000

Below about twelve big blinds the game is fold-or-shove, and it is small
enough to solve exactly (`push_fold.py`): at eight blinds the button shoves
62% of hands and the big blind calls 45%. Our 8bb rung instead raises to about
a third of its stack on 47% of hands and shoves on 6%. A first attempt to
price that (an earlier arena-log script) charged every non-shove as a
fold and read 0.44 big blinds a decision for every bot alike, which is the
signature of a measurement measuring itself: a half-pot raise at eight blinds
is not a fold.

This prices it properly, by playing it. Our rung sits in the small blind
against an opponent that plays the exact solution: it shoves its solved range,
and facing a raise of any size it calls with the hands whose equity beats the
price and folds the rest, which is the best response to being raised small.
Both sides see real cards, the hand is run out when the chips go in, and the
result is chips per hand from our seat. Zero would mean the small raises cost
nothing; a large negative number means the short rungs should be given
shove-or-fold and nothing else, which is the game they are in.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, legal_actions, schedule_from_args  # noqa: E402
from cfr.flat import load_strategy  # noqa: E402
from engine.cards import Card  # noqa: E402
from scripts.push_fold import combos_of, hand_labels, solve  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANKS = "23456789TJQKA"


def class_of(cards, labels_index):
    (a, b) = sorted(cards, key=lambda c: -(c % 13))
    high, low = RANKS[a % 13], RANKS[b % 13]
    suited = (a // 13) == (b // 13)
    key = f"{high}{high}" if a % 13 == b % 13 else f"{high}{low}{'s' if suited else 'o'}"
    return labels_index[key]


def showdown(mine, theirs, board):
    from engine.hand_eval_fast import score_hand_7_fast
    def score(hole):
        cards = list(hole) + list(board)
        return int(score_hand_7_fast(np.array([c % 13 for c in cards], dtype=np.int64),
                                     np.array([c // 13 for c in cards], dtype=np.int64)))
    a, b = score(mine), score(theirs)
    return 1 if a > b else (-1 if a < b else 0)


def play(rung_path, hands, seed, edge, labels, combos):
    """Our rung in the small blind against exact push-fold. Chips per hand, in big blinds."""
    saved = load_strategy(rung_path)
    strategy = saved["strategy"]
    args = saved["args"]
    cap = args["raise_cap"]
    cap = schedule_from_args(cap)
    stack, bb = int(args["stack"]), int(args["big_blind"])
    depth = stack / bb
    abstraction = saved["abstraction"]
    shove_ok, call_ok, _, value_shove, value_call = solve(edge, combos, depth)
    labels_index = {label: i for i, label in enumerate(labels)}
    actions = legal_actions(0, True, cap, street=0)       # the small blind's first decision
    rng = np.random.default_rng(seed)
    deck = np.arange(52)
    total = 0.0
    counts = {"fold": 0, "shove": 0, "small raise": 0, "call": 0}
    for _ in range(hands):
        rng.shuffle(deck)
        mine, theirs, board = deck[:2], deck[2:4], deck[4:9]
        klass = class_of([int(c) for c in mine], labels_index)
        hole = [Card(RANKS[int(c) % 13], "cdhs"[int(c) // 13]) for c in mine]
        bucket = abstraction.bucket(hole, [], np.random.default_rng(klass))
        probabilities = strategy.get(f"{bucket}|")
        if probabilities is None or len(probabilities) != len(actions):
            probabilities = np.full(len(actions), 1.0 / len(actions))
        choice = actions[int(rng.choice(len(actions), p=np.asarray(probabilities, dtype=float) /
                                        np.sum(probabilities)))]
        if choice == FOLD:
            total += -0.5
            counts["fold"] += 1
            continue
        if choice == ALL_IN:
            counts["shove"] += 1
            # The big blind calls its solved range.
            if not call_ok[class_of([int(c) for c in theirs], labels_index)]:
                total += 1.0
                continue
            total += depth * showdown(mine, theirs, board)
            continue
        if choice == CHECK_CALL:
            counts["call"] += 1
            # Limping is outside the push-fold game; the exact opponent checks
            # and the hand is decided at showdown for the blinds.
            total += 1.0 * showdown(mine, theirs, board)
            continue
        # A sized raise: the opponent re-shoves with the hands that beat the
        # price of calling our raise, and we are committed, so it is all in.
        counts["small raise"] += 1
        their_class = class_of([int(c) for c in theirs], labels_index)
        if call_ok[their_class]:
            total += depth * showdown(mine, theirs, board)
        else:
            total += 1.0
    n = max(hands, 1)
    return total / n, {k: v / n for k, v in counts.items()}, depth, shove_ok, combos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", default="results/cfr/ladder_a")
    parser.add_argument("--rungs", nargs="+", default=["5bb", "8bb", "12bb"])
    parser.add_argument("--hands", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", default=os.path.join(ROOT, "results", "cfr", "chance", "nolimit_12bb_preflop_allin.npy"))
    args = parser.parse_args()

    edge = np.load(args.table)
    labels = hand_labels(os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_12bb.pkl"))
    combos = combos_of(labels)
    weights = np.array(combos, dtype=float)
    weights = weights / weights.sum()

    print("| rung | depth | our fold | our small raise | our shove | our limp | correct shove | chips/hand (bb) |")
    print("|---|---|---|---|---|---|---|---|")
    for rung in args.rungs:
        matches = [p for p in glob.glob(os.path.join(args.ladder, f"*_{rung}.pkl")) if ".flat." not in p]
        if not matches:
            print(f"| {rung} | (not in {args.ladder}) |")
            continue
        value, counts, depth, shove_ok, _ = play(matches[0], args.hands, args.seed, edge, labels, combos)
        print(f"| {rung} | {depth:.0f}bb | {counts['fold']:.0%} | {counts['small raise']:.0%} | "
              f"{counts['shove']:.0%} | {counts['call']:.0%} | "
              f"{float(weights[shove_ok].sum()):.0%} | **{value:+.3f}** |")


if __name__ == "__main__":
    main()
