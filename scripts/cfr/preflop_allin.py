"""
Preflop all-in equity between bucket classes, for the full-width solver.

    venv/bin/python scripts/cfr/preflop_allin.py results/cfr/ladder169l/nolimit_8bb.pkl

The full-width solver factorises chance (each player's strength class given
their own bucket and the board texture, independently), which misvalues a
preflop all-in: AA against 72 is not "a class-5 hand against a class-1 hand on
an average board", it is a specific 82/18. At short stacks the preflop all-in
is most of the game, so its value is taken from this table instead: for every
pair of preflop classes, P(win) - P(lose) over sampled runouts, with the two
representative hands dealt from disjoint cards. Written beside the chance
tables as `<rung>_preflop_allin.npy`, a [169, 169] matrix from seat 0's view.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from abstraction.equity import FULL_DECK, RANKS, SUITS  # noqa: E402
from cfr.flat import load_strategy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def representatives(key):
    """Concrete hands for a class, over suit choices, as card indices."""
    high, low, suited = key
    out = []
    if high == low:
        for i in range(4):
            for j in range(i + 1, 4):
                out.append((FULL_DECK.index(type(FULL_DECK[0])(high, SUITS[i])),
                            FULL_DECK.index(type(FULL_DECK[0])(low, SUITS[j]))))
    elif suited:
        for s in SUITS:
            out.append((FULL_DECK.index(type(FULL_DECK[0])(high, s)), FULL_DECK.index(type(FULL_DECK[0])(low, s))))
    else:
        for s in SUITS:
            for u in SUITS:
                if s != u:
                    out.append((FULL_DECK.index(type(FULL_DECK[0])(high, s)), FULL_DECK.index(type(FULL_DECK[0])(low, u))))
    return out


def _edge(h0, h1, rng, samples):
    """P(win) - P(lose) for h0 against h1 over sampled five-card runouts, Python evaluator."""
    from engine.hand_eval_fast import score_hand_7_fast
    used = set(h0) | set(h1)
    deck = np.array([i for i in range(52) if i not in used])
    wins = losses = 0
    for _ in range(samples):
        board = rng.choice(deck, size=5, replace=False)
        r0 = np.array([c % 13 for c in list(h0) + list(board)], dtype=np.int64)
        s0 = np.array([c // 13 for c in list(h0) + list(board)], dtype=np.int64)
        r1 = np.array([c % 13 for c in list(h1) + list(board)], dtype=np.int64)
        s1 = np.array([c // 13 for c in list(h1) + list(board)], dtype=np.int64)
        a, b = score_hand_7_fast(r0, s0), score_hand_7_fast(r1, s1)
        wins += a > b
        losses += a < b
    return (wins - losses) / samples


def main():
    rung = sys.argv[1]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    a = load_strategy(rung)["abstraction"]
    classes = {b: representatives(key) for key, b in a._preflop.items()}
    n = max(classes) + 1
    table = np.zeros((n, n))
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    # The native `allin_edge` with sampled runouts read AA against 72 as +0.69
    # one way and -0.77 the other (19 Sept), so the table comes from the
    # Python evaluator, and only the upper triangle is sampled: the lower is
    # its negative by construction, which the native path did not honour.
    for b0 in range(n):
        for b1 in range(b0, n):
            total, count = 0.0, 0
            for _ in range(8):
                for _ in range(20):
                    h0 = classes[b0][rng.integers(len(classes[b0]))]
                    h1 = classes[b1][rng.integers(len(classes[b1]))]
                    if not set(h0) & set(h1):
                        break
                else:
                    continue
                total += _edge(h0, h1, rng, samples)
                count += 1
            table[b0, b1] = total / count if count else 0.0
            table[b1, b0] = -table[b0, b1]
        if b0 % 10 == 0 or b0 == n - 1:
            taken = time.perf_counter() - started
            done = (b0 + 1) * (2 * n - b0) / 2
            print(f"  {b0 + 1}/{n} rows  {taken:.0f}s  eta {(n * (n + 1) / 2 - done) * taken / done / 60:.1f} min", flush=True)
    out = os.path.join(ROOT, "results", "cfr", "chance", os.path.basename(rung).replace(".pkl", "_preflop_allin.npy"))
    np.save(out, table)
    inv = {b: key for key, b in a._preflop.items()}
    aa = next(b for b, k in inv.items() if k == ("A", "A", False)); s72 = next(b for b, k in inv.items() if k == ("7", "2", False))
    print(f"wrote {out}: AA vs 72o {table[aa, s72]:+.3f} (true about +0.65), mean {table.mean():+.4f}")


if __name__ == "__main__":
    main()
