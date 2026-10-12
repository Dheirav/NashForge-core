"""
The chance player of the abstract game, as tables.

    venv/bin/python scripts/cfr/chance_tables.py results/cfr/ladder169l/nolimit_18bb.pkl --deals 1000000

The solver in use samples real cards every iteration and updates only the
nodes that deal reaches; its error shrinks with the square root of the visits,
and on 17 September a cap-2 rung was still 12.7 BB/100 behind the one-raise
rung after 50M iterations. A full-width solver walks every branch every
iteration and needs no sampling, but it cannot deal cards: chance has to be
written down as the probabilities of moving between buckets. This script
counts them from sampled deals through the same abstraction the lookups use,
so the abstract game the full-width solver sees is the one the bot plays.

What is counted, per deal of two hole hands and a five-card board:

- `preflop[b0, b1]`: joint preflop buckets of the two players.
- `flop[b0, b1, t, s0, s1]`: given the preflop pair, the flop's texture class
  and each player's strength class. Texture is shared (it is the board's), so
  a postflop state is (texture, strength0, strength1) rather than two full
  buckets, which is 216 states with six strengths, not 1,296.
- `turn[state, state']`, `river[state, state']`: transitions between postflop
  states, Markov in the state. That forgets the earlier streets, which is the
  same imperfect recall the solver's keys already have (a key is the current
  bucket and the betting, nothing earlier).
- `showdown[state, 3]`: at the river state, how often player 0 wins, ties, loses.

Counts are stored, not probabilities, so tables from several runs add. The
postflop bucket is seeded from the cards exactly as `cfr_agent` seeds it, so a
situation buckets here the way it buckets in play.

`tests/test_chance_tables.py` checks a table against fresh deals: the state
marginals and the showdown rates must agree within sampling error, and a
transition that lowers the flush class must have zero count.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from abstraction.equity import FULL_DECK  # noqa: E402
from cfr.flat import load_strategy  # noqa: E402
from engine.hand_eval_fast import score_hand_7_fast  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREET_CARDS = {"flop": 3, "turn": 4, "river": 5}

_abstraction = None      # set per worker by `_init`


def _init(pickle_path):
    global _abstraction
    _abstraction = load_strategy(pickle_path, prefer_flat=True)["abstraction"]


def _bucket(hole, board):
    """`cfr_agent.bucket_for`: seeded from the cards, so play and tables agree."""
    key = (tuple(c.index for c in hole), tuple(c.index for c in board))
    return _abstraction.bucket(hole, board, np.random.default_rng(hash(key) % (2 ** 32)))


def _state(bucket, strengths):
    """(texture, strength) from a postflop bucket, as `CardAbstraction.bucket` packs it."""
    return bucket // strengths, bucket % strengths


def count(args):
    """Tables from `deals` deals, seeded; one worker's share."""
    seed, deals, pickle_path = args
    if _abstraction is None:
        _init(pickle_path)
    a = _abstraction
    pre_n = a.num_buckets("preflop")
    strengths = len(a._centroid_list["flop"])
    textures = 6 if getattr(a, "texture", False) else 1
    states = textures * strengths * strengths

    preflop = np.zeros((pre_n, pre_n), dtype=np.int64)
    flop = np.zeros((pre_n, pre_n, textures, strengths, strengths), dtype=np.int64)
    turn = np.zeros((states, states), dtype=np.int64)
    river = np.zeros((states, states), dtype=np.int64)
    showdown = np.zeros((states, 3), dtype=np.int64)

    rng = np.random.default_rng(seed)
    deck = np.arange(52)
    for _ in range(deals):
        rng.shuffle(deck)
        cards = [FULL_DECK[i] for i in deck[:9]]
        hole = (cards[0:2], cards[2:4])
        board = cards[4:9]
        b_pre = (_bucket(hole[0], []), _bucket(hole[1], []))
        preflop[b_pre] += 1
        previous = None
        for street, n in STREET_CARDS.items():
            b = (_bucket(hole[0], board[:n]), _bucket(hole[1], board[:n]))
            t0, s0 = _state(b[0], strengths)
            t1, s1 = _state(b[1], strengths)
            if t0 != t1:
                raise RuntimeError(f"texture differs between players on the same board: {b}")
            state = (t0 * strengths + s0) * strengths + s1
            if street == "flop":
                flop[b_pre[0], b_pre[1], t0, s0, s1] += 1
            elif street == "turn":
                turn[previous, state] += 1
            else:
                river[previous, state] += 1
                score = tuple(int(score_hand_7_fast(
                    np.array([c.index % 13 for c in h + board], dtype=np.int64),
                    np.array([c.index // 13 for c in h + board], dtype=np.int64))) for h in hole)
                showdown[state, 0 if score[0] > score[1] else (1 if score[0] == score[1] else 2)] += 1
            previous = state
    return preflop, flop, turn, river, showdown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickle", help="a solver pickle; its abstraction is the one counted")
    parser.add_argument("--deals", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", help="default results/cfr/chance/<pickle name>.npz")
    args = parser.parse_args()
    out = args.output or os.path.join(ROOT, "results", "cfr", "chance",
                                      os.path.basename(args.pickle).replace(".pkl", ".npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    chunks = max(args.workers * 4, 1)
    per = args.deals // chunks
    jobs = [(args.seed * 100_003 + k, per, args.pickle) for k in range(chunks)]
    started = time.perf_counter()
    totals = None
    done = 0
    with Pool(args.workers, initializer=_init, initargs=(args.pickle,)) as pool:
        for tables in pool.imap_unordered(count, jobs):
            totals = tables if totals is None else tuple(t + u for t, u in zip(totals, tables))
            done += per
            taken = time.perf_counter() - started
            # ETA from the measured rate, never estimated up front.
            print(f"  {done:>10,}/{per * chunks:,} deals  {taken:6.0f}s  "
                  f"eta {(per * chunks - done) * taken / done / 60:5.1f} min", flush=True)
    preflop, flop, turn, river, showdown = totals
    np.savez_compressed(out, preflop=preflop, flop=flop, turn=turn, river=river, showdown=showdown,
                        deals=per * chunks, seed=args.seed, source=os.path.relpath(args.pickle, ROOT))
    states = turn.shape[0]
    reached = int((turn.sum(axis=1) > 0).sum())
    print(f"wrote {out}: {per * chunks:,} deals, {preflop.shape[0]} preflop buckets, "
          f"{states} postflop states ({reached} reached on the flop), "
          f"showdown P(win) {showdown[:, 0].sum() / max(1, showdown.sum()):.3f}")


if __name__ == "__main__":
    main()
