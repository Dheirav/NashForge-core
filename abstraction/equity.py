"""
Monte Carlo hand equity, fast enough to build a card abstraction with.

Bucketing needs equity for tens of thousands of situations. The engine already
ships ``hand_strength_vs_random``, but it spends about 265 microseconds per
simulation — roughly twenty times the cost of the two hand evaluations a
simulation actually requires, the rest going on rebuilding a deck and sampling
through Python objects. At that rate a single abstraction would take hours.

This does the same job by drawing the opponent's cards and the remaining board
in one numpy call per simulation and evaluating through the fast 7-card
evaluator, which brings a 200-simulation estimate to a few milliseconds.

Equity here is the probability of winning against ONE uniformly random opponent
hand with the board completed at random, counting ties as half. That is the
standard signal for card abstraction: it is not the same as equity against a
range, but it orders hands sensibly and is what the clustering consumes.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from engine.cards import RANKS, SUITS, Card
from engine.hand_eval import RANK_ORDER
from engine.hand_eval_fast import evaluate_hand_fast, jit, score_hand_7_fast

#: The 52 distinct cards, in a fixed order, built once.
FULL_DECK: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]

#: Index lookup so a Card can be removed from the deck without a linear scan.
_CARD_INDEX = {(card.rank, card.suit): i for i, card in enumerate(FULL_DECK)}

BOARD_SIZE = 5
HOLE_SIZE = 2

#: `FULL_DECK` as two integer arrays, so a rollout never touches a Card object.
#: Public because `scripts/cfr/build_equity_table.py` drives the same rollout
#: from packed card indices and must use the identical mapping.
#: Built once at import: the hot loop indexes these by deck position.
DECK_RANKS = np.array([RANK_ORDER[c.rank] for c in FULL_DECK], dtype=np.int32)
DECK_SUITS = np.array([SUITS.index(c.suit) for c in FULL_DECK], dtype=np.int32)


@jit(nopython=True, cache=True)
def _rollouts(hole_r, hole_s, board_r, board_s, available, samples, draw,
              seed, deck_r, deck_s):
    """
    Every sampled showdown, compiled.

    This loop was 68% of the whole solver's runtime, measured 11 September:
    6,397,998 calls to `evaluate_hand_fast` for 2,000 MCCFR iterations, each one
    building a HandEvalResult carrying five Card objects that nothing here ever
    reads. Equity only needs an ordering, so this uses `score_hand_7` and stays
    in integers throughout.
    """
    board_n = board_r.shape[0]
    runout = draw - 2

    # Sampling lives here rather than in numpy. Drawing `draw` cards used to
    # cost a (samples x 45) matrix of randoms plus an argpartition per estimate,
    # which the 11 September profile put at 3.43s of argpartition alone across
    # 176,106 calls. A partial Fisher-Yates draws the same number of cards
    # without replacement while touching `draw` entries instead of 45.
    np.random.seed(seed)
    pool = available.copy()
    pool_n = pool.shape[0]

    mine_r = np.empty(7, dtype=np.int32)
    mine_s = np.empty(7, dtype=np.int32)
    opp_r = np.empty(7, dtype=np.int32)
    opp_s = np.empty(7, dtype=np.int32)
    mine_r[0] = hole_r[0]; mine_r[1] = hole_r[1]
    mine_s[0] = hole_s[0]; mine_s[1] = hole_s[1]

    wins = 0
    ties = 0
    for i in range(samples):
        # Partial shuffle: after this the first `draw` entries of `pool` are a
        # uniform draw without replacement, and the pool stays valid for the
        # next sample because the swaps only permute it.
        for k in range(draw):
            j = k + np.int64(np.random.random() * (pool_n - k))
            if j >= pool_n:
                j = pool_n - 1
            tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp

        for j in range(board_n):
            mine_r[2 + j] = board_r[j]; mine_s[2 + j] = board_s[j]
            opp_r[2 + j] = board_r[j];  opp_s[2 + j] = board_s[j]
        for k in range(runout):
            card = pool[2 + k]
            mine_r[2 + board_n + k] = deck_r[card]
            mine_s[2 + board_n + k] = deck_s[card]
            opp_r[2 + board_n + k] = deck_r[card]
            opp_s[2 + board_n + k] = deck_s[card]
        first = pool[0]; second = pool[1]
        opp_r[0] = deck_r[first];  opp_s[0] = deck_s[first]
        opp_r[1] = deck_r[second]; opp_s[1] = deck_s[second]

        mine = score_hand_7_fast(mine_r, mine_s)
        theirs = score_hand_7_fast(opp_r, opp_s)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            ties += 1
    return (wins + 0.5 * ties) / samples


def card_index(card: Card) -> int:
    """Position of ``card`` in :data:`FULL_DECK`."""
    return card.index


def remaining_deck(known: Sequence[Card]) -> np.ndarray:
    """Indices of the cards not among ``known``."""
    seen = {card_index(card) for card in known}
    return np.array([i for i in range(len(FULL_DECK)) if i not in seen], dtype=np.int64)


def equity_vs_random(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Probability that ``hole`` beats one random opponent hand, ties counted half.

    Args:
        hole: The player's two cards.
        board: Community cards so far — 0, 3, 4 or 5 of them.
        num_samples: Simulations. 200 gives a standard error near 0.035, which
            is finer than any sensible bucket boundary.
        rng: Generator, for reproducibility.

    Returns:
        Equity in [0, 1].
    """
    rng = rng if rng is not None else np.random.default_rng()
    board = list(board)
    available = remaining_deck(list(hole) + board)
    runout = BOARD_SIZE - len(board)
    draw = HOLE_SIZE + runout

    # Draw every sample in one vectorised step. Calling rng.choice per sample
    # with replace=False permutes the whole remaining deck each time, which cost
    # more than the two hand evaluations the sample exists for. Partitioning a
    # matrix of random keys gives the same uniform draw without replacement for
    # all samples at once.
    # One draw from the caller's generator instead of a whole matrix. Fitting
    # reuses a single generator across situations, so consuming exactly one
    # value per call keeps that stream deterministic and ordered.
    seed = int(rng.integers(0, 2 ** 31 - 1))

    hole = list(hole)
    return _rollouts(
        np.array([RANK_ORDER[c.rank] for c in hole], dtype=np.int32),
        np.array([SUITS.index(c.suit) for c in hole], dtype=np.int32),
        np.array([RANK_ORDER[c.rank] for c in board], dtype=np.int32),
        np.array([SUITS.index(c.suit) for c in board], dtype=np.int32),
        np.ascontiguousarray(available, dtype=np.int64), num_samples, draw,
        seed, DECK_RANKS, DECK_SUITS)


def sample_situations(
    street_board_size: int,
    count: int,
    rng: Optional[np.random.Generator] = None,
):
    """
    Random (hole, board) situations for a street, for fitting an abstraction.

    The flop alone has about 26 million distinct situations, so a clustering is
    fitted on a sample rather than on the whole space. Yields
    ``(hole, board)`` tuples of :class:`Card`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    total = HOLE_SIZE + street_board_size
    for _ in range(count):
        picked = rng.choice(len(FULL_DECK), size=total, replace=False)
        cards = [FULL_DECK[i] for i in picked]
        yield cards[:HOLE_SIZE], cards[HOLE_SIZE:]
