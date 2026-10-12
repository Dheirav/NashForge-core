"""Small policy helpers the Slumbot player shares with nothing else: the strength class a lookup used, a call priced as the
chips we can actually put in, and a crude fallback for a node the strategy never stored."""
from __future__ import annotations

from typing import Optional

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, RAISE_POT


def strength_class(abstraction, hole, board) -> int:
    """The hand's strength class, 0 to 5, as the solver reads it (seeded from the cards, as `cfr_agent` seeds)."""
    key = (tuple(c.index for c in hole), tuple(c.index for c in board))
    bucket = abstraction.bucket(hole, board, np.random.default_rng(hash(key) % (2 ** 32)))
    if hasattr(abstraction, "strength_of"):
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
        return abstraction.strength_of(bucket, street)
    return bucket


def capped_call(pot: int, to_call: int, stack: Optional[int]):
    """The pot and the call as they really are when the bet covers our stack: their chips beyond ours come back."""
    if stack is None or stack <= 0 or to_call <= stack:
        return pot, to_call
    return pot - (to_call - stack), stack


def fallback_choice(bucket: int, top: int, mask, pot: int, to_call: int, stack: Optional[int] = None) -> int:
    """A policy for a node the strategy never stored. Crude on purpose: buckets run from 0 (weakest) to `top`."""
    if stack is not None:
        pot, to_call = capped_call(pot, to_call, stack)

    def first_legal(*preferred):
        for action in preferred:
            if mask[action]:
                return int(action)
        return int(np.flatnonzero(mask)[0])

    if bucket >= top:
        return first_legal(RAISE_POT, ALL_IN, CHECK_CALL)
    if to_call > 0 and to_call >= pot - to_call:
        return first_legal(FOLD, CHECK_CALL)          # a bet of the pot or more: only the top class calls
    if bucket >= top - 1:
        return first_legal(CHECK_CALL, FOLD)
    if bucket >= top - 2:
        return first_legal(CHECK_CALL, FOLD) if to_call <= pot * 0.5 else first_legal(FOLD, CHECK_CALL)
    return first_legal(CHECK_CALL, FOLD) if to_call <= pot * 0.15 else first_legal(FOLD, CHECK_CALL)
