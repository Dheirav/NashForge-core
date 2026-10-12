"""
Suit-isomorphic identity for a (hole, board) situation.

Two situations that differ only in which suit is which are the same question:
equity is invariant under permuting all four suits together, so a hand on a
rainbow board plays identically whether the ace is a spade or a heart. Poker
engines exploit this to shrink precomputed tables; Slumbot counts its boards
"after accounting for suit isomorphism" for this reason.

Why this exists
---------------
`docs/optimisation-plan.md` measured two problems that turn out to be one. We
compute equity *during traversal*, about 43 times per iteration, at 40 Monte
Carlo samples. That is 44% of runtime, and the resulting estimate has a standard
deviation of 0.0667 against bucket centroids spaced 0.101 to 0.170 apart, so 42%
of situations land in the wrong bucket. Precomputing buckets offline fixes both,
because a table can afford a sample count no traversal could.

The table is only affordable because of the collapse this module performs:

    flop     25,989,600 raw  ->   1,286,792 canonical, measured, a 20.20x collapse
    turn    305,377,800 raw  ->  ~15,100,000 canonical at the same factor
    river 2,809,475,760 raw  -> ~139,000,000 canonical, too many for a table

1,286,792 is the published count of suit-isomorphic hole-plus-flop combinations,
so matching it exactly checks this module against the literature rather than only
against itself.

Deck index is ``suit * 13 + rank``.

Not to be confused with the reverted experiment
-----------------------------------------------
Canonicalising the solver's *runtime* cache was tried on 11 September and was a
regression: the miss rate there is dominated by the size of the situation space
rather than by suit redundancy, so the hit rate barely moved while the
canonicalisation cost landed on every miss. Shrinking a precomputed table is a
different use and the economics are the opposite way round, since the cost is
paid once offline rather than per lookup.
"""
from __future__ import annotations

import numpy as np

from engine.hand_eval_fast import jit

#: Cards in a full board. A key packs hole plus board as 6-bit card indices.
MAX_CARDS = 7


@jit(nopython=True, cache=True)
def unpack_key(key, hole_out, board_out):
    """
    The relabelled cards a key was packed from, board first out of the low bits.

    A table build reads keys back rather than carrying the cards alongside them,
    which halves the memory the enumeration needs at its peak.
    """
    for i in range(board_out.shape[0] - 1, -1, -1):
        board_out[i] = key & np.int64(63)
        key >>= np.int64(6)
    for i in range(hole_out.shape[0] - 1, -1, -1):
        hole_out[i] = key & np.int64(63)
        key >>= np.int64(6)


@jit(nopython=True, cache=True)
def canonical_key(hole, board):
    """
    One integer identifying a situation up to a relabelling of the suits.

    Suits are ordered by what they hold, hole cards ranked above board cards so
    that a private ace is never confused with a public one, then relabelled in
    that order. Suits holding identical sets are interchangeable, so breaking
    the tie between them by index cannot make the result depend on which of two
    isomorphic situations asked.

    The key packs the relabelled cards, hole first and each group sorted, as
    6-bit fields. Seven cards fit in 42 bits, so an int64 is ample and the key
    doubles as a sort order.
    """
    signature = np.zeros(4, dtype=np.int64)
    for i in range(hole.shape[0]):
        # Hole ranks occupy the high half so that a suit holding a hole card
        # always outranks one holding only board cards.
        signature[hole[i] // 13] |= np.int64(1) << np.int64(hole[i] % 13 + 13)
    for i in range(board.shape[0]):
        signature[board[i] // 13] |= np.int64(1) << np.int64(board[i] % 13)

    order = np.argsort(-signature)
    relabel = np.zeros(4, dtype=np.int64)
    for new_suit in range(4):
        relabel[order[new_suit]] = new_suit

    mapped_hole = np.empty(hole.shape[0], dtype=np.int64)
    for i in range(hole.shape[0]):
        mapped_hole[i] = relabel[hole[i] // 13] * 13 + hole[i] % 13
    mapped_board = np.empty(board.shape[0], dtype=np.int64)
    for i in range(board.shape[0]):
        mapped_board[i] = relabel[board[i] // 13] * 13 + board[i] % 13

    mapped_hole.sort()
    mapped_board.sort()

    key = np.int64(0)
    for i in range(mapped_hole.shape[0]):
        key = (key << np.int64(6)) | mapped_hole[i]
    for i in range(mapped_board.shape[0]):
        key = (key << np.int64(6)) | mapped_board[i]
    return key


@jit(nopython=True, cache=True)
def table_value(keys, values, hole, board):
    """
    A precomputed value for a situation, or -1.0 when the table lacks it.

    The whole lookup lives in compiled code on purpose. Routing it through
    `Card` objects and separate numpy calls was measured on 11 September at
    8.67 ms/iteration against 7.57 for simply sampling: five dict lookups, two
    array constructions and two numba dispatches per lookup cost more than the
    40-sample rollout they replaced. The caller already holds card indices, so
    all of that was waste.

    Binary search rather than `np.searchsorted` for the same reason: one
    compiled function, not a dispatch per call.
    """
    key = canonical_key(hole, board)
    low = 0
    high = keys.shape[0]
    while low < high:
        middle = (low + high) // 2
        if keys[middle] < key:
            low = middle + 1
        else:
            high = middle
    if low >= keys.shape[0] or keys[low] != key:
        return -1.0
    return values[low]


@jit(nopython=True, cache=True)
def nearest_centroid(centroids, value):
    """Index of the closest centroid. Ties go to the lower index, as bisect does."""
    best = 0
    best_distance = abs(centroids[0] - value)
    for i in range(1, centroids.shape[0]):
        distance = abs(centroids[i] - value)
        if distance < best_distance:
            best = i
            best_distance = distance
    return best


# ---------------------------------------------------------------------------
# Open-addressed lookup
# ---------------------------------------------------------------------------
#
# A sorted key array with binary search was the obvious structure and it lost to
# simply sampling: measured 11 September at 15.77 us per lookup on random keys
# against 1.00 us on one repeated key, and 10.29 us for the 40-sample rollout it
# was meant to replace. Twenty-one probes across a 10.3 MB array are twenty-one
# cache misses. The search was never the cost; the memory was.
#
# Linear probing at half load turns that into one or two probes. It costs about
# twice the memory and is worth it. A perfect index, which is what serious
# engines use, would remove the key array altogether; this is the cheap version
# of the same idea.


@jit(nopython=True, cache=True)
def _mix(key):
    """splitmix64 finaliser: scatters keys whose low bits are highly structured."""
    z = np.uint64(key) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@jit(nopython=True, cache=True)
def build_hash(keys, values, slot_keys, slot_values):
    """Fill an open-addressed table. `slot_keys` must start filled with -1."""
    mask = np.uint64(slot_keys.shape[0] - 1)
    for i in range(keys.shape[0]):
        slot = np.int64(_mix(keys[i]) & mask)
        while slot_keys[slot] != -1:
            slot += 1
            if slot == slot_keys.shape[0]:
                slot = 0
        slot_keys[slot] = keys[i]
        slot_values[slot] = values[i]


@jit(nopython=True, cache=True)
def hash_value(slot_keys, slot_values, hole, board):
    """A situation's precomputed value, or -1.0 when absent."""
    key = canonical_key(hole, board)
    mask = np.uint64(slot_keys.shape[0] - 1)
    slot = np.int64(_mix(key) & mask)
    while True:
        held = slot_keys[slot]
        if held == key:
            return slot_values[slot]
        if held == -1:
            return -1.0
        slot += 1
        if slot == slot_keys.shape[0]:
            slot = 0
