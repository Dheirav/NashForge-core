"""
Fast hand evaluation using iterative approach instead of checking all combinations.
This is 10-20x faster than the combinatorial approach.
Optimized with Numba JIT compilation for 2-3× additional speedup.
"""
import numpy as np
from typing import List, Tuple
from collections import Counter
from .cards import Card, RANKS
from .hand_eval import HandEvalResult, RANK_ORDER, VALUE_TO_RANK

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, fastmath=True)
def find_straight_jit(rank_values):
    """
    JIT-compiled straight detection.
    Returns (has_straight, high_card_value).
    """
    # Get unique values and sort descending
    unique = np.unique(rank_values)[::-1]
    
    # Check regular straights (5 consecutive cards)
    for i in range(len(unique) - 4):
        if unique[i] - unique[i+4] == 4:
            return True, unique[i]
    
    # Check wheel (A-2-3-4-5): values 12,0,1,2,3
    has_ace = 12 in unique
    has_five = 3 in unique
    has_four = 2 in unique
    has_three = 1 in unique
    has_two = 0 in unique
    
    if has_ace and has_two and has_three and has_four and has_five:
        return True, 3  # 5-high straight
    
    return False, -1


@jit(nopython=True, cache=True, fastmath=True)
def count_ranks_jit(rank_values):
    """
    JIT-compiled rank counting.
    Returns array where index=rank_value, value=count.
    """
    counts = np.zeros(13, dtype=np.int32)
    for val in rank_values:
        counts[val] += 1
    return counts


@jit(nopython=True, cache=True, fastmath=True)
def count_suits_jit(suit_indices):
    """
    JIT-compiled suit counting.
    Returns array where index=suit_index, value=count.
    """
    counts = np.zeros(4, dtype=np.int32)
    for suit_idx in suit_indices:
        counts[suit_idx] += 1
    return counts


def evaluate_hand_fast(cards: List[Card]) -> HandEvalResult:
    """
    Fast 7-card hand evaluation without checking all 21 combinations.
    Uses an iterative approach to find the best hand directly.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    
    # Sort cards by rank (descending)
    sorted_cards = sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)
    
    # Count ranks and suits in a single pass.
    #
    # This was two Counter() constructions. Both are only ever read through
    # .items(), and a plain dict built in first-encounter order has identical
    # contents AND identical iteration order, so the results are unchanged —
    # but it costs 0.61 us instead of 1.91 us. That matters because this is the
    # innermost function in the project: one abstraction lookup runs it eighty
    # times, and a profiled MCCFR run spent 5 of 20 seconds inside Counter
    # alone.
    rank_counts = {}
    suit_counts = {}
    for card in cards:
        rank, suit = card.rank, card.suit
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    rank_values = [RANK_ORDER[c.rank] for c in sorted_cards]
    
    # Check for flush
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    # Check for straight
    def find_straight(vals: List[int]) -> Tuple[bool, int]:
        """Returns (has_straight, high_card_value)"""
        unique = sorted(set(vals), reverse=True)
        # Check regular straights
        for i in range(len(unique) - 4):
            if unique[i] - unique[i+4] == 4:
                return True, unique[i]
        # Check wheel (A-2-3-4-5)
        if set([12, 0, 1, 2, 3]).issubset(set(unique)):
            return True, 3  # 5-high
        return False, -1
    
    has_straight, straight_high = find_straight(rank_values)
    
    # Check for straight flush / royal flush
    if flush_suit:
        flush_cards = [c for c in cards if c.suit == flush_suit]
        flush_vals = [RANK_ORDER[c.rank] for c in flush_cards]
        has_sf, sf_high = find_straight(flush_vals)
        if has_sf:
            # Get the 5 cards in the straight flush
            sf_cards = get_straight_cards(flush_cards, sf_high)
            if sf_high == 12:  # Ace-high
                return HandEvalResult(9, [sf_high], sf_cards[:5])  # Royal flush
            return HandEvalResult(8, [sf_high], sf_cards[:5])  # Straight flush
    
    # Check for quads
    quads = [r for r, c in rank_counts.items() if c == 4]
    if quads:
        quad_rank = quads[0]
        quad_cards = [c for c in sorted_cards if c.rank == quad_rank]
        kicker = [c for c in sorted_cards if c.rank != quad_rank][0]
        return HandEvalResult(7, [RANK_ORDER[quad_rank], RANK_ORDER[kicker.rank]], 
                            quad_cards + [kicker])
    
    # Check for full house
    trips = [r for r, c in rank_counts.items() if c == 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    
    if trips:
        # Sort trips by rank value
        trips.sort(key=lambda r: RANK_ORDER[r], reverse=True)
        trip_rank = trips[0]
        trip_cards = [c for c in sorted_cards if c.rank == trip_rank]
        
        # Find best pair (could be second set of trips)
        pair_candidates = []
        if len(trips) > 1:
            pair_candidates.append(trips[1])
        pair_candidates.extend(pairs)
        
        if pair_candidates:
            pair_candidates.sort(key=lambda r: RANK_ORDER[r], reverse=True)
            pair_rank = pair_candidates[0]
            pair_cards = [c for c in sorted_cards if c.rank == pair_rank][:2]
            return HandEvalResult(6, [RANK_ORDER[trip_rank], RANK_ORDER[pair_rank]], 
                                trip_cards + pair_cards)
    
    # Check for flush
    if flush_suit:
        flush_cards = sorted([c for c in cards if c.suit == flush_suit], 
                           key=lambda c: RANK_ORDER[c.rank], reverse=True)[:5]
        return HandEvalResult(5, [RANK_ORDER[c.rank] for c in flush_cards], flush_cards)
    
    # Check for straight
    if has_straight:
        straight_cards = get_straight_cards(sorted_cards, straight_high)
        return HandEvalResult(4, [straight_high], straight_cards[:5])
    
    # Three of a kind
    if trips:
        trip_rank = trips[0]
        trip_cards = [c for c in sorted_cards if c.rank == trip_rank]
        kickers = [c for c in sorted_cards if c.rank != trip_rank][:2]
        return HandEvalResult(3, [RANK_ORDER[trip_rank]] + [RANK_ORDER[k.rank] for k in kickers],
                            trip_cards + kickers)
    
    # Two pair
    if len(pairs) >= 2:
        pairs.sort(key=lambda r: RANK_ORDER[r], reverse=True)
        pair1, pair2 = pairs[0], pairs[1]
        pair1_cards = [c for c in sorted_cards if c.rank == pair1][:2]
        pair2_cards = [c for c in sorted_cards if c.rank == pair2][:2]
        kicker = [c for c in sorted_cards if c.rank not in [pair1, pair2]][0]
        return HandEvalResult(2, [RANK_ORDER[pair1], RANK_ORDER[pair2], RANK_ORDER[kicker.rank]],
                            pair1_cards + pair2_cards + [kicker])
    
    # One pair
    if pairs:
        pair_rank = pairs[0]
        pair_cards = [c for c in sorted_cards if c.rank == pair_rank][:2]
        kickers = [c for c in sorted_cards if c.rank != pair_rank][:3]
        return HandEvalResult(1, [RANK_ORDER[pair_rank]] + [RANK_ORDER[k.rank] for k in kickers],
                            pair_cards + kickers)
    
    # High card
    return HandEvalResult(0, [RANK_ORDER[c.rank] for c in sorted_cards[:5]], sorted_cards[:5])


def get_straight_cards(cards: List[Card], high_val: int) -> List[Card]:
    """Extract cards forming a straight with given high card value."""
    if high_val == 3:  # Wheel: A-2-3-4-5
        needed = {12, 0, 1, 2, 3}
    else:
        needed = set(range(high_val - 4, high_val + 1))
    
    result = []
    used = set()
    for c in sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True):
        val = RANK_ORDER[c.rank]
        if val in needed and val not in used:
            result.append(c)
            used.add(val)
            if len(result) == 5:
                break
    return result


def compare_hands_fast(hands: List[List[Card]]) -> List[int]:
    """
    Fast comparison of multiple hands.
    Returns indices of winning hand(s).
    """
    evals = [evaluate_hand_fast(h) for h in hands]
    best = max(evals)
    return [i for i, e in enumerate(evals) if e == best]


# ---------------------------------------------------------------------------
# The comparable score, for the hot path
# ---------------------------------------------------------------------------
#
# `evaluate_hand_fast` builds a HandEvalResult carrying the best five Card
# objects. Showdowns need that. Equity rollouts do not: `equity_vs_random` only
# ever asks `mine > theirs` and `mine == theirs`, and a profile on 11 September
# put 68% of the whole solver's runtime inside `evaluate_hand_fast`, over 6.4
# million calls for 2,000 MCCFR iterations.
#
# So the hot path gets a second function that returns one integer instead, which
# is what numba can compile. HandEvalResult compares lexicographically on
# (hand_rank, tiebreaker), and within a hand class the tiebreaker length is
# fixed, so packing class and five 4-bit tiebreakers into an int reproduces that
# ordering exactly. `test_hand_eval_fast.py` pins the two against each other
# over random hands, which is the only thing making this safe to use.


@jit(nopython=True, cache=True)
def _straight_high(present):
    """Highest value completing a straight, or -1. `present` is 13 flags."""
    for value in range(12, 3, -1):
        run = True
        for step in range(5):
            if present[value - step] == 0:
                run = False
                break
        if run:
            return value
    if present[12] and present[0] and present[1] and present[2] and present[3]:
        return 3                                    # the wheel, A-2-3-4-5
    return -1


@jit(nopython=True, cache=True)
def score_hand_7(ranks, suits):
    """
    One integer ordering seven cards the way `evaluate_hand_fast` orders them.

    `ranks` are 0-12 and `suits` 0-3. Returns class in the high bits and up to
    five tiebreakers below it, so plain integer comparison matches
    HandEvalResult's `__lt__`.
    """
    rank_counts = np.zeros(13, dtype=np.int32)
    suit_counts = np.zeros(4, dtype=np.int32)
    present = np.zeros(13, dtype=np.int32)
    for i in range(ranks.shape[0]):
        rank_counts[ranks[i]] += 1
        suit_counts[suits[i]] += 1
        present[ranks[i]] = 1

    flush_suit = -1
    for suit in range(4):
        if suit_counts[suit] >= 5:
            flush_suit = suit
            break

    t0 = t1 = t2 = t3 = t4 = 0
    cls = 0

    if flush_suit >= 0:
        flush_present = np.zeros(13, dtype=np.int32)
        for i in range(ranks.shape[0]):
            if suits[i] == flush_suit:
                flush_present[ranks[i]] = 1
        sf_high = _straight_high(flush_present)
        if sf_high >= 0:
            return ((9 if sf_high == 12 else 8) << 20) | (sf_high << 16)

    quad = -1
    trip_hi = -1
    trip_lo = -1
    pair_hi = -1
    pair_lo = -1
    for value in range(12, -1, -1):
        count = rank_counts[value]
        if count == 4 and quad < 0:
            quad = value
        elif count == 3:
            if trip_hi < 0:
                trip_hi = value
            elif trip_lo < 0:
                trip_lo = value
        elif count == 2:
            if pair_hi < 0:
                pair_hi = value
            elif pair_lo < 0:
                pair_lo = value

    if quad >= 0:
        kicker = -1
        for value in range(12, -1, -1):
            if value != quad and rank_counts[value] > 0:
                kicker = value
                break
        return (7 << 20) | (quad << 16) | (kicker << 12)

    if trip_hi >= 0:
        # A second trip plays as the pair, and outranks any actual pair because
        # the scan above runs high to low.
        pair = trip_lo if trip_lo > pair_hi else pair_hi
        if pair >= 0:
            return (6 << 20) | (trip_hi << 16) | (pair << 12)

    if flush_suit >= 0:
        cls = 5
        taken = 0
        for value in range(12, -1, -1):
            if taken == 5:
                break
            count = 0
            for i in range(ranks.shape[0]):
                if suits[i] == flush_suit and ranks[i] == value:
                    count += 1
            if count > 0:
                if taken == 0:   t0 = value
                elif taken == 1: t1 = value
                elif taken == 2: t2 = value
                elif taken == 3: t3 = value
                else:            t4 = value
                taken += 1
        return (cls << 20) | (t0 << 16) | (t1 << 12) | (t2 << 8) | (t3 << 4) | t4

    straight = _straight_high(present)
    if straight >= 0:
        return (4 << 20) | (straight << 16)

    if trip_hi >= 0:
        kickers = 0
        for value in range(12, -1, -1):
            if value != trip_hi and rank_counts[value] > 0:
                if kickers == 0:   t1 = value
                elif kickers == 1: t2 = value
                else:              break
                kickers += 1
        return (3 << 20) | (trip_hi << 16) | (t1 << 12) | (t2 << 8)

    if pair_hi >= 0 and pair_lo >= 0:
        kicker = -1
        for value in range(12, -1, -1):
            if value != pair_hi and value != pair_lo and rank_counts[value] > 0:
                kicker = value
                break
        return (2 << 20) | (pair_hi << 16) | (pair_lo << 12) | (kicker << 8)

    if pair_hi >= 0:
        kickers = 0
        for value in range(12, -1, -1):
            if value != pair_hi and rank_counts[value] > 0:
                if kickers == 0:   t1 = value
                elif kickers == 1: t2 = value
                elif kickers == 2: t3 = value
                else:              break
                kickers += 1
        return (1 << 20) | (pair_hi << 16) | (t1 << 12) | (t2 << 8) | (t3 << 4)

    taken = 0
    for value in range(12, -1, -1):
        if rank_counts[value] > 0:
            if taken == 0:   t0 = value
            elif taken == 1: t1 = value
            elif taken == 2: t2 = value
            elif taken == 3: t3 = value
            elif taken == 4: t4 = value
            else:            break
            taken += 1
    return (t0 << 16) | (t1 << 12) | (t2 << 8) | (t3 << 4) | t4


# ---------------------------------------------------------------------------
# Rank-mask tables
# ---------------------------------------------------------------------------
#
# `score_hand_7` counts ranks and suits, then scans for a straight, then walks
# the hand classes picking kickers. Measured 11 September at 7.0 M evaluations
# per second compiled, against 80 M for a two-plus-two style lookup table.
#
# The full two-plus-two table is ~130 MB and this project measured today exactly
# why that is a trap: a 50 MB equity table lost to not having one at all,
# because twenty-one cache misses cost more than the work they replaced. So
# these tables are deliberately tiny. Both are indexed by a 13-bit rank mask,
# 8,192 entries each, 40 KB together, which stays resident.
#
#   STRAIGHT_HIGH[mask]  high card of the best straight in those ranks, or -1
#   TOP_FIVE[mask]       the five highest ranks present, packed 4 bits each
#
# The design is what transfers to a port: replace scanning with indexing. The
# sizes are what make it work here.
#
# WHAT THIS IS WORTH, measured 11 September, so nobody spends a week on the
# full two-plus-two table expecting more:
#
#   score_hand_7       6.9 M evals/sec      score_hand_7_fast  9.6 M   1.39x
#   inside _rollouts   13.94 us             13.18 us                   1.06x
#
# and the solver's time divides as:
#
#   traversal floor 40.5%   equity 44.3%, of which rollout 20.8% of total,
#   of which evaluation 17.3% of total
#
# So evaluation is 17.3% of runtime. A FREE evaluator would be 1.21x overall and
# a two-plus-two at 80 M/s would be 1.19x. This one is about 1.01x. The evaluator
# is not where the time is, and no evaluator work can be, until the traversal
# floor is gone. That is a port, not an optimisation.


def _build_rank_tables():
    straight = np.full(8192, -1, dtype=np.int32)   # int32, not int8: without numba, `high << 16` overflowed an int8
    top_five = np.zeros(8192, dtype=np.int32)
    for mask in range(8192):
        present = [r for r in range(12, -1, -1) if mask & (1 << r)]

        high = -1
        for value in range(12, 3, -1):
            if all(mask & (1 << (value - step)) for step in range(5)):
                high = value
                break
        if high < 0 and all(mask & (1 << r) for r in (12, 0, 1, 2, 3)):
            high = 3                              # the wheel, A-2-3-4-5
        straight[mask] = high

        packed = 0
        for i in range(5):
            packed = (packed << 4) | (present[i] if i < len(present) else 0)
        top_five[mask] = packed
    return straight, top_five


STRAIGHT_HIGH, TOP_FIVE = _build_rank_tables()


@jit(nopython=True, cache=True)
def score_hand_7_fast(ranks, suits):
    """
    `score_hand_7` with the scans replaced by two table reads.

    Returns the identical packed score, which `tests/test_abstraction.py` pins
    against the full evaluator. A hand evaluator that disagrees with itself
    would not crash, it would quietly shift every equity estimate.
    """
    rank_counts = np.zeros(13, dtype=np.int32)
    suit_masks = np.zeros(4, dtype=np.int32)
    suit_counts = np.zeros(4, dtype=np.int32)
    mask = 0
    for i in range(ranks.shape[0]):
        rank = ranks[i]
        suit = suits[i]
        rank_counts[rank] += 1
        suit_masks[suit] |= 1 << rank
        suit_counts[suit] += 1
        mask |= 1 << rank

    flush_suit = -1
    for suit in range(4):
        if suit_counts[suit] >= 5:
            flush_suit = suit
            break

    if flush_suit >= 0:
        flush_mask = suit_masks[flush_suit]
        high = STRAIGHT_HIGH[flush_mask]
        if high >= 0:
            return ((9 if high == 12 else 8) << 20) | (high << 16)

    quad = -1
    trip_hi = -1
    trip_lo = -1
    pair_hi = -1
    pair_lo = -1
    for value in range(12, -1, -1):
        count = rank_counts[value]
        if count == 4 and quad < 0:
            quad = value
        elif count == 3:
            if trip_hi < 0:
                trip_hi = value
            elif trip_lo < 0:
                trip_lo = value
        elif count == 2:
            if pair_hi < 0:
                pair_hi = value
            elif pair_lo < 0:
                pair_lo = value

    if quad >= 0:
        kicker = TOP_FIVE[mask & ~(1 << quad)] >> 16
        return (7 << 20) | (quad << 16) | (kicker << 12)

    if trip_hi >= 0:
        pair = trip_lo if trip_lo > pair_hi else pair_hi
        if pair >= 0:
            return (6 << 20) | (trip_hi << 16) | (pair << 12)

    if flush_suit >= 0:
        return (5 << 20) | TOP_FIVE[suit_masks[flush_suit]]

    high = STRAIGHT_HIGH[mask]
    if high >= 0:
        return (4 << 20) | (high << 16)

    if trip_hi >= 0:
        kickers = TOP_FIVE[mask & ~(1 << trip_hi)]
        return ((3 << 20) | (trip_hi << 16)
                | (((kickers >> 16) & 15) << 12) | (((kickers >> 12) & 15) << 8))

    if pair_hi >= 0 and pair_lo >= 0:
        kicker = TOP_FIVE[mask & ~(1 << pair_hi) & ~(1 << pair_lo)] >> 16
        return (2 << 20) | (pair_hi << 16) | (pair_lo << 12) | (kicker << 8)

    if pair_hi >= 0:
        kickers = TOP_FIVE[mask & ~(1 << pair_hi)]
        return ((1 << 20) | (pair_hi << 16)
                | (((kickers >> 16) & 15) << 12) | (((kickers >> 12) & 15) << 8)
                | (((kickers >> 8) & 15) << 4))

    return TOP_FIVE[mask]
