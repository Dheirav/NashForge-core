from typing import List, Tuple
from collections import Counter
from .cards import Card, RANKS, SUITS

RANK_ORDER = {r: i for i, r in enumerate(RANKS)}
# Reverse lookup: value -> rank string
VALUE_TO_RANK = {i: r for i, r in enumerate(RANKS)}

HAND_RANKS = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",  # Special case of straight flush with A-high
]

class HandEvalResult:
    def __init__(self, hand_rank: int, tiebreaker: List[int], hand: List[Card]):
        self.hand_rank = hand_rank  # 0 = High Card, 8 = Straight Flush, 9 = Royal Flush
        self.tiebreaker = tiebreaker  # List of rank indices for tie-breaking
        self.hand = hand  # The best 5-card hand

    def __lt__(self, other):
        if self.hand_rank != other.hand_rank:
            return self.hand_rank < other.hand_rank
        return self.tiebreaker < other.tiebreaker

    def __gt__(self, other):
        if self.hand_rank != other.hand_rank:
            return self.hand_rank > other.hand_rank
        return self.tiebreaker > other.tiebreaker

    def __eq__(self, other):
        return self.hand_rank == other.hand_rank and self.tiebreaker == other.tiebreaker

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or self > other

    def __repr__(self):
        return f"{HAND_RANKS[self.hand_rank]}: {self.hand}"

def evaluate_hand(cards: List[Card]) -> HandEvalResult:
    """
    Given 7 cards (2 hole + 5 community), return the best 5-card hand.
    """
    from itertools import combinations
    best = None
    for combo in combinations(cards, 5):
        result = rank_five_card_hand(list(combo))
        if best is None or result > best:
            best = result
    return best

def rank_five_card_hand(cards: List[Card]) -> HandEvalResult:
    # Count suits and ranks
    suits = [c.suit for c in cards]
    ranks = [c.rank for c in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    rank_values = sorted([RANK_ORDER[r] for r in ranks], reverse=True)
    
    # Check for flush
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    # Check for straight (and straight flush)
    def get_straight_high(vals):
        """
        Returns the high card value of a straight, or None if no straight.
        For wheel (A-2-3-4-5), returns 3 (the '5' card value, making it the lowest straight).
        """
        unique_vals = sorted(set(vals), reverse=True)
        # Check for regular straights
        for i in range(len(unique_vals) - 4):
            window = unique_vals[i:i+5]
            if window[0] - window[4] == 4:
                return window[0]
        # Check for wheel straight (A-2-3-4-5): A=12, 2=0, 3=1, 4=2, 5=3
        if set([12, 0, 1, 2, 3]).issubset(set(unique_vals)):
            return 3  # 5-high straight (lowest possible)
        return None
    
    def get_straight_cards(cards, high_val):
        """
        Get the 5 cards forming a straight with the given high card.
        For wheel, high_val=3 means A-2-3-4-5.
        """
        if high_val == 3 and 12 in [RANK_ORDER[c.rank] for c in cards]:
            # Wheel straight: need A, 2, 3, 4, 5
            needed = {12, 0, 1, 2, 3}  # A, 2, 3, 4, 5
        else:
            needed = set(range(high_val - 4, high_val + 1))
        
        straight_cards = []
        used_vals = set()
        for c in sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True):
            val = RANK_ORDER[c.rank]
            if val in needed and val not in used_vals:
                straight_cards.append(c)
                used_vals.add(val)
        return straight_cards
    
    straight_high = get_straight_high(rank_values)
    
    # Straight flush (and Royal Flush)
    if flush_suit:
        flush_cards = [c for c in cards if c.suit == flush_suit]
        flush_vals = [RANK_ORDER[c.rank] for c in flush_cards]
        sf_high = get_straight_high(flush_vals)
        if sf_high is not None:
            sf_cards = get_straight_cards(flush_cards, sf_high)
            # Royal Flush is A-high straight flush (high card = 12 = Ace)
            if sf_high == 12:
                return HandEvalResult(9, [sf_high], sf_cards)  # Royal Flush
            return HandEvalResult(8, [sf_high], sf_cards)  # Straight Flush
    
    # Four of a kind
    for rank, count in rank_counts.items():
        if count == 4:
            kicker = max([RANK_ORDER[r] for r in ranks if r != rank])
            quads = [c for c in cards if c.rank == rank]
            kicker_card = max([c for c in cards if c.rank != rank], key=lambda c: RANK_ORDER[c.rank])
            return HandEvalResult(7, [RANK_ORDER[rank], kicker], quads + [kicker_card])
    
    # Full house
    trips = [r for r, c in rank_counts.items() if c == 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    if trips:
        trip = max(trips, key=lambda r: RANK_ORDER[r])
        trip_cards = [c for c in cards if c.rank == trip]
        if len(trips) > 1:
            pair = max([r for r in trips if r != trip] + pairs, key=lambda r: RANK_ORDER[r])
            pair_cards = [c for c in cards if c.rank == pair][:2]
            return HandEvalResult(6, [RANK_ORDER[trip], RANK_ORDER[pair]], trip_cards + pair_cards)
        elif pairs:
            pair = max(pairs, key=lambda r: RANK_ORDER[r])
            pair_cards = [c for c in cards if c.rank == pair][:2]
            return HandEvalResult(6, [RANK_ORDER[trip], RANK_ORDER[pair]], trip_cards + pair_cards)
    
    # Flush
    if flush_suit:
        flush_cards = sorted([c for c in cards if c.suit == flush_suit], key=lambda c: RANK_ORDER[c.rank], reverse=True)[:5]
        return HandEvalResult(5, [RANK_ORDER[c.rank] for c in flush_cards], flush_cards)
    
    # Straight
    if straight_high is not None:
        straight_cards = get_straight_cards(cards, straight_high)
        return HandEvalResult(4, [straight_high], straight_cards)
    
    # Three of a kind
    if trips:
        trip = max(trips, key=lambda r: RANK_ORDER[r])
        trip_cards = [c for c in cards if c.rank == trip]
        kicker_cards = sorted([c for c in cards if c.rank != trip], key=lambda c: RANK_ORDER[c.rank], reverse=True)[:2]
        kickers = [RANK_ORDER[c.rank] for c in kicker_cards]
        return HandEvalResult(3, [RANK_ORDER[trip]] + kickers, trip_cards + kicker_cards)
    
    # Two pair
    if len(pairs) >= 2:
        # Sort pairs by rank value, descending
        sorted_pairs = sorted(pairs, key=lambda r: RANK_ORDER[r], reverse=True)[:2]
        top2_vals = [RANK_ORDER[r] for r in sorted_pairs]
        # Find kicker (highest card not in either pair)
        pair_ranks_set = set(sorted_pairs)
        kicker_cards = [c for c in cards if c.rank not in pair_ranks_set]
        kicker_val = max([RANK_ORDER[c.rank] for c in kicker_cards])
        # Build hand
        pair1_cards = [c for c in cards if c.rank == sorted_pairs[0]][:2]
        pair2_cards = [c for c in cards if c.rank == sorted_pairs[1]][:2]
        kicker_card = max(kicker_cards, key=lambda c: RANK_ORDER[c.rank])
        return HandEvalResult(2, top2_vals + [kicker_val], pair1_cards + pair2_cards + [kicker_card])
    
    # One pair
    if len(pairs) == 1:
        pair = pairs[0]
        pair_cards = [c for c in cards if c.rank == pair]
        kicker_cards = sorted([c for c in cards if c.rank != pair], key=lambda c: RANK_ORDER[c.rank], reverse=True)[:3]
        kickers = [RANK_ORDER[c.rank] for c in kicker_cards]
        return HandEvalResult(1, [RANK_ORDER[pair]] + kickers, pair_cards + kicker_cards)
    
    # High card
    sorted_cards = sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)[:5]
    return HandEvalResult(0, [RANK_ORDER[c.rank] for c in sorted_cards], sorted_cards)

def compare_hands(hands: List[List[Card]]) -> List[int]:
    """
    Given a list of hands (each 7 cards), return the indices of the winning hand(s).
    Handles split pots.
    """
    evals = [evaluate_hand(h) for h in hands]
    best = max(evals)
    return [i for i, e in enumerate(evals) if e == best]
