import numpy as np
from typing import List, Tuple

SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

class Card:
    def __init__(self, rank: str, suit: str):
        assert rank in RANKS, f"Invalid rank: {rank}"
        assert suit in SUITS, f"Invalid suit: {suit}"
        self.rank = rank
        self.suit = suit
        #: Position in the standard deck, suit * 13 + rank. Computed once here
        #: rather than looked up per use: the 52 cards are built at import and
        #: shared by every deck, so this costs 52 multiplications in total and
        #: turns `card_index(card)` from a tuple build plus a dict lookup into
        #: an attribute read, on a path that runs several times per decision.
        self.index = SUITS.index(suit) * 13 + RANKS.index(rank)

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

# The 52 distinct cards, built once. A Card carries only a rank and a suit and
# is never mutated after construction — every use site compares, none assigns —
# so every deck can share these instances instead of allocating its own.
#
# This matters because a deck is rebuilt for each hand: a single profiled
# generation constructed 63,000 decks and therefore 3.3 million Card objects,
# about 2.6 s of the run. Copying the tuple into a fresh list keeps each deck's
# ordering private, so shuffling and dealing are unchanged.
_STANDARD_DECK = tuple(Card(rank, suit) for suit in SUITS for rank in RANKS)


class Deck:
    def __init__(self, seed: int = None):
        self.cards = list(_STANDARD_DECK)
        # Use numpy RNG for faster shuffling (1.05-1.1× speedup)
        self._rng = np.random.default_rng(seed)
        self.shuffle()

    def shuffle(self):
        # Use numpy shuffle - faster than Python random.shuffle
        self._rng.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        assert n <= len(self.cards), "Not enough cards to deal"
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def __len__(self):
        return len(self.cards)

