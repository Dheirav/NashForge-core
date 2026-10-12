"""
Kuhn poker — the smallest poker with a known exact solution.

Three cards (J, Q, K), two players, one betting round. Twelve information sets
in total. It exists in this project for one reason: **its equilibrium is known
analytically, so a solver can be proved correct before anything is built on
it.** The game value to the first player is exactly -1/18, and the equilibria
form a one-parameter family in which player 1 bluffs the jack with probability
alpha and bets the king with probability 3*alpha, for alpha in [0, 1/3].

Rules as implemented (the standard formulation the -1/18 value assumes):

* Both players ante 1 chip, so the pot starts at 2.
* Each is dealt one card from a three-card deck.
* Player 0 acts first and may pass or bet 1.
    - pass, pass          -> showdown for a pot of 2
    - pass, bet, pass     -> player 0 folds, player 1 takes the pot
    - pass, bet, bet      -> showdown for a pot of 4
    - bet, pass           -> player 1 folds, player 0 takes the pot
    - bet, bet            -> showdown for a pot of 4

Utilities are net chips relative to the ante: winning an uncontested pot is +1,
winning a showdown after both bet is +2.
"""
from __future__ import annotations

from typing import Hashable, List, NamedTuple, Sequence, Tuple

from .base import CHANCE, Game

#: Actions. 'p' is pass — check when facing no bet, fold when facing one.
#: 'b' is bet — open when facing no bet, call when facing one.
PASS = "p"
BET = "b"
ACTIONS = (PASS, BET)

#: Card ranks, ordered weakest to strongest.
CARD_NAMES = ("J", "Q", "K")


class KuhnState(NamedTuple):
    """
    Immutable game state.

    ``cards`` is empty at the root (before the deal) and holds one card per
    player afterwards. ``history`` is the string of actions taken so far, which
    is public and therefore part of every player's information set.
    """
    cards: Tuple[int, ...]
    history: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        dealt = "".join(CARD_NAMES[c] for c in self.cards) or "-"
        return f"KuhnState({dealt}, {self.history or 'root'})"


#: Histories at which the hand is over.
_TERMINAL = frozenset({"pp", "pbp", "pbb", "bp", "bb"})


class KuhnPoker(Game):
    """Kuhn poker as a :class:`~games.base.Game`."""

    num_players = 2

    def initial_state(self) -> KuhnState:
        return KuhnState(cards=(), history="")

    # ---- structure -------------------------------------------------------

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in _TERMINAL

    def current_player(self, state: KuhnState) -> int:
        if not state.cards:
            return CHANCE
        return len(state.history) % 2

    def legal_actions(self, state: KuhnState) -> Sequence[str]:
        return ACTIONS

    def next_state(self, state: KuhnState, action) -> KuhnState:
        if not state.cards:
            # Chance actions are the dealt pair itself.
            return KuhnState(cards=tuple(action), history="")
        return KuhnState(cards=state.cards, history=state.history + action)

    # ---- chance ----------------------------------------------------------

    def chance_outcomes(self, state: KuhnState) -> Sequence[Tuple[Tuple[int, int], float]]:
        """All six ways to deal two distinct cards from three, uniformly."""
        deals: List[Tuple[Tuple[int, int], float]] = []
        for first in range(3):
            for second in range(3):
                if first != second:
                    deals.append(((first, second), 1.0 / 6.0))
        return deals

    # ---- payoffs and knowledge -------------------------------------------

    def utility(self, state: KuhnState, player: int) -> float:
        history = state.history
        winner_by_card = 0 if state.cards[0] > state.cards[1] else 1

        if history == "pp":              # checked down, pot of 2
            payoff_to_0 = 1.0 if winner_by_card == 0 else -1.0
        elif history == "bp":            # player 1 folded to a bet
            payoff_to_0 = 1.0
        elif history == "pbp":           # player 0 folded to a bet
            payoff_to_0 = -1.0
        else:                            # "bb" or "pbb": showdown, pot of 4
            payoff_to_0 = 2.0 if winner_by_card == 0 else -2.0

        return payoff_to_0 if player == 0 else -payoff_to_0

    def information_set(self, state: KuhnState, player: int) -> Hashable:
        """
        The player's own card plus the public history — never the opponent's card.

        Indexing ``state.cards`` by ``player`` rather than by position is what
        keeps hidden information hidden.
        """
        return f"{CARD_NAMES[state.cards[player]]}{state.history}"
