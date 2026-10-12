"""
Leduc Hold'em — the standard small benchmark for imperfect-information solvers.

Six cards (two suits of J, Q, K), two players, two betting rounds with one
community card between them. Large enough to have a board, pairs and raises;
small enough that exploitability can still be computed exactly. Kuhn proves a
solver correct; Leduc is where it is actually exercised.

Rules as implemented (the standard formulation, Southey et al. 2005):

* Both players ante 1.
* Each is dealt one private card from the six.
* Round one: betting, bet size 2, at most two bets or raises.
* One community card is revealed.
* Round two: betting, bet size 4, at most two bets or raises.
* Showdown: pairing the community card wins. Otherwise the higher private card
  wins; equal ranks split.

Two implementation choices worth stating:

* **Pot contributions are derived from the betting history, never stored.**
  Carrying a running pot alongside the history invites the two to drift, and a
  desynchronised pot is exactly the class of bug that produced silently wrong
  results elsewhere in this project. Here the history is the single source of
  truth and the money is a pure function of it.

* **Information sets are keyed on rank, not on the specific card.** Suits carry
  no information in Leduc — hand strength depends only on ranks — so holding the
  Q of one suit is strategically identical to holding the other. Collapsing them
  is a lossless abstraction, and it is what the information-set counts quoted in
  the literature refer to.
"""
from __future__ import annotations

from typing import Hashable, List, NamedTuple, Sequence, Tuple

from .base import CHANCE, Game

#: Actions. 'c' checks when facing no bet and calls when facing one; 'r' bets or
#: raises; 'f' folds. Folding with nothing to call is legal in real poker but
#: strictly dominated, so it is omitted from the tree.
CALL = "c"
RAISE = "r"
FOLD = "f"

#: Rank names, weakest to strongest. Two suited copies of each make the 6-card deck.
RANK_NAMES = ("J", "Q", "K")
DECK_SIZE = 6

#: Bet size per round, and the cap on bets/raises within a round.
BET_SIZES = (2, 4)
RAISE_CAP = 2

ANTE = 1


class LeducState(NamedTuple):
    """
    Immutable state.

    ``hole`` is empty before the deal. ``board`` is -1 until the community card
    is revealed. ``history`` records actions with ``/`` separating the rounds, so
    it encodes both what happened and which round is live.
    """
    hole: Tuple[int, ...]
    board: int
    history: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        hole = "".join(RANK_NAMES[c // 2] for c in self.hole) or "-"
        board = RANK_NAMES[self.board // 2] if self.board >= 0 else "-"
        return f"LeducState(hole={hole}, board={board}, history={self.history or 'root'!r})"


def rank_of(card: int) -> int:
    """Rank index of a card; the two suits of a rank share it."""
    return card // 2


def _round_status(sequence: str) -> Tuple[bool, int, int, int]:
    """
    Interpret one round's actions.

    Returns ``(closed, level, actor, folder)`` where ``level`` is the number of
    bets/raises made, ``actor`` is whose turn it is next, and ``folder`` is the
    player who folded or -1.

    A round closes when a bet is called or when both players check.
    """
    level = 0
    actor = 0
    for index, action in enumerate(sequence):
        if action == FOLD:
            return True, level, 1 - actor, actor
        if action == CALL:
            if level > 0:                                   # called a bet
                return True, level, 1 - actor, -1
            if index > 0 and sequence[index - 1] == CALL:    # checked behind
                return True, level, 1 - actor, -1
        elif action == RAISE:
            level += 1
        actor = 1 - actor
    return False, level, actor, -1


def _contributions(history: str) -> Tuple[List[int], int]:
    """
    Each player's total contribution, and who folded (-1 if nobody).

    In limit poker a call matches the current level and a raise moves to the
    next, so a player's contribution within a round is simply ``level * bet``
    for the level they matched.
    """
    contributed = [ANTE, ANTE]
    folder = -1

    for round_index, sequence in enumerate(history.split("/")):
        bet = BET_SIZES[min(round_index, len(BET_SIZES) - 1)]
        level = 0
        actor = 0
        in_round = [0, 0]
        for action in sequence:
            if action == FOLD:
                folder = actor
                break
            if action == CALL:
                in_round[actor] = level * bet
            elif action == RAISE:
                level += 1
                in_round[actor] = level * bet
            actor = 1 - actor
        contributed[0] += in_round[0]
        contributed[1] += in_round[1]
        if folder != -1:
            break

    return contributed, folder


class LeducHoldem(Game):
    """Leduc Hold'em as a :class:`~games.base.Game`."""

    num_players = 2

    def initial_state(self) -> LeducState:
        return LeducState(hole=(), board=-1, history="")

    # ---- structure -------------------------------------------------------

    def is_terminal(self, state: LeducState) -> bool:
        if not state.hole:
            return False
        rounds = state.history.split("/")
        if any(FOLD in sequence for sequence in rounds):
            return True
        if len(rounds) == 1:
            return False                     # round one live, or board pending
        return _round_status(rounds[1])[0]

    def current_player(self, state: LeducState) -> int:
        if not state.hole:
            return CHANCE                    # deal the hole cards
        rounds = state.history.split("/")
        if len(rounds) == 1:
            closed, _, actor, _ = _round_status(rounds[0])
            return CHANCE if closed else actor   # closed -> reveal the board
        return _round_status(rounds[1])[2]

    def legal_actions(self, state: LeducState) -> Sequence[str]:
        rounds = state.history.split("/")
        _, level, _, _ = _round_status(rounds[-1])
        if level == 0:
            return (CALL, RAISE)             # check or open; folding is dominated
        if level >= RAISE_CAP:
            return (FOLD, CALL)              # capped: no further raises
        return (FOLD, CALL, RAISE)

    def next_state(self, state: LeducState, action) -> LeducState:
        if not state.hole:                                   # dealing hole cards
            return LeducState(hole=tuple(action), board=-1, history="")
        if self.current_player(state) == CHANCE:              # revealing the board
            return LeducState(hole=state.hole, board=action,
                              history=state.history + "/")
        return LeducState(hole=state.hole, board=state.board,
                          history=state.history + action)

    # ---- chance ----------------------------------------------------------

    def chance_outcomes(self, state: LeducState) -> Sequence[Tuple[object, float]]:
        if not state.hole:
            deals = [((a, b), 0.0)
                     for a in range(DECK_SIZE) for b in range(DECK_SIZE) if a != b]
            probability = 1.0 / len(deals)
            return [(cards, probability) for cards, _ in deals]

        remaining = [c for c in range(DECK_SIZE) if c not in state.hole]
        probability = 1.0 / len(remaining)
        return [(card, probability) for card in remaining]

    # ---- payoffs and knowledge -------------------------------------------

    def _showdown_winner(self, state: LeducState) -> int:
        """Index of the winning player at showdown, or -1 for a split."""
        board_rank = rank_of(state.board)
        ranks = [rank_of(card) for card in state.hole]

        # Only one player can pair the board: a rank has two copies and the
        # board holds one of them.
        for player, rank in enumerate(ranks):
            if rank == board_rank:
                return player

        if ranks[0] == ranks[1]:
            return -1
        return 0 if ranks[0] > ranks[1] else 1

    def utility(self, state: LeducState, player: int) -> float:
        contributed, folder = _contributions(state.history)
        opponent = 1 - player

        if folder != -1:
            winner = 1 - folder
        else:
            winner = self._showdown_winner(state)
            if winner == -1:
                return 0.0                   # split; contributions are equal here

        if winner == player:
            return float(contributed[opponent])
        return -float(contributed[player])

    def information_set(self, state: LeducState, player: int) -> Hashable:
        """
        The player's own rank, the board rank once revealed, and the public
        betting history. Suits are excluded because they carry no information.
        """
        own = RANK_NAMES[rank_of(state.hole[player])]
        board = RANK_NAMES[rank_of(state.board)] if state.board >= 0 else ""
        return f"{own}{board}:{state.history}"
