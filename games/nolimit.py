"""
Heads-up no-limit Hold'em as a traversable, abstracted game.

This is where CFR meets the project's actual poker. Three things are borrowed
and one is rebuilt:

* **Borrowed:** the engine's card representation, its fast 7-card evaluator for
  showdowns, and the six-action abstraction the rest of the project uses.
* **Rebuilt:** the betting. ``engine.PokerGame`` mutates state in place and
  consumes a dealt deck, so a traverser cannot branch over an action and back
  up. The pot arithmetic here is small, immutable, and carried in the state.

**Abstraction applies to information sets, not to the game.** Cards are dealt
for real and showdowns are settled on real hands; only the key a player uses to
look up their strategy is bucketed. Abstracting the game itself would require
estimating bucket-to-bucket transition probabilities and would throw away the
correlation between the two players' holdings. This way the game stays exact and
only the solver's view is coarse — which is also what the exploitability of the
resulting strategy is measured against.

Chance here cannot be enumerated: the opening deal alone has about 1.6 million
outcomes. :meth:`chance_outcomes` therefore raises, and :meth:`sample_chance`
deals directly. Vanilla CFR cannot run on this game, which is a true statement
about no-limit Hold'em rather than a gap in the implementation.
"""
from __future__ import annotations

from typing import Hashable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from abstraction.betting import (ALL_IN, CHECK_CALL, FOLD, RAISE_HALF,
                                 RAISE_POT, RAISE_TWO, legal_actions)
from abstraction.buckets import CardAbstraction, flop_table_arrays
from abstraction.canonical import hash_value, nearest_centroid
from abstraction.equity import FULL_DECK
from engine.hand_eval_fast import evaluate_hand_fast

from .base import CHANCE, Game

#: Board cards revealed at the start of each street.
#: Entries the hole-and-board bucket memo keeps before clearing.
#:
#: The memo is keyed on (hole, board), a space of roughly 1,326 x 2.1 million,
#: so it never saturates: measured on 10 September it reached 2,800,227 entries
#: by 60,000 iterations and accounted for 491 MB of the solver process. That is
#: what put a 1280-second ceiling on `scripts/cfr/bucket_sweep.py`, since three
#: arms train at once and the machine has 8 GB.
#:
#: Clearing is safe rather than merely cheap. `bucket_for` seeds its generator
#: from `hash(key)`, so it is a pure function of hole and board and the memo
#: only ever saves recomputation. `test_nolimit.py` pins that: the same holding
#: buckets identically across a clear.
#:
#: 500,000 bounds it near 90 MB. Raise it if a run has memory to spare and is
#: missing often; the cost of a clear is recomputation, never a wrong answer.
BUCKET_CACHE_LIMIT = 500_000

STREET_BOARD_SIZE = (0, 3, 4, 5)
STREET_NAMES = ("preflop", "flop", "turn", "river")

#: Pot fraction each raise action adds on top of calling.
RAISE_FRACTION = {RAISE_HALF: 0.5, RAISE_POT: 1.0, RAISE_TWO: 2.0}


class NoLimitState(NamedTuple):
    """
    Immutable state.

    ``contributions`` is chips put in this hand, ``committed`` chips put in on
    the current street — the difference between them is what a call must match.
    Both are carried rather than re-derived because a raise is sized from the
    pot at the moment it is made, so replaying the history to recover them would
    mean re-deriving the pot at every step.
    """
    hole: Tuple[Tuple[int, ...], ...]
    board: Tuple[int, ...]
    history: str
    contributions: Tuple[int, int]
    committed: Tuple[int, int]
    stacks: Tuple[int, int]
    street: int


def _street_actions(history: str) -> str:
    """Actions taken on the current street."""
    return history.split("/")[-1]


def _flop_table_for(abstraction):
    """The (keys, equity, flop centroids) a compiled lookup needs, or None."""
    return flop_table_arrays(abstraction)


class NoLimitHoldem(Game):
    """
    Heads-up no-limit Hold'em over a card abstraction.

    Args:
        abstraction: Fitted :class:`~abstraction.buckets.CardAbstraction`
            supplying the bucket used in information-set keys.
        starting_stack: Chips each player begins with.
        small_blind / big_blind: Blind sizes.
        raise_cap: Bets or raises allowed per street. This is the parameter that
            decides whether the game fits in memory — see
            ``scripts/cfr/measure_abstraction.py``.
        equity_samples: Monte Carlo samples per postflop bucket lookup.
        bucket_cache_limit: Entries kept in the hole-and-board bucket memo
            before it is cleared. See :data:`BUCKET_CACHE_LIMIT`.
    """

    num_players = 2

    def __init__(self, abstraction: CardAbstraction, starting_stack: int = 200,
                 small_blind: int = 1, big_blind: int = 2, raise_cap: int = 1,
                 equity_samples: int = 60,
                 bucket_cache_limit: int = BUCKET_CACHE_LIMIT):
        self.abstraction = abstraction
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.raise_cap = raise_cap
        self.equity_samples = equity_samples
        self.bucket_cache_limit = bucket_cache_limit
        self._bucket_cache: dict = {}
        # A precomputed flop table, if the abstraction carries one. Held here
        # rather than reached through the abstraction on every lookup, with
        # reusable buffers, because the cost being removed is per-call overhead
        # and reintroducing it in the fast path would defeat the point.
        self._flop_table = _flop_table_for(abstraction)
        self._hole_buf = np.empty(2, dtype=np.int64)
        self._board_buf = np.empty(3, dtype=np.int64)

    # ---- structure -------------------------------------------------------

    def initial_state(self) -> NoLimitState:
        """Blinds are posted before the deal, so the root is a chance node."""
        return NoLimitState(
            hole=(), board=(), history="",
            contributions=(self.small_blind, self.big_blind),
            committed=(self.small_blind, self.big_blind),
            stacks=(self.starting_stack - self.small_blind,
                    self.starting_stack - self.big_blind),
            street=0,
        )

    def is_terminal(self, state: NoLimitState) -> bool:
        if not state.hole:
            return False
        if str(FOLD) in state.history:
            return True
        if state.street >= len(STREET_NAMES):
            return True
        # All-in with the board complete: nobody can act and there is nothing
        # left to deal, so the hand is over. Without this the state is neither
        # terminal nor actionable and the traverser asks chance for a street
        # that does not exist.
        if self._all_in(state) and len(state.board) == STREET_BOARD_SIZE[-1]:
            return True
        return False

    def _street_closed(self, state: NoLimitState) -> bool:
        """
        True when betting on the current street is complete.

        Both players must have acted and the last action must be a check or
        call: preflop, the small blind calling the big blind still leaves the
        big blind its option, and postflop a single check leaves the other
        player to act.
        """
        actions = _street_actions(state.history)
        if len(actions) < 2 or actions[-1] != str(CHECK_CALL):
            return False
        if state.committed[0] == state.committed[1]:
            return True
        # A call that could not cover the bet still closes the street — the
        # caller is all-in and the uncalled excess is returned at showdown.
        return min(state.stacks) == 0

    def _all_in(self, state: NoLimitState) -> bool:
        """
        True once no one can act again: someone is out of chips and the bets are
        level, so the board simply runs out to showdown.

        Without this the traverser kept producing decision nodes on every later
        street and asking a check/call from a player holding nothing. Those
        filler actions are not free: they extend `history`, and `history` is the
        information-set key, so the same hand keyed differently here than in the
        engine — which ends the hand at an all-in call. A fraction of every
        solver's table was therefore fitted for situations that cannot occur,
        and the two evaluation paths disagreed on which of two strategies was
        stronger. Measured 8 September: 19.7% of sampled decision nodes had a
        legal-action list the engine's reconstruction could not match.
        """
        return min(state.stacks) == 0 and state.committed[0] == state.committed[1]

    def current_player(self, state: NoLimitState) -> int:
        if not state.hole:
            return CHANCE
        if self._all_in(state):
            return CHANCE                      # run the board out; nobody acts
        if self._street_closed(state):
            return CHANCE                      # deal the next street
        actions = _street_actions(state.history)
        # Preflop the small blind (seat 0) acts first; afterwards the big blind.
        first = 0 if state.street == 0 else 1
        return (first + len(actions)) % 2

    def legal_actions(self, state: NoLimitState) -> Sequence[int]:
        actions = _street_actions(state.history)
        raises = sum(1 for a in actions if int(a) in RAISE_FRACTION or int(a) == ALL_IN)
        facing = state.committed[0] != state.committed[1]
        last = int(actions[-1]) if actions else None

        available = list(legal_actions(raises, facing, self.raise_cap, last, street=state.street))

        player = self.current_player(state)
        to_call = abs(state.committed[0] - state.committed[1])
        if state.stacks[player] <= to_call:
            # Cannot raise what one cannot cover; calling is capped at the stack.
            available = [a for a in available if a in (FOLD, CHECK_CALL)]
        return tuple(available)

    def next_state(self, state: NoLimitState, action) -> NoLimitState:
        if not state.hole:
            hole, board = action
            return state._replace(hole=hole, board=board)

        if self.current_player(state) == CHANCE:
            # Reveal the next street and reset per-street commitments.
            street = state.street + 1
            return state._replace(board=action, history=state.history + "/",
                                  committed=(0, 0), street=street)

        return self._apply(state, int(action))

    def _apply(self, state: NoLimitState, action: int) -> NoLimitState:
        player = self.current_player(state)
        return self._commit(state, player, self._cost(state, player, action),
                            str(action))

    def _cost(self, state: NoLimitState, player: int, action: int) -> int:
        """Chips ``action`` costs ``player`` here."""
        opponent = 1 - player
        to_call = state.committed[opponent] - state.committed[player]

        if action == FOLD:
            return 0
        if action == CHECK_CALL:
            return min(max(to_call, 0), state.stacks[player])
        if action == ALL_IN:
            return state.stacks[player]
        return self._raise_cost(state, player, RAISE_FRACTION[action])

    def _raise_cost(self, state: NoLimitState, player: int, fraction: float) -> int:
        """Chips a raise of ``fraction`` times the pot costs, calling included."""
        opponent = 1 - player
        to_call = state.committed[opponent] - state.committed[player]
        pot = sum(state.contributions) + to_call
        raise_by = int(round(fraction * pot))
        return min(max(to_call, 0) + max(raise_by, self.big_blind),
                   state.stacks[player])

    def _commit(self, state: NoLimitState, player: int, pay: int,
                recorded: str) -> NoLimitState:
        """
        Move ``pay`` chips and append ``recorded`` to the history.

        The two are separate arguments because they are separate things. A
        player betting a size the abstraction does not contain still moves the
        real chips, while the history — which is what an opponent's strategy is
        keyed on — can only name an action the abstraction has. See
        :meth:`raise_by_fraction`.
        """
        contributions = list(state.contributions)
        committed = list(state.committed)
        stacks = list(state.stacks)

        contributions[player] += pay
        committed[player] += pay
        stacks[player] -= pay

        updated = state._replace(
            history=state.history + recorded,
            contributions=tuple(contributions),
            committed=tuple(committed),
            stacks=tuple(stacks),
        )

        # If the street just closed on the river, the hand is over.
        if updated.street == len(STREET_NAMES) - 1 and self._street_closed(updated):
            return updated._replace(street=len(STREET_NAMES))
        return updated

    def raise_by_fraction(self, state: NoLimitState, fraction: float,
                          perceived: int) -> NoLimitState:
        """
        Raise ``fraction`` times the pot, recorded in the history as ``perceived``.

        This is how a player bets outside the abstraction. The pot grows by what
        was actually wagered, so every payoff is exact; the history names one of
        the abstract raises, so an opponent keyed on ``bucket|history`` still has
        an entry to answer with. Which abstract raise is a translation question,
        and not one this method decides — see :mod:`abstraction.translation`.

        Args:
            state: The state to act in.
            fraction: Real raise size, as a multiple of the pot.
            perceived: The abstract raise action the opponent is to believe it
                faced. Must be one the abstraction contains.
        """
        if perceived not in RAISE_FRACTION and perceived != ALL_IN:
            raise ValueError(
                f"perceived must be an abstract raise, got {perceived!r}")
        player = self.current_player(state)
        return self._commit(state, player,
                            self._raise_cost(state, player, fraction),
                            str(perceived))

    # ---- chance ----------------------------------------------------------

    def chance_outcomes(self, state: NoLimitState):
        raise NotImplementedError(
            "no-limit chance cannot be enumerated (about 1.6 million opening "
            "deals); use sample_chance, and a sampling solver such as MCCFRSolver"
        )

    def sample_chance(self, state: NoLimitState, rng) -> object:
        """Deal the hole cards, or reveal the next street."""
        if not state.hole:
            picked = rng.choice(len(FULL_DECK), size=4, replace=False)
            return ((tuple(picked[:2]), tuple(picked[2:])), ())

        needed = STREET_BOARD_SIZE[state.street + 1] - len(state.board)
        seen = set(state.board) | set(state.hole[0]) | set(state.hole[1])
        available = np.array([i for i in range(len(FULL_DECK)) if i not in seen])
        drawn = rng.choice(available, size=needed, replace=False)
        return state.board + tuple(int(c) for c in drawn)

    # ---- payoffs and knowledge -------------------------------------------

    def utility(self, state: NoLimitState, player: int) -> float:
        opponent = 1 - player

        if str(FOLD) in state.history:
            folded = self._who_folded(state)
            if folded == player:
                return -float(state.contributions[player])
            return float(state.contributions[opponent])

        mine = evaluate_hand_fast([FULL_DECK[c] for c in
                                   tuple(state.hole[player]) + state.board])
        theirs = evaluate_hand_fast([FULL_DECK[c] for c in
                                     tuple(state.hole[opponent]) + state.board])

        # Only the matched portion is at risk; anything a player put in beyond
        # what the opponent could cover is returned, so it nets to nothing.
        at_risk = min(state.contributions)
        if mine > theirs:
            return float(at_risk)
        if mine < theirs:
            return -float(at_risk)
        return 0.0

    def _who_folded(self, state: NoLimitState) -> int:
        """Index of the player whose action was the fold."""
        position = state.history.index(str(FOLD))
        prefix = state.history[:position]
        street = prefix.count("/")
        actions_before = len(prefix.split("/")[-1])
        first = 0 if street == 0 else 1
        return (first + actions_before) % 2

    def information_set(self, state: NoLimitState, player: int) -> Hashable:
        """
        Bucketed hand strength plus the public betting history.

        The bucket is the entire card abstraction: two situations sharing a
        bucket and a history are indistinguishable to the solver. The
        opponent's cards never appear.
        """
        return f"{self._bucket(state, player)}|{state.history}"

    def information_set_with(self, state: NoLimitState, player: int,
                             abstraction: CardAbstraction) -> Hashable:
        """
        The key this state would have under a DIFFERENT card abstraction.

        Needed to play two agents trained on different abstractions against each
        other: each must be asked the question its own bucketing poses. Handing
        an agent the other's key would look like bad play rather than a
        mismatched lookup.
        """
        hole = [FULL_DECK[c] for c in state.hole[player]]
        board = [FULL_DECK[c] for c in state.board]
        seed = hash((state.hole[player], state.board)) % (2 ** 32)
        bucket = abstraction.bucket(hole, board, np.random.default_rng(seed))
        return f"{bucket}|{state.history}"

    def bucket_for(self, hole: Tuple[int, ...], board: Tuple[int, ...]) -> int:
        """
        The bucket a holding falls into on a board, as this game keys it.

        Public because anything reasoning about what a *hypothetical* holding
        would have done — an exploiter tracking the opponent's range, say — must
        ask the question exactly as the solver was asked it. Computing the bucket
        separately would risk a different rng draw and therefore a different
        answer for the same situation, which would silently break the lookup.
        """
        key = (tuple(int(c) for c in hole), tuple(int(c) for c in board))
        cached = self._bucket_cache.get(key)
        if cached is None and self._flop_table is not None and len(board) == 3:
            keys, values, centroids = self._flop_table
            self._hole_buf[0] = hole[0]
            self._hole_buf[1] = hole[1]
            self._board_buf[0] = board[0]
            self._board_buf[1] = board[1]
            self._board_buf[2] = board[2]
            value = hash_value(keys, values, self._hole_buf, self._board_buf)
            if value >= 0.0:
                cached = int(nearest_centroid(centroids, value))
                if len(self._bucket_cache) >= self.bucket_cache_limit:
                    self._bucket_cache.clear()
                self._bucket_cache[key] = cached
                return cached
        if cached is None:
            cards = [FULL_DECK[c] for c in key[0]]
            community = [FULL_DECK[c] for c in key[1]]
            cached = self.abstraction.bucket(cards, community,
                                             np.random.default_rng(hash(key) % (2 ** 32)))
            if len(self._bucket_cache) >= self.bucket_cache_limit:
                # Cleared wholesale rather than evicted least-recently-used: an
                # LRU costs bookkeeping on every hit, and this is on the hot
                # path of every traversal. The memo is pure, so the only cost of
                # dropping it is recomputing what it held.
                self._bucket_cache.clear()
            self._bucket_cache[key] = cached
        return cached

    def _bucket(self, state: NoLimitState, player: int) -> int:
        return self.bucket_for(state.hole[player], state.board)
