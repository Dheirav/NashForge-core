"""
The interface a game must expose to be solved by regret minimization.

CFR does not *play* a game, it **traverses** one: at every decision it recurses
into all of the acting player's actions, and at every deal it recurses into all
outcomes weighted by probability. A simulator cannot support that. The existing
``engine.PokerGame`` mutates state in place, consumes a dealt deck, and has no
notion of "what this player knows" — so a traverser cannot branch over actions,
back up, and try the other one.

This module defines what tree search needs instead. Kuhn poker, Leduc Hold'em
and eventually an adapter over the no-limit engine all implement it, so the
solver, the best-response calculator and the exploitability metric are written
once against this and never against a specific game.

Two rules matter more than the rest:

* ``next_state`` must return a NEW state and never mutate its argument. A
  traverser holds a parent state while exploring each child; mutating in place
  corrupts the traversal in ways that surface as subtly wrong strategies rather
  than as crashes.

* ``information_set`` must contain exactly what the acting player knows — their
  own private cards and the public history — and nothing else. Leaking an
  opponent's card into that key lets the solver condition on hidden information.
  It will converge happily to a strategy that cheats, and the exploitability
  number will look excellent, because the best response is computed against the
  same leaky abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable, Sequence, Tuple

#: Sentinel returned by :meth:`Game.current_player` at a chance node — a deal,
#: not a decision. Negative so it can never collide with a player index.
CHANCE = -1


class Game(ABC):
    """A finite extensive-form game with imperfect information."""

    #: Number of players making decisions (chance is not a player).
    num_players: int = 2

    # ---- structure -------------------------------------------------------

    @abstractmethod
    def initial_state(self) -> Any:
        """The root of the game tree, before any cards are dealt."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """True when no further actions are possible and payoffs are defined."""

    @abstractmethod
    def current_player(self, state: Any) -> int:
        """
        Index of the player to act, or :data:`CHANCE` at a deal.

        Only called on non-terminal states.
        """

    @abstractmethod
    def legal_actions(self, state: Any) -> Sequence[Any]:
        """
        Actions available to :meth:`current_player`.

        Must be deterministic and consistently ordered for a given state: regret
        and strategy vectors are indexed positionally against it.
        """

    @abstractmethod
    def next_state(self, state: Any, action: Any) -> Any:
        """
        The state reached by taking ``action``.

        Must not mutate ``state``. See the module docstring.
        """

    # ---- chance ----------------------------------------------------------

    @abstractmethod
    def chance_outcomes(self, state: Any) -> Sequence[Tuple[Any, float]]:
        """
        ``(action, probability)`` pairs at a chance node, summing to 1.

        Only called when :meth:`current_player` returns :data:`CHANCE`.
        Enumerating outcomes rather than sampling one is what lets vanilla CFR
        weight subtrees exactly; samplers may still choose to sample from this.

        A game whose chance is too large to enumerate — no-limit Hold'em deals
        one of about 1.6 million hole-card pairs — should raise here and
        override :meth:`sample_chance` instead. Vanilla CFR then cannot run on
        it, which is the truth rather than a limitation to work around.
        """

    def sample_chance(self, state: Any, rng) -> Any:
        """
        Draw one chance outcome.

        Defaults to sampling from :meth:`chance_outcomes`, which is right for
        small games. Games with unenumerable chance override this to deal
        directly; a sampler never needs the full list, so requiring it would
        rule out exactly the games sampling exists for.
        """
        outcomes = self.chance_outcomes(state)
        index = rng.choice(len(outcomes), p=[p for _, p in outcomes])
        return outcomes[int(index)][0]

    # ---- payoffs and knowledge -------------------------------------------

    @abstractmethod
    def utility(self, state: Any, player: int) -> float:
        """
        Payoff to ``player`` at a terminal state, in chips.

        Only called when :meth:`is_terminal` is true.
        """

    @abstractmethod
    def information_set(self, state: Any, player: int) -> Hashable:
        """
        A key identifying everything ``player`` knows at ``state``.

        Two states a player cannot tell apart must produce equal keys, and two
        states they can tell apart must not. This is the single most
        consequential method to get right — see the module docstring.
        """

    # ---- provided --------------------------------------------------------

    def is_chance(self, state: Any) -> bool:
        """True at a deal rather than a decision."""
        return self.current_player(state) == CHANCE
