"""
Measuring a strategy: expected value, best response, and exploitability.

A solver that converges to the right *value* has not necessarily found an
equilibrium — many non-equilibrium strategies share the same value against a
particular opponent. The measure that actually pins it down is **exploitability**:
how much a best-responding opponent wins. At a Nash equilibrium of a two-player
zero-sum game it is zero, and it decreases monotonically as CFR converges.

Computing a best response in an imperfect-information game is not "take the
argmax at each node". A player cannot see which state within an information set
they are in, so one action must be chosen for the whole set, and its value is
the sum over the set's states weighted by how likely the *opponent and chance*
were to produce each of them. Choosing per state instead computes the value of a
cheater and reports exploitability far too high.

The implementation below groups states by information set and decides deepest
sets first, so that when a set is decided every choice beneath it is already
fixed. This relies on perfect recall, which Kuhn and Leduc both have.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, List, Sequence, Tuple

import numpy as np

from games.base import Game

Strategy = Dict[Hashable, np.ndarray]


def _policy(strategy: Strategy, key: Hashable, num_actions: int) -> np.ndarray:
    """Action probabilities at ``key``, uniform if the solver never saw it."""
    probabilities = strategy.get(key)
    if probabilities is None:
        return np.full(num_actions, 1.0 / num_actions)
    return probabilities


def expected_value(game: Game, strategy: Strategy) -> np.ndarray:
    """Expected utility to each player when everyone plays ``strategy``."""

    def walk(state: Any) -> np.ndarray:
        if game.is_terminal(state):
            return np.array(
                [game.utility(state, p) for p in range(game.num_players)],
                dtype=np.float64,
            )
        if game.is_chance(state):
            total = np.zeros(game.num_players, dtype=np.float64)
            for action, probability in game.chance_outcomes(state):
                total += probability * walk(game.next_state(state, action))
            return total

        player = game.current_player(state)
        actions = game.legal_actions(state)
        sigma = _policy(strategy, game.information_set(state, player), len(actions))
        total = np.zeros(game.num_players, dtype=np.float64)
        for index, action in enumerate(actions):
            if sigma[index] > 0.0:
                total += sigma[index] * walk(game.next_state(state, action))
        return total

    return walk(game.initial_state())


def best_response_value(game: Game, strategy: Strategy, responder: int) -> float:
    """
    The most ``responder`` can win against opponents playing ``strategy``.

    Returns the expected utility to ``responder`` under their best response.
    """
    # Phase 1 — collect the responder's information sets, the states inside
    # each, and the probability the opponents and chance produced each state.
    sets: Dict[Hashable, List[Tuple[Any, float]]] = defaultdict(list)
    depth_of: Dict[Hashable, int] = {}

    def gather(state: Any, counterfactual: float, depth: int) -> None:
        if game.is_terminal(state) or counterfactual == 0.0:
            return
        if game.is_chance(state):
            for action, probability in game.chance_outcomes(state):
                gather(game.next_state(state, action),
                       counterfactual * probability, depth + 1)
            return

        player = game.current_player(state)
        actions = game.legal_actions(state)
        if player == responder:
            key = game.information_set(state, player)
            sets[key].append((state, counterfactual))
            depth_of[key] = max(depth_of.get(key, depth), depth)
            # The responder's own probabilities do not enter the weighting.
            for action in actions:
                gather(game.next_state(state, action), counterfactual, depth + 1)
        else:
            sigma = _policy(strategy, game.information_set(state, player), len(actions))
            for index, action in enumerate(actions):
                if sigma[index] > 0.0:
                    gather(game.next_state(state, action),
                           counterfactual * sigma[index], depth + 1)

    gather(game.initial_state(), 1.0, 0)

    # Phase 2 — decide deepest information sets first, so every choice below a
    # set is already fixed when the set itself is decided.
    #
    # That ordering is also what makes caching sound: when deciding a set at
    # depth d we only evaluate states deeper than d, whose own decisions are
    # already final and will never be revised. Without the cache each of the
    # game's information sets re-traverses its whole subtree once per action,
    # which is free on Kuhn's twelve sets and intractable on Leduc's 288.
    chosen: Dict[Hashable, int] = {}
    cache: Dict[Any, float] = {}

    def value(state: Any) -> float:
        cached = cache.get(state)
        if cached is not None:
            return cached

        if game.is_terminal(state):
            result = game.utility(state, responder)
        elif game.is_chance(state):
            result = sum(p * value(game.next_state(state, a))
                         for a, p in game.chance_outcomes(state))
        else:
            player = game.current_player(state)
            actions = game.legal_actions(state)
            if player == responder:
                index = chosen[game.information_set(state, player)]
                result = value(game.next_state(state, actions[index]))
            else:
                sigma = _policy(strategy, game.information_set(state, player),
                                len(actions))
                result = sum(sigma[i] * value(game.next_state(state, a))
                             for i, a in enumerate(actions) if sigma[i] > 0.0)

        cache[state] = result
        return result

    for key in sorted(sets, key=lambda k: depth_of[k], reverse=True):
        states = sets[key]
        actions = game.legal_actions(states[0][0])
        totals = np.zeros(len(actions), dtype=np.float64)
        for index, action in enumerate(actions):
            totals[index] = sum(
                weight * value(game.next_state(state, action))
                for state, weight in states
            )
        chosen[key] = int(np.argmax(totals))

    return value(game.initial_state())


def exploitability(game: Game, strategy: Strategy) -> float:
    """
    Distance from equilibrium, in chips per hand, for two-player zero-sum games.

    The average of what each player gains by best-responding while the other
    keeps playing ``strategy``. Zero exactly at a Nash equilibrium; positive
    otherwise, and never negative for a zero-sum game.
    """
    if game.num_players != 2:
        raise ValueError("exploitability is defined here for two-player zero-sum games")
    return (best_response_value(game, strategy, 0)
            + best_response_value(game, strategy, 1)) / 2.0
