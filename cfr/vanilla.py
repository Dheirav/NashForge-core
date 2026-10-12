"""
Vanilla Counterfactual Regret Minimization.

Implements Zinkevich et al. (2007): at every information set, accumulate
counterfactual regret for each action, pick the next strategy by regret
matching, and return the *average* strategy over all iterations. In a
two-player zero-sum game that average converges to a Nash equilibrium.

The two things people get wrong here, both of which produce a solver that runs
happily and converges to the wrong thing:

* **It is the average strategy that converges, not the current one.** The
  regret-matching strategy keeps oscillating forever. Reading the final
  iteration's strategy instead of the average is the classic error.

* **Regret is weighted by the *counterfactual* reach — the probability of
  arriving at the information set contributed by everyone except the player
  themselves** (opponents and chance). Weighting by the player's own reach
  instead makes the update self-referential and breaks the convergence proof.

This traverses the full tree every iteration, which is exact but exponential in
depth. It is the reference implementation: correct, slow, and used to validate
the sampling variants that replace it on larger games.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Sequence

import numpy as np

from games.base import Game


class InfoSetNode:
    """Accumulated regret and strategy for one information set."""

    __slots__ = ("num_actions", "regret_sum", "strategy_sum", "last_discounted")

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)
        #: Iteration in which this node's accumulators were last decayed.
        #: Discount schedules are defined per iteration, but a sampler may
        #: reach the same information set many times within one — see
        #: :meth:`cfr.mccfr.MCCFRSolver._discount_once`.
        self.last_discounted = 0

    def strategy(self) -> np.ndarray:
        """
        Current strategy by regret matching: play each action in proportion to
        its positive cumulative regret, uniformly if none is positive.
        """
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        if total > 0.0:
            return positive / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def average_strategy(self) -> np.ndarray:
        """
        The strategy that actually converges — the reach-weighted average of
        every strategy played, not the latest one.
        """
        total = self.strategy_sum.sum()
        if total > 0.0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)


class CFRSolver:
    """Vanilla CFR over any :class:`~games.base.Game`."""

    def __init__(self, game: Game):
        self.game = game
        self.nodes: Dict[Hashable, InfoSetNode] = {}
        self.iterations = 0

    # ------------------------------------------------------------------

    def _node(self, key: Hashable, num_actions: int) -> InfoSetNode:
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(num_actions)
            self.nodes[key] = node
        return node

    def _walk(self, state: Any, reach: np.ndarray, chance_reach: float) -> np.ndarray:
        """
        Recurse from ``state``, returning the utility vector over players.

        ``reach[i]`` is the probability player i's own strategy contributes to
        arriving here; ``chance_reach`` is what the deal contributes. Returning
        a vector rather than a scalar keeps this correct for games that do not
        strictly alternate, and avoids the sign errors that the compact
        two-player formulation invites.
        """
        game = self.game

        if game.is_terminal(state):
            return np.array(
                [game.utility(state, p) for p in range(game.num_players)],
                dtype=np.float64,
            )

        if game.is_chance(state):
            value = np.zeros(game.num_players, dtype=np.float64)
            for action, probability in game.chance_outcomes(state):
                value += probability * self._walk(
                    game.next_state(state, action), reach, chance_reach * probability
                )
            return value

        player = game.current_player(state)
        actions: Sequence[Any] = game.legal_actions(state)
        node = self._node(game.information_set(state, player), len(actions))
        strategy = node.strategy()

        value = np.zeros(game.num_players, dtype=np.float64)
        action_values: List[np.ndarray] = []
        for index, action in enumerate(actions):
            child_reach = reach.copy()
            child_reach[player] *= strategy[index]
            child_value = self._walk(
                game.next_state(state, action), child_reach, chance_reach
            )
            action_values.append(child_value)
            value += strategy[index] * child_value

        # Counterfactual reach: everyone EXCEPT this player, times chance.
        counterfactual = chance_reach
        for other in range(game.num_players):
            if other != player:
                counterfactual *= reach[other]

        for index in range(len(actions)):
            node.regret_sum[index] += counterfactual * (
                action_values[index][player] - value[player]
            )
        node.strategy_sum += reach[player] * strategy

        return value

    # ------------------------------------------------------------------

    def train(self, iterations: int) -> float:
        """
        Run ``iterations`` CFR passes.

        Returns the expected value to player 0 under the current strategy on the
        final pass. For the converged figure use
        :func:`cfr.evaluate.expected_value` with :meth:`average_strategy`.
        """
        game = self.game
        value = np.zeros(game.num_players, dtype=np.float64)
        for _ in range(iterations):
            value = self._walk(
                game.initial_state(),
                np.ones(game.num_players, dtype=np.float64),
                1.0,
            )
            self.iterations += 1
        return float(value[0])

    def average_strategy(self) -> Dict[Hashable, np.ndarray]:
        """The converged strategy: information set key -> action probabilities."""
        return {key: node.average_strategy() for key, node in self.nodes.items()}
