"""
External-sampling Monte Carlo CFR (Lanctot et al., 2009).

Vanilla CFR walks the entire tree every iteration, which is exact and becomes
impossible the moment a game is interesting. External sampling keeps the
traverser's own decisions exhaustive but samples everything external to them —
chance, and the opponent's actions. Per-iteration cost becomes proportional to a
sampled path rather than to the whole tree, and the regret estimates stay
unbiased, so convergence guarantees survive.

Two details that are easy to get subtly wrong, and that produce a solver which
converges to something plausible rather than failing loudly:

* **Only the traverser's actions are enumerated.** Sampling the traverser's
  action too gives outcome sampling, a different (higher-variance) estimator
  with a different correction term. Mixing the two silently biases the regrets.

* **The average strategy is accumulated at the opponent's nodes**, where the
  reach probability the estimator needs is exactly the sampling probability that
  brought the walk there. Accumulating it at the traverser's nodes instead —
  which reads naturally, since that is where the strategy is being improved —
  weights the average by the wrong distribution.

One "iteration" runs the traversal once per player, so both players' regrets
advance together.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Sequence

import numpy as np

from games.base import Game

from .updates import VANILLA, UpdateRule
from .vanilla import InfoSetNode


class MCCFRSolver:
    """External-sampling MCCFR with a configurable regret update rule."""

    def __init__(self, game: Game, rule: UpdateRule = VANILLA, seed: int | None = None,
                 average_from: int = 0):
        #: Accumulate the average only from this iteration on; see the native
        #: solver's `set_average_from` for why. Zero is the classical average.
        self.average_from = average_from
        self.game = game
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self.nodes: Dict[Hashable, InfoSetNode] = {}
        self.iterations = 0

    # ------------------------------------------------------------------

    def _node(self, key: Hashable, num_actions: int) -> InfoSetNode:
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(num_actions)
            self.nodes[key] = node
        return node

    def _sample(self, probabilities: Sequence[float]) -> int:
        """Draw an index from a distribution."""
        return int(self.rng.choice(len(probabilities), p=np.asarray(probabilities)))

    def _discount_once(self, node: InfoSetNode, iteration: int) -> None:
        """
        Apply the update rule's decay at most once per node per iteration.

        Discount schedules are defined per iteration of the algorithm, but a
        sampler reaches the same information set repeatedly within a single
        iteration: the traverser enumerates every action, so whole subtrees are
        revisited, and each iteration traverses once per player. Decaying on
        every visit compounds the schedule an unpredictable number of times —
        it silently changes the algorithm being measured rather than raising
        anything, and it penalises exactly the rules that decay most.
        """
        if node.last_discounted != iteration:
            self.rule.discount(node, iteration)
            node.last_discounted = iteration

    def _walk(self, state: Any, traverser: int) -> float:
        """
        Sampled counterfactual value of ``state`` to ``traverser``.

        Recurses into every action at the traverser's nodes and into a single
        sampled action everywhere else.
        """
        game = self.game

        if game.is_terminal(state):
            return game.utility(state, traverser)

        if game.is_chance(state):
            # Through sample_chance rather than chance_outcomes: a no-limit deal
            # has ~1.6 million outcomes, and building that list to pick one from
            # it would cost more than the rest of the traversal put together.
            return self._walk(
                game.next_state(state, game.sample_chance(state, self.rng)), traverser)

        player = game.current_player(state)
        actions: Sequence[Any] = game.legal_actions(state)
        node = self._node(game.information_set(state, player), len(actions))
        strategy = node.strategy()

        if player != traverser:
            # Opponent: sample one action, and accumulate their average
            # strategy here, where the reach weighting is correct.
            self._discount_once(node, self.iterations + 1)
            if self.iterations >= self.average_from:
                node.strategy_sum += self.rule.strategy_weight(self.iterations + 1) * strategy
            index = self._sample(strategy)
            return self._walk(game.next_state(state, actions[index]), traverser)

        # Traverser: enumerate every action.
        action_values = np.empty(len(actions), dtype=np.float64)
        for index, action in enumerate(actions):
            action_values[index] = self._walk(game.next_state(state, action), traverser)
        value = float(np.dot(strategy, action_values))

        self._discount_once(node, self.iterations + 1)
        self.rule.add_regret(node, action_values - value)
        return value

    # ------------------------------------------------------------------

    def train(self, iterations: int, on_progress=None,
              progress_every: int = 0) -> None:
        """
        Run ``iterations`` passes, each traversing once per player.

        ``on_progress(done, total, solver)`` is called every ``progress_every``
        iterations when both are given. Optional and off by default, so every
        existing caller is unaffected.

        It exists because this loop is silent, and a silent loop is one nobody
        can size. A 500,000-iteration run was killed at six hours having printed
        nothing and written nothing: there was no way to tell whether it was
        halfway or nearly done, and no partial result to keep. A callback lets
        the caller report progress and checkpoint without this module needing to
        know what either means.
        """
        for done in range(1, iterations + 1):
            for traverser in range(self.game.num_players):
                self._walk(self.game.initial_state(), traverser)
            self.iterations += 1
            if on_progress and progress_every and done % progress_every == 0:
                on_progress(done, iterations, self)

    def average_strategy(self) -> Dict[Hashable, np.ndarray]:
        """The converged strategy: information set key -> action probabilities."""
        return {key: node.average_strategy() for key, node in self.nodes.items()}
