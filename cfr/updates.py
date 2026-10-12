"""
Regret update rules — vanilla CFR, CFR+, and Discounted CFR.

These are usually presented as three algorithms. They are better understood as
one algorithm with three discount schedules applied to the accumulators before
each new contribution is added:

* **Vanilla** keeps everything at full weight. Iteration 1 counts as much as
  iteration 100,000, so early exploratory regret never fades.
* **CFR+** floors cumulative regret at zero, so an action that has been bad is
  forgiven the moment it stops being bad, and weights the average strategy by
  the iteration number.
* **Discounted CFR** decays positive and negative regret at different rates and
  decays the strategy sum too, which is a smoother version of the same idea.

Why this matters here rather than being a footnote: **CFR+'s guarantees are
proven for deterministic full traversal** (Tammelin 2014; Tammelin et al. 2015).
Under Monte Carlo sampling the flooring interacts badly with estimator variance
— sampling noise that pushes a regret negative is discarded outright rather than
averaged away, and the information is gone. Discounted CFR (Brown & Sandholm
2019) was introduced partly in response. Which one actually wins under external
sampling is an empirical question, and Leduc is small enough to answer it
exactly rather than by folklore.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UpdateRule:
    """
    A regret and strategy accumulation schedule.

    Attributes:
        name: Label used in reports.
        floor_regret: Clamp cumulative regret at zero after each update (CFR+).
        alpha: Discount exponent for positive regret; ``None`` for no decay.
        beta: Discount exponent for negative regret; ``None`` for no decay.
        gamma: Discount exponent for the strategy sum; ``None`` for no decay.
        linear_strategy: Weight each strategy contribution by the iteration
            number, as CFR+ does.
    """
    name: str
    floor_regret: bool = False
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    linear_strategy: bool = False

    # ------------------------------------------------------------------

    @staticmethod
    def cumulative(exponent: float, last: int, now: int) -> float:
        """
        The product of u^e / (u^e + 1) over the iterations (last, now].

        The schedules are defined per iteration of the algorithm, but a sampled
        node is only touched when the sampler reaches it. Applying one
        iteration's factor per visit left a node visited at 1 and at 1,000,000
        with a factor of 0.999999 instead of 0.000002, so at rarely reached
        nodes every rule collapsed to vanilla. That is exactly where the
        169-class preflop stayed at 50/50 facing a shove. Exact for e = 1, where
        the product telescopes; for other exponents the log of the product is
        the integral of -u^-e, within 1e-3 for e >= 1. Mirrored bit for bit in
        `native/src/mccfr.hpp`.
        """
        if now <= last:
            return 1.0
        if exponent == 1.0:
            return (last + 1) / (now + 1)
        if exponent == 0.0:
            return 0.5 ** (now - last)
        # Exact over the first 64 iterations of the gap, which is the whole gap
        # for every node a dense traversal touches; the tail, where only the
        # sampled rare nodes go, uses the integral of log(u^e / (u^e + 1)) to
        # second order, within 1e-3 of the product for e >= 1.
        product = 1.0
        head = min(now, last + 64)
        for u in range(last + 1, head + 1):
            scale = u ** exponent
            product *= scale / (scale + 1.0)
        if head < now:
            a, b = head + 0.5, now + 0.5
            first = (a ** (1.0 - exponent) - b ** (1.0 - exponent)) / (exponent - 1.0)
            second = (a ** (1.0 - 2 * exponent) - b ** (1.0 - 2 * exponent)) / (2 * exponent - 1.0)
            product *= float(np.exp(-first + second / 2.0))
        return product

    def discount(self, node, iteration: int) -> None:
        """
        Decay the accumulators in place, before this iteration's contribution,
        by every iteration since the node was last touched.

        ``iteration`` is 1-indexed, matching the t in the published schedules;
        ``node.last_discounted`` is the iteration of the previous touch, 0 for
        never.
        """
        last = getattr(node, "last_discounted", 0)
        if self.alpha is not None or self.beta is not None:
            regret = node.regret_sum
            if self.alpha is not None:
                regret[regret > 0] *= self.cumulative(self.alpha, last, iteration)
            if self.beta is not None:
                regret[regret < 0] *= self.cumulative(self.beta, last, iteration)

        if self.gamma is not None:
            node.strategy_sum *= self.cumulative(1.0, last, iteration) ** self.gamma

    def add_regret(self, node, instantaneous: np.ndarray) -> None:
        """Accumulate this iteration's counterfactual regret."""
        node.regret_sum += instantaneous
        if self.floor_regret:
            # Regret matching+: a negative cumulative regret is reset rather
            # than remembered, so an action recovers immediately once it starts
            # paying again instead of first working off its history.
            np.maximum(node.regret_sum, 0.0, out=node.regret_sum)

    def strategy_weight(self, iteration: int) -> float:
        """Weight applied to this iteration's strategy contribution."""
        return float(iteration) if self.linear_strategy else 1.0


#: Zinkevich et al. (2007). No discounting; the reference behaviour.
VANILLA = UpdateRule(name="vanilla")

#: Tammelin (2014). Guarantees are for full traversal, not for sampling.
CFR_PLUS = UpdateRule(name="cfr+", floor_regret=True, linear_strategy=True)

#: Brown & Sandholm (2019), with the parameters recommended there.
DISCOUNTED = UpdateRule(name="dcfr", alpha=1.5, beta=0.0, gamma=2.0)

#: Linear CFR — the simple special case where everything decays at rate t.
LINEAR = UpdateRule(name="linear", alpha=1.0, beta=1.0, gamma=1.0)

ALL_RULES = (VANILLA, CFR_PLUS, DISCOUNTED, LINEAR)
