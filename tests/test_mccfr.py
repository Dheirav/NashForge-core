"""
Milestone 3: external-sampling MCCFR and the regret update rules.

Two things are checked here, and they are deliberately different in kind.

**Correctness** is asserted: every update rule, under sampling, must still reach
Kuhn's known value of -1/18 and must still reduce exploitability on Leduc. A
sampler that is subtly biased — enumerating the wrong player's actions, or
accumulating the average strategy at the wrong nodes — converges to something
plausible rather than failing loudly, so the analytic answer is the check.

**Which rule is fastest is not asserted.** That is the experiment
``scripts/cfr/compare_update_rules.py`` exists to answer, across several seeds.
Encoding its expected outcome as a test would be assuming the result and then
reporting the assumption as evidence.

    python -m pytest tests/test_mccfr.py -q
"""
import numpy as np
import pytest

from cfr import (ALL_RULES, CFR_PLUS, DISCOUNTED, LINEAR, MCCFRSolver, VANILLA,
                 exploitability, expected_value)
from cfr.vanilla import InfoSetNode
from games import KuhnPoker, LeducHoldem

KUHN_VALUE = -1.0 / 18.0


# ---------------------------------------------------------------------------
# The update rules in isolation
# ---------------------------------------------------------------------------

def test_cfr_plus_floors_cumulative_regret():
    """
    Regret matching+ resets a negative cumulative regret rather than
    remembering it, so an action recovers as soon as it starts paying instead
    of first working off its history.
    """
    node = InfoSetNode(3)
    CFR_PLUS.add_regret(node, np.array([-5.0, 2.0, 0.0]))
    assert node.regret_sum.tolist() == [0.0, 2.0, 0.0]

    CFR_PLUS.add_regret(node, np.array([1.0, -3.0, 0.0]))
    assert node.regret_sum.tolist() == [1.0, 0.0, 0.0]


def test_vanilla_remembers_negative_regret():
    node = InfoSetNode(3)
    VANILLA.add_regret(node, np.array([-5.0, 2.0, 0.0]))
    assert node.regret_sum.tolist() == [-5.0, 2.0, 0.0]


def test_discounted_decays_positive_and_negative_differently():
    """
    With alpha=1.5 and beta=0, positive regret is barely touched at large t
    while negative regret is halved every iteration — the asymmetry is the
    point of the rule.
    """
    node = InfoSetNode(2)
    node.regret_sum[:] = [100.0, -100.0]
    node.last_discounted = 9                  # touched last iteration: one step
    DISCOUNTED.discount(node, iteration=10)

    positive_scale = 10 ** 1.5 / (10 ** 1.5 + 1)
    assert node.regret_sum[0] == pytest.approx(100.0 * positive_scale)
    assert node.regret_sum[1] == pytest.approx(-50.0)          # beta=0 -> 1/2


def test_discounted_decays_the_strategy_sum():
    node = InfoSetNode(2)
    node.strategy_sum[:] = [8.0, 4.0]
    node.last_discounted = 2
    DISCOUNTED.discount(node, iteration=3)
    expected = (3 / 4) ** 2
    assert node.strategy_sum.tolist() == pytest.approx([8.0 * expected, 4.0 * expected])


def test_a_discount_covers_every_iteration_since_the_node_was_last_touched():
    """
    Schedules are per iteration of the algorithm, but a sampled node is only
    touched when the sampler reaches it. One factor per visit left a node
    visited at 1 and at 1,000 with a factor of 1000/1001 instead of 2/1001, so
    at rarely reached nodes every rule collapsed to vanilla: the 169-class
    preflop's shove nodes stayed at their early 50/50 for three million
    iterations. The product must cover the gap.
    """
    node = InfoSetNode(2)
    node.strategy_sum[:] = [1.0, 1.0]
    node.regret_sum[:] = [1.0, -1.0]
    node.last_discounted = 1
    LINEAR.discount(node, iteration=1000)
    assert node.strategy_sum.tolist() == pytest.approx([2 / 1001, 2 / 1001])
    assert node.regret_sum.tolist() == pytest.approx([2 / 1001, -2 / 1001])
    # The long-gap approximation for a non-unit exponent stays close to the
    # exact product, and a short gap is exact.
    exact = 1.0
    for u in range(2, 1001):
        exact *= u ** 1.5 / (u ** 1.5 + 1)
    assert DISCOUNTED.cumulative(1.5, 1, 1000) == pytest.approx(exact, rel=2e-3)
    assert DISCOUNTED.cumulative(1.5, 9, 10) == pytest.approx(10 ** 1.5 / (10 ** 1.5 + 1))
    assert DISCOUNTED.cumulative(0.0, 1, 11) == pytest.approx(0.5 ** 10)


def test_vanilla_discount_is_a_no_op():
    node = InfoSetNode(2)
    node.regret_sum[:] = [3.0, -7.0]
    node.strategy_sum[:] = [1.0, 2.0]
    VANILLA.discount(node, iteration=99)
    assert node.regret_sum.tolist() == [3.0, -7.0]
    assert node.strategy_sum.tolist() == [1.0, 2.0]


def test_strategy_weighting():
    assert CFR_PLUS.strategy_weight(7) == 7.0     # linear averaging
    assert VANILLA.strategy_weight(7) == 1.0
    assert LINEAR.strategy_weight(7) == 1.0       # decays instead, via gamma


# ---------------------------------------------------------------------------
# The sampler reaches the known answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_reaches_the_known_kuhn_value(rule):
    """
    Every rule must land on -1/18 under sampling. A biased sampler converges to
    a plausible-looking number instead, which is why the game with an analytic
    answer is the one that validates it.
    """
    game = KuhnPoker()
    solver = MCCFRSolver(game, rule=rule, seed=0)
    solver.train(30_000)
    value = expected_value(game, solver.average_strategy())[0]
    assert value == pytest.approx(KUHN_VALUE, abs=0.01), rule.name


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_exploitability_falls_on_kuhn(rule):
    game = KuhnPoker()
    solver = MCCFRSolver(game, rule=rule, seed=0)

    solver.train(500)
    early = exploitability(game, solver.average_strategy())
    solver.train(30_000 - solver.iterations)
    late = exploitability(game, solver.average_strategy())

    assert late < early, f"{rule.name}: {early:.4f} -> {late:.4f}"
    assert late < 0.05, rule.name


def test_reduces_exploitability_on_leduc():
    game = LeducHoldem()
    solver = MCCFRSolver(game, rule=VANILLA, seed=0)

    solver.train(2_000)
    early = exploitability(game, solver.average_strategy())
    solver.train(30_000 - solver.iterations)
    late = exploitability(game, solver.average_strategy())

    assert late < early, f"{early:.4f} -> {late:.4f}"


def test_visits_essentially_the_whole_leduc_tree():
    """
    Sampling should still reach all 288 information sets given enough
    iterations; a systematic gap would mean part of the tree is unreachable
    under the sampling scheme.
    """
    game = LeducHoldem()
    solver = MCCFRSolver(game, rule=VANILLA, seed=0)
    solver.train(20_000)
    assert len(solver.nodes) == 288


def test_discount_is_applied_once_per_iteration():
    """
    Discount schedules are defined per iteration, but external sampling reaches
    the same information set many times within one — the traverser enumerates
    every action, and each iteration traverses once per player.

    Decaying on every visit compounds the schedule an unpredictable number of
    times. It raises nothing; it just quietly changes the algorithm, and it
    penalises whichever rules decay hardest — which is precisely the comparison
    this milestone exists to make.
    """
    game = LeducHoldem()
    solver = MCCFRSolver(game, rule=DISCOUNTED, seed=0)

    calls = {"count": 0}
    original = DISCOUNTED.discount

    def counting_discount(node, iteration):
        calls["count"] += 1
        original(node, iteration)

    object.__setattr__(DISCOUNTED, "discount", counting_discount)
    try:
        solver.train(1)
    finally:
        object.__setattr__(DISCOUNTED, "discount", original)

    # At most one decay per information set actually reached this iteration.
    assert calls["count"] <= len(solver.nodes), (
        f"{calls['count']} decays across {len(solver.nodes)} information sets"
    )


# ---------------------------------------------------------------------------
# Sampling hygiene
# ---------------------------------------------------------------------------

def test_seed_makes_runs_reproducible():
    game = KuhnPoker()
    first = MCCFRSolver(game, rule=VANILLA, seed=42)
    second = MCCFRSolver(game, rule=VANILLA, seed=42)
    first.train(500)
    second.train(500)

    left, right = first.average_strategy(), second.average_strategy()
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].tolist() == right[key].tolist(), key


def test_different_seeds_differ():
    """Otherwise the seed is not reaching the sampler and the spread reported
    by the comparison script would be meaningless."""
    game = KuhnPoker()
    first = MCCFRSolver(game, rule=VANILLA, seed=1)
    second = MCCFRSolver(game, rule=VANILLA, seed=2)
    first.train(500)
    second.train(500)

    assert any(not np.array_equal(first.average_strategy()[k],
                                  second.average_strategy()[k])
               for k in first.average_strategy())


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_strategies_are_probability_distributions(rule):
    game = LeducHoldem()
    solver = MCCFRSolver(game, rule=rule, seed=0)
    solver.train(1_000)
    for key, probabilities in solver.average_strategy().items():
        assert probabilities.sum() == pytest.approx(1.0), key
        assert (probabilities >= 0.0).all(), key
