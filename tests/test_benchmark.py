"""
The shared benchmark panel.

This harness decides what every trained agent's number means, so its own bugs
are worth more than the agents'. The one found while writing it is the type
specimen: `is_hand_over()` means the betting finished, not that the pot was
paid, so reading stacks at that point scored every hand as a loss of whatever
had been contributed. It looked completely stable — every deal returned exactly
−2 — and was completely wrong. Stability is not correctness, so the first test
here checks the chips, not the plausibility of the result.

    python -m pytest tests/test_benchmark.py -q
"""
import numpy as np
import pytest

from evaluation import always_call_agent, benchmark, random_agent
from evaluation.benchmark import _constrain, _play_hand
from abstraction.betting import CHECK_CALL, FOLD

STACK = 200
TOTAL = 2 * STACK


def deals(count=8, base=777):
    return [base * 1_000_003 + i for i in range(count)]


# ---------------------------------------------------------------------------
# The pot must actually be paid
# ---------------------------------------------------------------------------

def test_every_hand_settles_and_conserves_chips():
    """
    A hand that ends without the pot being awarded leaves chips in the middle,
    and the harness reports the contributions as losses. Both agents' results
    must sum to zero and the table must still hold 400 chips.
    """
    for seed in deals():
        first = _play_hand([always_call_agent(), always_call_agent()],
                           seed, STACK, 1, 2, 1)
        # Seat 0's gain is seat 1's loss: a settled heads-up hand is zero-sum.
        mirrored = _play_hand([always_call_agent(), always_call_agent()],
                              seed, STACK, 1, 2, 1)
        assert first == mirrored, "identical deterministic play diverged"
        assert abs(first) <= STACK, f"seat 0 moved {first}, more than a stack"


def test_the_result_is_not_a_constant():
    """
    The settlement bug produced a fixed −2 on every deal. Real poker over
    several deals does not, so a constant is the signature to watch for.
    """
    rng = np.random.default_rng(0)
    results = [_play_hand([random_agent(rng), random_agent(rng)],
                          seed, STACK, 1, 2, 1)
               for seed in deals(12)]
    assert len(set(results)) > 1, (
        f"every deal returned the same value {results[0]} — the pot is "
        f"probably not being awarded")


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def test_a_symmetric_matchup_scores_exactly_zero():
    """
    Two identical deterministic agents on duplicated deals play the same hand
    twice, so the difference is zero by construction rather than on average.
    Anything else means the seats are not being swapped cleanly.
    """
    result = benchmark(always_call_agent(), always_call_agent(),
                       "always-call", hands=60, seed=3)
    assert result.chips_per_hand == pytest.approx(0.0, abs=1e-9), result.summary()


def test_random_against_random_stays_within_noise():
    """The true value is zero; the estimate must not wander far from it."""
    rng = np.random.default_rng(1)
    result = benchmark(random_agent(rng), random_agent(rng), "random",
                       hands=400, seed=4)
    assert abs(result.chips_per_hand) < 3.0 * result.stderr, result.summary()


def test_the_result_reports_its_own_uncertainty():
    rng = np.random.default_rng(2)
    result = benchmark(random_agent(rng), always_call_agent(), "always-call",
                       hands=200, seed=5)
    assert result.hands == 200
    assert result.stderr > 0.0
    low, high = result.ci95
    assert low < result.chips_per_hand < high
    assert result.bb_per_100 == pytest.approx(result.chips_per_hand / 2 * 100)


# ---------------------------------------------------------------------------
# Keeping the engine's legal set aligned with the solver's tree
# ---------------------------------------------------------------------------

def test_folding_with_nothing_to_call_is_removed():
    """
    The engine always offers a fold; the solver's tree excludes it when there
    is nothing to call, as dominated. Leaving it in produces histories the
    solver has never seen — and throws away a free card.
    """
    mask = np.ones(6, dtype=np.float32)
    constrained = _constrain(mask, to_call=0, raises_this_street=0, raise_cap=1)
    assert constrained[FOLD] == 0.0
    assert constrained[CHECK_CALL] == 1.0

    facing = _constrain(mask, to_call=10, raises_this_street=0, raise_cap=1)
    assert facing[FOLD] == 1.0, "folding must stay legal when facing a bet"


def test_the_raise_cap_is_enforced():
    """
    The engine permits unlimited raises per street. The solver was trained with
    a cap, so beyond it every raise is an information set it has never seen.
    """
    mask = np.ones(6, dtype=np.float32)
    under = _constrain(mask, to_call=10, raises_this_street=0, raise_cap=1)
    assert under[2:].any(), "raising should still be available under the cap"

    at_cap = _constrain(mask, to_call=10, raises_this_street=1, raise_cap=1)
    assert not at_cap[2:].any(), "raises must be gone once the cap is reached"
    assert at_cap[CHECK_CALL] == 1.0, "the actor must still have a legal move"


def test_the_actor_is_never_stranded():
    """Whatever is removed, something legal must remain."""
    for to_call in (0, 10):
        for raises in (0, 1, 5):
            mask = _constrain(np.ones(6, dtype=np.float32), to_call, raises, 1)
            assert mask.any(), (to_call, raises)


# ---------------------------------------------------------------------------
# The panel reports when it has stopped being a benchmark
# ---------------------------------------------------------------------------

def test_a_strategy_with_no_entries_is_reported_as_missing():
    """
    A CFR agent whose lookups all miss plays uniformly — it has silently become
    a second random opponent. That has to surface as a number rather than as a
    benchmark that merely looks easy.
    """
    from abstraction.buckets import CardAbstraction
    from evaluation.benchmark import cfr_agent

    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=60, equity_samples=20,
                                  strength="made_hand").fit(np.random.default_rng(0))
    misses = [0, 0]
    empty = cfr_agent({}, abstraction, np.random.default_rng(0), misses)

    result = benchmark(always_call_agent(), empty, "cfr", hands=40, seed=6,
                       misses=misses)
    assert misses[0] > 0, "an empty strategy should miss every lookup"
    assert result.lookup_miss_rate > 0.0, result.summary()


# ---------------------------------------------------------------------------
# The checkpointing convention
# ---------------------------------------------------------------------------

def test_checkpoints_land_ten_times_across_any_run():
    """
    The point of a proportion rather than a constant: the same rule gives ten
    saves whether the run is fifty generations or sixty thousand iterations.
    """
    from evaluation import checkpoint_every, checkpoint_points

    for total in (50, 500, 59_050, 1_000_000):
        assert len(checkpoint_points(total)) == 10, total
        assert checkpoint_every(total) == round(total * 0.10)


def test_a_short_run_still_checkpoints():
    """
    Ten percent of five generations rounds toward zero. A run that saves every
    zero units saves never, and would look identical to one that was working.
    """
    from evaluation import checkpoint_every

    for total in (1, 2, 5, 9):
        assert checkpoint_every(total) >= 1, total
    assert checkpoint_every(0) == 1
    assert checkpoint_every(3) <= 3, "cannot checkpoint less often than never running"


# ---------------------------------------------------------------------------
# Purification
# ---------------------------------------------------------------------------

def test_purification_plays_the_most_probable_action_postflop_and_samples_preflop():
    """
    Purification takes the argmax of the stored average rather than sampling
    it; the ACPC record credits it with +17 to +25 mbb/h for an unconverged
    blueprint. Postflop mode leaves preflop mixed, as Baby Tartanian8 did. Off
    by default, so every panel figure keeps its meaning.
    """
    from types import SimpleNamespace
    from abstraction.buckets import CardAbstraction
    from engine.cards import Card
    from evaluation.benchmark import cfr_agent

    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4).fit(np.random.default_rng(0))
    hole = [Card("A", "h"), Card("K", "h")]
    board = [Card("Q", "s"), Card("7", "d"), Card("2", "c")]
    pre = abstraction.bucket(hole, [])
    post = abstraction.bucket(hole, board, np.random.default_rng(
        hash((tuple(c.index for c in hole), tuple(c.index for c in board))) % (2 ** 32)))
    # Nothing to call: the solver's action list is check/call and the raises.
    mixed = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    strategy = {f"{pre}|": mixed, f"{post}|11/": mixed}
    mask = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def game(board_cards):
        actor = SimpleNamespace(hole_cards=hole, bet=0)
        return SimpleNamespace(players=[actor, SimpleNamespace(hole_cards=[], bet=0)],
                               state=SimpleNamespace(community_cards=board_cards), current_bet=0)

    sampled = cfr_agent(strategy, abstraction, np.random.default_rng(1), raise_cap=1)
    assert len({sampled(game(board), 0, mask, "11/") for _ in range(60)}) > 1
    postflop = cfr_agent(strategy, abstraction, np.random.default_rng(1), raise_cap=1, purify="postflop")
    assert {postflop(game(board), 0, mask, "11/") for _ in range(30)} == {1}
    assert len({postflop(game([]), 0, mask, "") for _ in range(60)}) > 1
    everywhere = cfr_agent(strategy, abstraction, np.random.default_rng(1), raise_cap=1, purify="all")
    assert {everywhere(game([]), 0, mask, "") for _ in range(30)} == {1}
    with pytest.raises(ValueError):
        cfr_agent(strategy, abstraction, np.random.default_rng(1), purify="sometimes")
