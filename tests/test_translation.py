"""
Action translation: what a strategy thinks it faced when bet a size it lacks.

The mapping sits *inside* a measurement — LBR bets off-tree precisely to escape
the abstraction it is measuring — so a flaw here does not produce a wrong-looking
number, it produces a plausible one. Exploitability inflated by a gameable
mapping reads exactly like exploitability found by a good exploiter.

So the properties asserted here are the ones that make the mapping honest rather
than merely functional: it must agree with the abstraction where the abstraction
already has an answer, it must not concentrate probability at a boundary an
opponent could sit beside, and it must never invent a size that is not there.

    python -m pytest tests/test_translation.py -q
"""
import numpy as np
import pytest

from abstraction.translation import (bracket, pseudo_harmonic_weight, translate,
                                     translation_distribution)

#: The project's raise sizes as pot fractions, per abstraction/betting.py.
SIZES = [0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Agreement with the abstraction
# ---------------------------------------------------------------------------

def test_a_size_the_abstraction_has_is_perceived_as_itself():
    """Translation must be the identity where no translation is needed."""
    for index, size in enumerate(SIZES):
        weights = translation_distribution(SIZES, size)
        assert weights[index] == pytest.approx(1.0), f"{size} -> {weights}"


def test_a_bet_outside_the_range_maps_to_the_nearest_end():
    """
    Below the smallest or above the largest there is no bracketing pair, and
    nothing probabilistic to do: it can only be read as the end it exceeds.
    """
    assert bracket(SIZES, 0.1) == (0, 0)
    assert bracket(SIZES, 9.0) == (2, 2)

    assert translation_distribution(SIZES, 0.1)[0] == pytest.approx(1.0)
    assert translation_distribution(SIZES, 9.0)[-1] == pytest.approx(1.0)


def test_it_never_invents_a_size():
    """All the probability must land on the two straddling sizes, and no others."""
    weights = translation_distribution(SIZES, 0.75)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[2] == 0.0, "a 0.75-pot bet cannot be read as 2x pot"
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# The anti-exploitation property
# ---------------------------------------------------------------------------

def test_the_weighting_is_not_linear_and_favours_the_larger_size():
    """
    At the midpoint an even split would be the naive answer. Pseudo-harmonic
    sends more than half to the *larger* neighbour, which is what removes the
    profit from shading a bet downward toward a boundary.
    """
    weight = pseudo_harmonic_weight(0.5, 1.0, 0.75)
    assert weight == pytest.approx(0.4286, abs=0.001)
    assert weight < 0.5, "shading downward must not be rewarded"


def test_perception_shifts_smoothly_rather_than_snapping():
    """
    A deterministic mapping has a boundary and an exploiter sits just inside it.
    The probability must move gradually across the interval, so that creeping
    downward buys a little less each time rather than everything at once.
    """
    probe = np.linspace(0.5, 1.0, 21)
    weights = [pseudo_harmonic_weight(0.5, 1.0, x) for x in probe]

    # Monotone: more chips can never make it look like a smaller bet.
    assert all(a >= b for a, b in zip(weights, weights[1:])), weights

    # And no single step carries a large jump — the nearest-neighbour mapping
    # would show a step of 1.0 at the midpoint.
    steps = [a - b for a, b in zip(weights, weights[1:])]
    assert max(steps) < 0.15, f"largest step {max(steps):.3f} is a boundary"


def test_the_ends_of_the_interval_are_certain():
    assert pseudo_harmonic_weight(0.5, 1.0, 0.5) == pytest.approx(1.0)
    assert pseudo_harmonic_weight(0.5, 1.0, 1.0) == pytest.approx(0.0)


def test_a_bet_just_above_the_smaller_size_is_still_read_as_smaller():
    """The property the whole mapping exists for, stated directly."""
    assert pseudo_harmonic_weight(0.5, 1.0, 0.51) > 0.95
    assert pseudo_harmonic_weight(0.5, 1.0, 0.99) < 0.05


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_follows_the_distribution():
    """`translate` draws from what `translation_distribution` describes."""
    expected = translation_distribution(SIZES, 0.75)
    rng = np.random.default_rng(0)

    draws = np.zeros(len(SIZES))
    for _ in range(4000):
        draws[translate(SIZES, 0.75, rng)] += 1
    draws /= draws.sum()

    assert draws == pytest.approx(expected, abs=0.02), f"{draws} vs {expected}"


def test_sampling_is_deterministic_given_a_generator():
    first = [translate(SIZES, 0.8, np.random.default_rng(7)) for _ in range(5)]
    second = [translate(SIZES, 0.8, np.random.default_rng(7)) for _ in range(5)]
    assert first == second


def test_degenerate_interval_does_not_divide_by_zero():
    assert pseudo_harmonic_weight(1.0, 1.0, 1.0) == 1.0
    assert translation_distribution([1.0], 0.6)[0] == pytest.approx(1.0)
