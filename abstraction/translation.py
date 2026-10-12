"""
Mapping a bet the abstraction does not contain onto the ones it does.

A solver trained on four raise sizes has no answer to a bet of 0.7 pot. Something
must decide what it *thinks* it faced, and that decision is not cosmetic: it is
the seam an opponent attacks. Bet just under the boundary where the mapping
flips, and a player who reasons about pot-sized bets can be made to defend as
though facing a half-pot one.

Two mappings and why the obvious one is wrong
---------------------------------------------
**Nearest neighbour** snaps 0.74 pot to half-pot and 0.76 to pot. It is
deterministic, so an exploiter finds the boundary and sits just inside it,
paying 0.74 to be defended against as though it had paid 0.50. Exploitability
measured through it is partly an artifact of the mapping rather than a fact
about the strategy — a false positive, which is worse than a bound that is
merely loose.

**Pseudo-harmonic** (Ganzfried & Sandholm, 2013) maps probabilistically to the
two neighbours, weighted so that the gain from creeping toward a boundary decays
smoothly instead of jumping at it. For neighbouring sizes ``A < x < B``, all
expressed as fractions of the pot:

    f_A(x) = (B - x)(1 + A) / ((B - A)(1 + x))

and ``f_B = 1 - f_A``. The weighting is deliberately not linear. At the midpoint
of [0.5, 1.0] it sends 43% to the smaller size and 57% to the larger, so a bet
placed just below a boundary is still read as the larger one more often than
even odds would suggest, and the incentive to shade downward is blunted.

The paper derives it as the mapping under which the opponent's exploitation of
the translation itself is bounded, which is the property that matters when the
translation sits inside a measurement. See `cfr/lbr.py`, which bets off-tree
precisely to escape the abstraction it is measuring.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def pseudo_harmonic_weight(smaller: float, larger: float, actual: float) -> float:
    """
    Probability that ``actual`` is perceived as ``smaller`` rather than ``larger``.

    All three are bet sizes as fractions of the pot. ``actual`` is expected to
    lie in ``[smaller, larger]``; it is clamped if not, so a caller that has
    already chosen the bracketing pair cannot produce a probability outside
    [0, 1] through floating-point drift.

    Args:
        smaller: The lower neighbouring abstract size, ``A``.
        larger: The upper neighbouring abstract size, ``B``.
        actual: The size actually bet, ``x``.

    Returns:
        ``f_A(x)`` in [0, 1].
    """
    if larger <= smaller:
        return 1.0
    actual = min(max(actual, smaller), larger)
    weight = ((larger - actual) * (1.0 + smaller)) / ((larger - smaller) * (1.0 + actual))
    return float(min(max(weight, 0.0), 1.0))


def bracket(sizes: Sequence[float], actual: float) -> Tuple[int, int]:
    """
    Indices of the two abstract sizes straddling ``actual``.

    Outside the range the pair collapses to the single nearest end: a bet larger
    than anything in the abstraction can only be perceived as the largest, and
    there is nothing probabilistic about that.
    """
    ordered = list(sizes)
    if actual <= ordered[0]:
        return 0, 0
    if actual >= ordered[-1]:
        return len(ordered) - 1, len(ordered) - 1

    upper = next(i for i, size in enumerate(ordered) if size >= actual)
    return upper - 1, upper


def translate(sizes: Sequence[float], actual: float,
              rng: np.random.Generator) -> int:
    """
    Index of the abstract size ``actual`` is perceived as, sampled.

    Randomised rather than rounded: a deterministic mapping has a boundary, and
    a boundary is a thing to sit just inside of.
    """
    low, high = bracket(sizes, actual)
    if low == high:
        return low

    weight = pseudo_harmonic_weight(sizes[low], sizes[high], actual)
    return low if rng.random() < weight else high


def translation_distribution(sizes: Sequence[float], actual: float) -> np.ndarray:
    """
    The full distribution over abstract sizes, rather than a single draw.

    Useful where an expectation is wanted instead of a sample — computing how
    often an opponent folds to an off-tree bet, say, which is an average over
    both perceptions rather than one of them.
    """
    weights = np.zeros(len(sizes))
    low, high = bracket(sizes, actual)
    if low == high:
        weights[low] = 1.0
        return weights

    weight = pseudo_harmonic_weight(sizes[low], sizes[high], actual)
    weights[low] = weight
    weights[high] = 1.0 - weight
    return weights
