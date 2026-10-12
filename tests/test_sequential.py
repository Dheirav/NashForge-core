"""
The stopping rule's guarantee, checked by simulation: peeking does not raise the error rate.

Every simulation is seeded, so a failure here is a property failing, not an unlucky draw.
"""
import json
import os
import sys

import numpy as np
import pytest

from evaluation.sequential import (ALPHA, PRIOR, bernoulli_cs, chips_verdict_one, chips_verdict_two,
                                   log_mixture_ratio, match_verdict_one, match_verdict_two, mean_cs)

RUNS, MATCHES, EVERY = 2000, 200, 20


def _wins(p, seed, runs=RUNS, n=MATCHES):
    return np.random.default_rng(seed).random((runs, n)) < p


def _rejects_at_looks(wins, p0=0.5, looks=None):
    """Whether p0 is rejected at any look; at looks=None every single match is a look."""
    s = np.cumsum(wins, axis=1)
    n = np.arange(1, wins.shape[1] + 1)
    rej = log_mixture_ratio(p0, s, n, PRIOR) >= np.log(1 / ALPHA)
    if looks is not None:
        rej = rej[:, looks - 1]
    return rej.any(axis=1)


def test_a_fair_coin_peeked_every_burst_stays_under_alpha():
    looks = np.arange(EVERY, MATCHES + 1, EVERY)
    assert _rejects_at_looks(_wins(0.5, 1), looks=looks).mean() <= ALPHA


def test_a_fair_coin_peeked_after_every_match_stays_under_alpha():
    # the strongest form of peeking; Ville's inequality covers it, so it must hold too
    assert _rejects_at_looks(_wins(0.5, 2)).mean() <= ALPHA


def test_the_fixed_test_it_replaces_does_inflate():
    # the problem being solved: a z-test at 1.96 read at the same ten looks
    wins = _wins(0.5, 3)
    looks = np.arange(EVERY, MATCHES + 1, EVERY)
    s = np.cumsum(wins, axis=1)[:, looks - 1]
    z = (s - 0.5 * looks) / np.sqrt(0.25 * looks)
    assert (np.abs(z) > 1.96).any(axis=1).mean() > 0.10


def test_two_equal_versions_rarely_separate_under_peeking():
    # one call at 200 matches counts every earlier peek, because the interval only ever narrows
    a, b = _wins(0.5, 4, runs=1000), _wins(0.5, 5, runs=1000)
    assert match_verdict_two(list(a[0, :20]), list(b[0, :20])).verdict == "undecided"
    false = sum(match_verdict_two(list(x), list(y)).verdict != "undecided" for x, y in zip(a, b))
    assert false / len(a) <= ALPHA


def test_a_clearly_better_version_is_found_well_before_200():
    rej = np.cumsum(_wins(0.7, 6), axis=1)
    n = np.arange(1, MATCHES + 1)
    hit = log_mixture_ratio(0.5, rej, n, PRIOR) >= np.log(1 / ALPHA)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, MATCHES + 1)
    assert hit.any(axis=1).mean() >= 0.95
    assert np.median(first) <= 80


def test_the_verdict_names_the_better_version():
    rng = np.random.default_rng(7)
    a, b = list(rng.random(200) < 0.75), list(rng.random(200) < 0.40)
    assert match_verdict_two(a, b, names=("A", "B")).verdict == "A better"
    assert match_verdict_two(b, a, names=("A", "B")).verdict == "B better"
    assert match_verdict_one(a, name="A").verdict == "A better than 50%"


def test_no_matches():
    v = match_verdict_one([], name="A")
    assert v.verdict == "undecided" and v.interval == (0.0, 1.0)
    assert match_verdict_two([], [True]).verdict == "undecided"
    assert chips_verdict_one([], name="A").verdict == "undecided"
    assert bernoulli_cs(0, 0) == (0.0, 1.0)


def test_all_wins():
    v = match_verdict_one([True] * 20, name="A")
    assert v.interval[1] == 1.0 and v.interval[0] > 0.5
    assert v.verdict == "A better than 50%"
    lo, hi = bernoulli_cs(3, 3)                     # three wins are not yet evidence of anything
    assert hi == 1.0 and lo < 0.5


def test_one_burst_of_14_from_20_is_undecided_with_a_labelled_estimate():
    # interleaved: fourteen straight wins first would be real evidence and is rightly decided
    v = match_verdict_one([True, True, False, True, True, False, True] * 2 + [True, False] * 3, name="A")
    assert v.verdict == "undecided"
    assert v.projection.startswith("estimate: about")
    assert v.interval[0] < 0.5 < v.interval[1]


def test_the_interval_covers_the_estimate_and_narrows():
    w = list(_wins(0.6, 8, runs=1, n=400)[0])
    wide = match_verdict_one(w[:40]).interval
    narrow = match_verdict_one(w).interval
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert narrow[0] <= np.mean(w) <= narrow[1]


def _hands(rng, mean, size):
    # mostly small pots with occasional all-ins, the shape of real per-hand nets
    small = rng.normal(mean, 300, size)
    jam = rng.random(size) < 0.03
    return np.where(jam, rng.choice([-1, 1], size) * 8000 + mean, small).clip(-10_000, 10_000)


def test_chips_peeking_under_a_zero_mean_stays_under_alpha():
    # the interval is a running intersection, so a verdict reached at any earlier look is still
    # the verdict at the last one; one call per run therefore counts every peek
    rng = np.random.default_rng(9)
    runs = [_hands(rng, 0.0, 600) for _ in range(120)]
    assert all(chips_verdict_one(runs[0][:n]).verdict == "undecided" for n in range(100, 601, 100))
    false = sum(chips_verdict_one(x).verdict != "undecided" for x in runs)
    assert false / len(runs) <= ALPHA


def test_chips_a_real_edge_is_found_and_clipping_is_reported():
    rng = np.random.default_rng(10)
    x = rng.normal(400, 300, 1500)
    assert chips_verdict_one(x, name="A").verdict == "A above +0"
    assert chips_verdict_two(x, x - 800, names=("A", "B")).verdict == "A better"
    v = chips_verdict_one(_hands(rng, 0.0, 300), bound=2000)
    assert "clip ±2,000" in v.notes[0] and not v.notes[0].startswith("clip ±2,000 chips, 0 of")


def test_mean_cs_contains_the_mean_and_respects_the_bound():
    x = np.full(50, 10_000.0)
    lo, hi, clipped = mean_cs(x)
    assert clipped == 0 and hi == pytest.approx(10_000) and lo > 0
    lo, hi, clipped = mean_cs(np.full(50, 30_000.0))
    assert clipped == 50 and hi <= 10_000

