"""
Hand-strength distributions, and clustering them by earth mover's distance.

The abstraction in `buckets.py` clusters situations on one number, expected
hand strength against a random hand. That number cannot tell a draw from a
made hand: 6c6d and KcQc preflop have E[HS] 0.634 and 0.633 and land in the
same bucket, while one of them wins about 63% of showdowns steadily and the
other wins big or loses big depending on the board. Johanson, Burch, Valenzano
and Bowling (AAMAS 2013) showed that clustering on the *distribution* of
end-of-hand strength, with earth mover's distance between histograms, produces
stronger strategies than E[HS] at every abstraction size they tried; Ganzfried
and Sandholm (AAAI 2014) went further with potential-aware distributions. This
module is the histogram feature and the EMD clustering, in Python, as the
reference the native trainer's mirror is pinned against.

A situation's histogram: sample runouts to the river; for each runout, the
hand's equity against a random opponent hand on that final board (a sample of
opponent hands); bin that equity into `bins` equal intervals of [0, 1]. The
E[HS] the old feature used is this histogram's mean, so nothing is lost.

EMD between two one-dimensional histograms with equal mass is the L1 distance
between their cumulative sums, a linear scan. k-means under EMD uses the
histogram mean under L1 in cumulative space as the centroid update, which is
the bin-wise median of the cumulative sums; the mean is used here instead,
which is the standard approximation and is what the 2013 paper's k-means did.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from abstraction.equity import FULL_DECK, remaining_deck
from engine.cards import Card
from engine.hand_eval_fast import score_hand_7_fast

DECK_R = np.array([c.index % 13 for c in FULL_DECK], dtype=np.int64)
DECK_S = np.array([c.index // 13 for c in FULL_DECK], dtype=np.int64)


def strength_histogram(hole: Sequence[Card], board: Sequence[Card], bins: int = 20,
                       runouts: int = 100, opponents: int = 50,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Distribution of the hand's river equity over sampled runouts, as `bins`
    probabilities that sum to one.

    `runouts` boards are completed at random; on each, the hand is scored
    against `opponents` random opponent hands and the win rate (ties half)
    is binned. On the river there is one runout and the histogram is a spike
    at the hand's sampled equity (`opponents` draws, so about 0.07 of noise
    at 50), which is the scalar feature's own estimate; nothing is lost there.
    """
    rng = rng if rng is not None else np.random.default_rng()
    known = list(hole) + list(board)
    pool = remaining_deck(known)
    need = 5 - len(board)
    mine = np.array([c.index for c in hole], dtype=np.int64)
    base = np.array([c.index for c in board], dtype=np.int64)
    hist = np.zeros(bins)
    n_runouts = 1 if need == 0 else runouts
    for _ in range(n_runouts):
        drawn = rng.permutation(pool)
        full = np.concatenate([base, drawn[:need]])
        rest = drawn[need:]
        my_cards = np.concatenate([mine, full])
        my_score = score_hand_7_fast(DECK_R[my_cards], DECK_S[my_cards])
        wins = 0.0
        for _ in range(opponents):
            # Each opponent hand is an independent draw of two cards from the
            # rest of the deck, so the native mirror can match the
            # distribution with a partial shuffle per opponent.
            i = rng.integers(rest.size)
            j = rng.integers(rest.size - 1)
            j = j + 1 if j >= i else j
            opp = rest[[i, j]]
            cards = np.concatenate([opp, full])
            their = score_hand_7_fast(DECK_R[cards], DECK_S[cards])
            wins += 1.0 if my_score > their else (0.5 if my_score == their else 0.0)
        equity = wins / opponents
        hist[min(int(equity * bins), bins - 1)] += 1.0
    return hist / hist.sum()


def emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Earth mover's distance between two equal-mass 1-D histograms, in bins."""
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum())


def kmeans_emd(hists: np.ndarray, k: int, iterations: int = 40,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """
    k-means over histograms with EMD as the distance. Returns centroids sorted
    by mean equity, so "bucket 0 is weakest" holds as it does for the scalar.

    Initialised at quantiles of the mean equity rather than at random, for the
    same reason the scalar fit is: the rows are ordered by a natural statistic
    and quantile seeds remove run-to-run variation from a fixed artifact.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n, bins = hists.shape
    centres = (np.arange(bins) + 0.5) / bins
    means = hists @ centres
    if n <= k:
        # Fewer situations than buckets: the distinct rows, weakest first, as
        # the scalar fit does; duplicate seeds would otherwise leave dead
        # buckets that `num_buckets` still counts.
        uniq = np.unique(hists, axis=0)
        return uniq[np.argsort(uniq @ centres)]
    order = np.argsort(means)
    seeds = order[(np.arange(k) * (n - 1) / max(k - 1, 1)).astype(int)]
    centroids = hists[seeds].copy()
    cum_h = np.cumsum(hists, axis=1)
    for _ in range(iterations):
        cum_c = np.cumsum(centroids, axis=1)
        dist = np.abs(cum_h[:, None, :] - cum_c[None, :, :]).sum(axis=2)     # [n, k]
        assign = dist.argmin(axis=1)
        new = np.array([hists[assign == j].mean(axis=0) if np.any(assign == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    final_order = np.argsort(centroids @ centres)
    return centroids[final_order]


def nearest_emd(centroids: np.ndarray, hist: np.ndarray) -> int:
    """Index of the centroid nearest to `hist` under EMD."""
    return int(np.abs(np.cumsum(centroids, axis=1) - np.cumsum(hist)[None, :]).sum(axis=1).argmin())
