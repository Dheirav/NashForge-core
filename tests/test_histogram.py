"""
The histogram feature must separate what expected hand strength cannot.

6c6d and KcQc preflop have E[HS] 0.634 and 0.633 (Johanson et al. 2013), so a
scalar abstraction puts them in one bucket; their end-of-hand strength
distributions are nothing alike. This pins that the histogram sees the
difference, that EMD is a proper distance on these histograms, and that
clustering under it keeps the "bucket 0 is weakest" order the solver relies on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.histogram import emd_1d, kmeans_emd, nearest_emd, strength_histogram  # noqa: E402
from engine.cards import Card  # noqa: E402


def _hand(a, b):
    return [Card(a[0], a[1]), Card(b[0], b[1])]


def test_a_pair_and_two_overcards_share_a_mean_but_not_a_histogram():
    rng = np.random.default_rng(3)
    pair = strength_histogram(_hand("6c", "6d"), [], bins=20, runouts=200, opponents=60, rng=rng)
    overs = strength_histogram(_hand("Kc", "Qc"), [], bins=20, runouts=200, opponents=60, rng=rng)
    centres = (np.arange(20) + 0.5) / 20
    assert abs(pair @ centres - overs @ centres) < 0.06, "the means should be close, that is the point"
    # The pair's mass sits in the middle; the overcards' mass is at both ends.
    assert emd_1d(pair, overs) > 1.5, f"EMD {emd_1d(pair, overs):.2f}: the histograms should differ"
    assert overs[:6].sum() > pair[:6].sum() and overs[-5:].sum() > pair[-5:].sum()


def test_emd_is_a_distance_and_a_river_hand_is_a_spike():
    rng = np.random.default_rng(5)
    a = strength_histogram(_hand("As", "Kd"), [], bins=10, runouts=50, opponents=30, rng=rng)
    b = strength_histogram(_hand("7h", "2c"), [], bins=10, runouts=50, opponents=30, rng=rng)
    assert emd_1d(a, a) == 0.0 and abs(emd_1d(a, b) - emd_1d(b, a)) < 1e-9 and emd_1d(a, b) > 0
    river = strength_histogram(_hand("As", "Kd"), _hand("Ah", "7d") + _hand("2c", "9s") + [Card("3", "h")],
                               bins=10, runouts=1, opponents=200, rng=rng)
    assert river.max() == 1.0, "one runout on the river: the histogram is a single spike"


def test_kmeans_emd_orders_centroids_weakest_first_and_assigns_consistently():
    rng = np.random.default_rng(7)
    hists = np.vstack([np.histogram(rng.beta(a, b, 200), bins=10, range=(0, 1))[0] / 200.0
                       for a, b in [(1, 5), (2, 5), (3, 3), (5, 2), (5, 1)] * 20])
    centroids = kmeans_emd(hists, 4, rng=rng)
    centres = (np.arange(10) + 0.5) / 10
    means = centroids @ centres
    assert np.all(np.diff(means) > 0), f"centroids not ordered weakest first: {means}"
    weak = np.histogram(rng.beta(1, 6, 200), bins=10, range=(0, 1))[0] / 200.0
    strong = np.histogram(rng.beta(6, 1, 200), bins=10, range=(0, 1))[0] / 200.0
    assert nearest_emd(centroids, weak) < nearest_emd(centroids, strong)


def test_the_native_histogram_matches_the_python_in_distribution_and_the_lookup_exactly():
    """
    The trainer buckets in C++ and the player in Python, so the two must agree
    on what a histogram is. Streams differ (each seeds its own generator), so
    the pin is distributional: over 30 situations, the average EMD between the
    Python and native histograms must be within the noise of two independent
    Python draws. The nearest-centroid lookup is deterministic and is pinned
    exactly on the same input.
    """
    import pokerbot_native as native
    from abstraction.equity import FULL_DECK
    rng = np.random.default_rng(11)
    cross, within = [], []
    for _ in range(30):
        deck = rng.permutation(52)
        n = int(rng.choice([3, 4]))
        hole = [FULL_DECK[int(c)] for c in deck[:2]]
        board = [FULL_DECK[int(c)] for c in deck[2:2 + n]]
        py1 = strength_histogram(hole, board, 10, 80, 40, np.random.default_rng(int(rng.integers(1 << 30))))
        py2 = strength_histogram(hole, board, 10, 80, 40, np.random.default_rng(int(rng.integers(1 << 30))))
        cpp = np.array(native.strength_histogram([int(c) for c in deck[:2]], [int(c) for c in deck[2:2 + n]],
                                                 10, 80, 40, int(rng.integers(1 << 30))))
        assert abs(cpp.sum() - 1.0) < 1e-9
        cross.append(emd_1d(py1, cpp))
        within.append(emd_1d(py1, py2))
    assert np.mean(cross) < 1.5 * np.mean(within) + 0.1, (
        f"native vs python EMD {np.mean(cross):.3f}, python vs python {np.mean(within):.3f}")
    hists = np.vstack([np.histogram(rng.beta(a, b, 100), bins=10, range=(0, 1))[0] / 100.0
                       for a, b in [(1, 5), (3, 3), (5, 1)] * 10])
    centroids = kmeans_emd(hists, 3, rng=rng)
    for h in hists[:12]:
        assert native.nearest_emd(centroids.tolist(), h.tolist()) == nearest_emd(centroids, h)


def test_the_bucket_table_covers_every_flop_and_agrees_with_the_histogram():
    # The table exists because a histogram costs 0.4 ms and a solve looked
    # one up on unseen cards several times an iteration (5.0 ms/it against
    # 0.30 with the histogram made cheap, 21 September 2026). Its key is the
    # suit-isomorphic form, so it must be invariant under suit renaming,
    # complete, and equal to the direct computation seeded from the same key.
    native = __import__("pokerbot_native")
    rng = np.random.default_rng(11)
    for _ in range(500):
        cards = rng.choice(52, 6, replace=False)
        hole, board = [int(c) for c in cards[:2]], [int(c) for c in cards[2:2 + rng.integers(3, 5)]]
        perm = rng.permutation(4)
        renamed = lambda c: int(perm[c // 13]) * 13 + c % 13
        assert native.canonical_key(hole, board) == native.canonical_key([renamed(h) for h in hole], [renamed(b) for b in board])
        assert native.canonical_key(hole, board) == native.canonical_key(hole[::-1], board[::-1])
    bins, runouts, opponents = 8, 3, 4
    centroids = [[1.0 / bins] * bins, [0.0] * (bins - 1) + [1.0], [1.0] + [0.0] * (bins - 1)]
    table = native.build_bucket_table(3, centroids, bins, runouts, opponents, 2)
    assert len(table) == 1_286_792, "every canonical flop situation, exactly once"
    for _ in range(300):
        cards = [int(c) for c in rng.choice(52, 5, replace=False)]
        hole, board = cards[:2], cards[2:]
        got = table.lookup(hole, board)
        assert got >= 0
        # Direct computation on the table's own representative and seed.
        key = native.canonical_key(hole, board)
        rep = [(key >> s) & 63 for s in (24, 18, 12, 6, 0)]
        hist = native.strength_histogram(rep[:2], rep[2:], bins, runouts, opponents, (key * 0x9E3779B97F4A7C15) % 2 ** 64)
        assert got == native.nearest_emd(centroids, hist)


def test_a_fitted_histogram_abstraction_with_tables_pickles_and_looks_up_through_them():
    import pickle
    from abstraction.buckets import CardAbstraction
    native = __import__("pokerbot_native")
    rng = np.random.default_rng(5)
    abstraction = CardAbstraction(preflop_buckets=169, postflop_buckets=4, samples=40, equity_samples=20,
                                  texture=True, strength="histogram", hist_bins=8, hist_runouts=1,
                                  hist_opponents=1).fit(rng)
    abstraction.build_tables(threads=2)
    assert set(abstraction._hist_tables) == {"flop", "turn"}
    again = pickle.loads(pickle.dumps(abstraction))
    hole = _hand("Ah", "Kh")
    for board in ([Card("2", "c"), Card("7", "d"), Card("T", "s")],
                  [Card("2", "c"), Card("7", "d"), Card("T", "s"), Card("Q", "h")]):
        expect = again.native_tables()["flop" if len(board) == 3 else "turn"].lookup(
            [c.index for c in hole], [c.index for c in board])
        expect += 4 * native.board_texture([c.index for c in board])
        assert again.bucket(hole, board, np.random.default_rng(0)) == expect
        assert abstraction.bucket(hole, board, np.random.default_rng(0)) == expect
    # The river has no table and still answers.
    river = [Card("2", "c"), Card("7", "d"), Card("T", "s"), Card("Q", "h"), Card("3", "h")]
    assert 0 <= again.bucket(hole, river, np.random.default_rng(0)) < again.num_buckets("river")


def test_the_native_large_sample_fit_orders_its_classes_and_the_default_fit_is_unchanged():
    # Opt-in: the default fit must stay byte-for-byte what a ladder mid-training
    # started with, so a rung fitted tonight matches one fitted this afternoon.
    from abstraction.buckets import CardAbstraction
    kwargs = dict(preflop_buckets=169, postflop_buckets=5, samples=30, equity_samples=20,
                  texture=False, strength="histogram", hist_bins=8, hist_runouts=2, hist_opponents=3)
    a = CardAbstraction(**kwargs).fit(np.random.default_rng(1))
    b = CardAbstraction(**kwargs).fit(np.random.default_rng(1))
    for street in ("flop", "turn", "river"):
        assert np.array_equal(a._hist_centroids[street], b._hist_centroids[street])
    c = CardAbstraction(**kwargs).fit(np.random.default_rng(1), fit_samples=400)
    centres = (np.arange(8) + 0.5) / 8
    for street in ("flop", "turn", "river"):
        means = c._hist_centroids[street] @ centres
        assert np.all(np.diff(means) >= 0), "classes are ordered weakest first"
        assert c._hist_centroids[street].shape == (5, 8)
        assert np.allclose(c._hist_centroids[street].sum(axis=1), 1.0)
