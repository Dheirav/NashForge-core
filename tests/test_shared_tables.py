"""Histogram rungs with identical bucket tables share one native set and one set of arrays."""
import gc
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

native = pytest.importorskip("pokerbot_native")

from abstraction import buckets as B  # noqa: E402
from abstraction.buckets import CardAbstraction  # noqa: E402


def tables(n_flop=50, n_turn=80, seed=0):
    rng = np.random.default_rng(seed)
    return {"flop": (np.arange(n_flop, dtype=np.uint64) * 7 + 1, rng.integers(0, 20, n_flop).astype(np.uint8)),
            "turn": (np.arange(n_turn, dtype=np.uint64) * 11 + 3, rng.integers(0, 20, n_turn).astype(np.uint8))}


def abstraction_with(t):
    a = CardAbstraction.__new__(CardAbstraction)
    a._hist_tables = {s: (k.copy(), b.copy()) for s, (k, b) in t.items()}
    a._native_tables = None
    return a


def test_identical_tables_share_one_native_set_and_one_set_of_arrays():
    a, b = abstraction_with(tables()), abstraction_with(tables())
    ta, tb = a.native_tables(), b.native_tables()
    assert ta is tb
    assert a._hist_tables["turn"][0] is b._hist_tables["turn"][0]       # the duplicate arrays are let go
    assert a._hist_tables is not b._hist_tables                         # but each rung keeps a dict of its own
    assert a.native_tables() is ta                                      # and a second call is the cache


@pytest.mark.parametrize("change", ["last bucket", "first key", "one entry more", "streets swapped"])
def test_tables_that_differ_anywhere_do_not_share(change):
    base = tables(n_flop=3_000, n_turn=200_000)           # far past any sample a lazy digest might take
    other = {s: (k.copy(), b.copy()) for s, (k, b) in base.items()}
    if change == "last bucket":
        other["turn"][1][-1] ^= 1                                       # the very end: a sampled digest misses it
    elif change == "first key":
        other["flop"][0][0] += 1
    elif change == "one entry more":
        k, b = other["turn"]
        other["turn"] = (np.append(k, k[-1] + np.uint64(1)), np.append(b, np.uint8(0)))   # keys stay sorted
    else:
        other = {"flop": base["turn"], "turn": base["flop"]}
    a, b = abstraction_with(base), abstraction_with(other)
    assert a.native_tables() is not b.native_tables()
    assert B._tables_digest(a._hist_tables) != B._tables_digest(b._hist_tables)


def test_each_street_gets_its_own_table():
    a = abstraction_with(tables(n_flop=30, n_turn=90))
    t = a.native_tables()
    assert set(t) == {"flop", "turn"}
    assert len(t["flop"]) == 30 and len(t["turn"]) == 90


def test_the_same_values_in_another_dtype_share():
    # The native table is built from uint64 keys either way, so equal values are an equal table.
    t = tables()
    wide = {s: (k.astype(np.int64), b.astype(np.int64)) for s, (k, b) in t.items()}
    assert abstraction_with(t).native_tables() is abstraction_with(wide).native_tables()


def test_the_shared_arrays_are_read_only():
    a, b = abstraction_with(tables()), abstraction_with(tables())
    a.native_tables(); b.native_tables()
    with pytest.raises(ValueError):
        b._hist_tables["flop"][1][0] = 3                                # would have changed a's buckets too
    # Reassigning a street in one rung's dict leaves the other alone.
    b._hist_tables["flop"] = None
    assert a._hist_tables["flop"] is not None


def test_a_shared_set_goes_with_the_last_rung_holding_it():
    a, b = abstraction_with(tables(seed=9)), abstraction_with(tables(seed=9))
    a.native_tables(); b.native_tables()
    digest = B._tables_digest(a._hist_tables)
    assert digest in B._SHARED_TABLES
    del a; gc.collect()
    assert digest in B._SHARED_TABLES                                   # b still holds it
    del b; gc.collect()
    assert digest not in B._SHARED_TABLES


def test_no_tables_is_an_empty_set_and_nothing_cached():
    before = len(B._SHARED_TABLES)
    for empty in (None, {}):
        a = CardAbstraction.__new__(CardAbstraction)
        a._hist_tables, a._native_tables = empty, None
        assert a.native_tables() == {}
    assert len(B._SHARED_TABLES) == before


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(5)
    a = CardAbstraction(preflop_buckets=169, postflop_buckets=4, samples=40, equity_samples=20, texture=True,
                        strength="histogram", hist_bins=8, hist_runouts=1, hist_opponents=1).fit(rng)
    return a.build_tables(threads=2)


def test_real_tables_shared_across_pickled_copies_look_up_as_unshared(fitted):
    from engine.cards import RANKS, SUITS, Card
    copies = [pickle.loads(pickle.dumps(fitted)) for _ in range(3)]
    own = {s: native.BucketTable(np.ascontiguousarray(k, dtype=np.uint64), np.ascontiguousarray(b, dtype=np.uint8))
           for s, (k, b) in fitted._hist_tables.items()}
    assert copies[0].native_tables() is copies[1].native_tables() is copies[2].native_tables()
    rng = np.random.default_rng(1)
    card = lambda i: Card(RANKS[i % 13], SUITS[i // 13])  # noqa: E731
    for _ in range(500):
        c = [int(x) for x in rng.choice(52, 6, replace=False)]
        n = 3 if rng.random() < 0.5 else 4
        street = "flop" if n == 3 else "turn"
        want = own[street].lookup(c[:2], c[2:2 + n])
        assert all(a.native_tables()[street].lookup(c[:2], c[2:2 + n]) == want for a in copies)
        assert len({a.bucket([card(i) for i in c[:2]], [card(i) for i in c[2:2 + n]]) for a in copies}) == 1
    # A shared rung still pickles, and its copy looks up the same.
    again = pickle.loads(pickle.dumps(copies[0]))
    assert again.native_tables() is copies[0].native_tables()

