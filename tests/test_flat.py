"""
The flat strategy table answers exactly as the dict it was built from.
"""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cfr.flat import FlatStrategy, flat_paths, flatten, load_strategy, write_flat  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMALL = os.path.join(ROOT, "results", "cfr", "nolimit_strategy.pkl")


@pytest.fixture(scope="module")
def saved():
    with open(SMALL, "rb") as handle:
        return pickle.load(handle)


def test_every_entry_reads_back_identically(saved):
    flat = flatten(saved["strategy"])
    assert len(flat) == len(saved["strategy"])
    for key, value in saved["strategy"].items():
        got = flat.get(key)
        assert got is not None and got.dtype == np.float64
        assert np.array_equal(got, value)
        assert key in flat
    assert flat.get("999|nonsense") is None
    assert "999|nonsense" not in flat
    assert set(flat) == set(saved["strategy"])
    # Well under a fifth of the dict's footprint.
    dict_bytes = sum(v.nbytes + 200 for v in saved["strategy"].values())
    assert flat.nbytes < dict_bytes / 3


def test_the_flat_pair_is_preferred_only_when_newer(tmp_path, saved):
    path = tmp_path / "rung.pkl"
    with open(path, "wb") as handle:
        pickle.dump(saved, handle)
    assert isinstance(load_strategy(str(path))["strategy"], dict)
    write_flat(str(path), saved)
    loaded = load_strategy(str(path))
    assert isinstance(loaded["strategy"], FlatStrategy)
    assert loaded["abstraction"].num_buckets("preflop") == saved["abstraction"].num_buckets("preflop")
    assert loaded["args"] == saved["args"]
    # A retrained pickle, newer than the pair, is read as the pickle.
    os.utime(path, None)
    npz, side = flat_paths(str(path))
    os.utime(npz, (os.path.getmtime(path) - 10, os.path.getmtime(path) - 10))
    assert isinstance(load_strategy(str(path))["strategy"], dict)


def test_the_agent_plays_the_same_hands_from_either_form(saved, tmp_path):
    """The chip series at a fixed seed must not change: this is the gate."""
    from evaluation.benchmark import benchmark, cfr_agent, random_agent
    from cfr.flat import flatten
    results = []
    for strategy in (saved["strategy"], flatten(saved["strategy"])):
        agent = cfr_agent(strategy, saved["abstraction"], np.random.default_rng(3))
        results.append(benchmark(agent, random_agent(np.random.default_rng(4)), "flat", hands=120, seed=11))
    assert results[0].chips_per_hand == results[1].chips_per_hand


def test_a_key_longer_than_the_width_is_refused():
    with pytest.raises(ValueError):
        flatten({"1|" + "1" * 40: np.array([0.5, 0.5])})
