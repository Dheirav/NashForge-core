"""Purification and thresholding, applied the same way to the table LBR reads and to the agent that plays."""
import numpy as np
import pytest

from cfr.purify import apply_to_table, row, valid


def test_the_modes():
    p = [0.05, 0.15, 0.5, 0.3]
    assert list(row(p, "none", True)) == p
    assert list(row(p, "all", False)) == [0, 0, 1.0, 0]
    assert list(row(p, "postflop", False)) == p                  # preflop untouched
    assert list(row(p, "postflop", True)) == [0, 0, 1.0, 0]
    assert row(p, "t10", True) == pytest.approx([0, 0.15 / 0.95, 0.5 / 0.95, 0.3 / 0.95])
    assert row(p, "t20", True) == pytest.approx([0, 0, 0.5 / 0.8, 0.3 / 0.8])


def test_a_threshold_nothing_clears_keeps_the_largest():
    assert row([0.3, 0.35, 0.35], "t50", True) == pytest.approx([0, 1.0, 0])


def test_mass_is_kept_and_bad_modes_refused():
    assert row([0.2, 0.8], "t30", True).sum() == pytest.approx(1.0)
    assert not valid("t0") and not valid("tx") and not valid("x") and valid("t15")
    with pytest.raises(ValueError):
        row([0.5, 0.5], "t", True)


def test_a_table_is_purified_by_street():
    table = {"3|": [0.1, 0.9], "3|1/": [0.6, 0.4]}
    out = apply_to_table(table, "postflop")
    assert list(out["3|"]) == [0.1, 0.9] and list(out["3|1/"]) == [1.0, 0.0]


def test_the_flat_table_is_purified_like_the_dict():
    from cfr.flat import FlatStrategy
    keys = np.array([b"3|", b"3|1/"], dtype="S8")
    flat = FlatStrategy(keys, np.array([0, 2, 4]), np.array([0.1, 0.9, 0.6, 0.4]))
    out = apply_to_table(flat, "t20")
    assert list(out["3|"]) == pytest.approx([0.0, 1.0]) and list(out["3|1/"]) == pytest.approx([0.6, 0.4])
