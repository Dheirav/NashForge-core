"""Warm-start entries carried from one betting tree to another by action, not by position."""
import pytest

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, RAISE_HALF, RAISE_POT, RAISE_TWO, StreetSchedule
from cfr.warm import actions_at, facing_a_bet, remap, remap_entries

TAPER = (4, 2, 1)
PRE = ((2, 3, 4, 5), (2, 4, 5), (5,))
REST = ((2, 3, 4, 5), (4, 5), (5,))
T1 = StreetSchedule((PRE, REST, REST, REST))
T2 = StreetSchedule((PRE, PRE, PRE, REST))


def test_who_faces_a_bet():
    assert facing_a_bet("")            # the small blind, facing the big blind
    assert not facing_a_bet("1")       # the big blind after a limp
    assert facing_a_bet("3")
    assert not facing_a_bet("31/")     # a new street
    assert facing_a_bet("31/2")


def test_the_same_tree_gives_the_entry_back():
    for key, width in (("7|", 6), ("7|3", 4), ("7|31/", 5), ("7|31/12", 4)):
        probs = [0.1 * (i + 1) for i in range(width)]
        assert remap(key, probs, TAPER, TAPER) == pytest.approx(probs)


def test_an_inserted_size_starts_at_zero_and_the_rest_keep_their_actions():
    # Preflop, after an open: today fold, call, 2x, jam; T1 adds the half-pot re-raise.
    assert actions_at("3", TAPER) == [FOLD, CHECK_CALL, RAISE_TWO, ALL_IN]
    assert actions_at("3", T1) == [FOLD, CHECK_CALL, RAISE_HALF, RAISE_TWO, ALL_IN]
    assert remap("7|3", [0.1, 0.2, 0.3, 0.4], TAPER, T1) == pytest.approx([0.1, 0.2, 0.0, 0.3, 0.4])


def test_the_flop_changes_only_under_t2():
    old = [0.1, 0.2, 0.3, 0.4]            # after a flop bet: fold, call, 2x, jam
    assert remap("7|31/2", old, TAPER, T1) == pytest.approx(old)
    assert remap("7|31/2", old, TAPER, T2) == pytest.approx([0.1, 0.2, 0.0, 0.3, 0.4])
    assert remap("7|31/1", [0.5, 0.5, 0.0, 0.0, 0.0], TAPER, T2) == pytest.approx([0.5, 0.5, 0.0, 0.0, 0.0])


def test_a_stack_capped_node_stays_capped():
    assert remap("7|3", [0.4, 0.6], TAPER, T1) == pytest.approx([0.4, 0.6])


def test_an_entry_whose_width_matches_nothing_is_dropped():
    out, dropped = remap_entries([("7|3", [0.2, 0.3, 0.5]), ("7|3", [0.1, 0.2, 0.3, 0.4])], TAPER, T1)
    assert dropped == 1 and len(out) == 1
