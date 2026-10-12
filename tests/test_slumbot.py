"""
The Slumbot betting string, parsed.

No network. These cover the reading of Slumbot's action notation, which is where
the first class of translation bug lives: a misread betting string produces a
legal-looking action at the wrong price, and the result is a losing number that
looks like a weak strategy rather than a bug. `docs/EXTERNAL_BENCHMARK.md` is
explicit that the two are indistinguishable from the result alone, which is why
this is tested away from the server.

The notation, confirmed against the live API: ``b200`` bets *to* 200, ``c``
calls, ``k`` checks, ``f`` folds, ``/`` separates streets.
"""
import pytest

from slumbot.api import BIG_BLIND, SMALL_BLIND, STARTING_STACK, HandState


def state(action, **kwargs):
    return HandState(token="t", action=action, old_action="", client_pos=0,
                     hole_cards=["As", "Kd"], board=[], **kwargs)


# --- streets ---------------------------------------------------------------

@pytest.mark.parametrize("action,expected", [
    ("", 0),
    ("b200", 0),
    ("b200c/", 1),
    ("b200c/kk/", 2),
    ("b200c/kb200c/kk/b500", 3),
])
def test_street_counts_separators(action, expected):
    assert state(action).street == expected


# --- who owes what ---------------------------------------------------------

@pytest.mark.parametrize("action,facing", [
    ("", False),
    ("b200", True),
    ("b200c", False),
    ("b200c/", False),
    ("b200c/k", False),
    ("b200c/kb200", True),
    ("b200c/kb200c/", False),
])
def test_facing_bet_reads_only_the_current_street(action, facing):
    """
    A bet on a previous street is not a bet to call now.

    Reading the whole string rather than the current street is the obvious way to
    get this wrong, and it would send `c` into a spot where only `k` is legal --
    rejected by the server, which is the mercy. The reverse, sending `k` when
    facing a bet, is the same bug in the direction that also gets rejected.
    """
    assert state(action).facing_bet is facing


@pytest.mark.parametrize("action,levels", [
    ("", []),
    ("b200", [200]),
    ("b200c/", []),
    ("b200c/kb400", [400]),
    ("b200b600", [200, 600]),
    ("b200b600b1800", [200, 600, 1800]),
])
def test_bet_levels_are_cumulative_not_increments(action, levels):
    """
    Slumbot's amounts are levels bet *to*, not increments.

    `b600` after `b200` is a raise to 600, costing 400 more -- not a second bet of
    600. This exposes the levels and stops there: preflop the blinds are already
    committed and absent from the string, so the big blind facing `b200` owes 100,
    not 200. Turning levels into an amount owed needs position and the posted
    blinds, which belong to the translation layer.
    """
    assert state(action).bet_levels() == levels


# --- terminal states -------------------------------------------------------

def test_a_hand_is_over_only_when_winnings_arrive():
    assert state("b200c/kk/kk/kk").over is False
    assert state("b200c/kk/kk/kk", winnings=200).over is True


def test_zero_winnings_still_counts_as_over():
    """A split pot pays nothing and is still a finished hand."""
    assert state("b200c/kk/kk/kk", winnings=0).over is True


# --- the stakes ------------------------------------------------------------

def test_the_stakes_are_the_ones_measured_against_the_server():
    """
    Slumbot plays 200 big blinds; this project's solver was fitted for 100.

    Pinned as a test because it is the single most consequential fact about the
    bridge and the easiest to carry over wrongly from a reference implementation.
    A strategy fitted for 100bb is off-tree at 200bb from the first decision.
    """
    assert (SMALL_BLIND, BIG_BLIND, STARTING_STACK) == (50, 100, 20_000)
    assert STARTING_STACK / BIG_BLIND == 200


def test_the_small_blind_faces_a_bet_before_any_action():
    """The blinds are not in the action string; the button still owes half a blind."""
    from slumbot.api import HandState, always_fold
    sb = HandState(token="t", action="", old_action="", client_pos=1, hole_cards=["7h", "2d"], board=[])
    bb = HandState(token="t", action="", old_action="", client_pos=0, hole_cards=["7h", "2d"], board=[])
    assert sb.facing_bet and always_fold(sb) == "f"
    assert not bb.facing_bet and always_fold(bb) == "k"
