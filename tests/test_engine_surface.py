"""
The parts of the engine's public surface nothing else exercises.

`tests/` was an empty directory before the August audit, and three hand-run
scripts under `scripts/testing/` stood in for it. Those scripts were removed on
15 August: the suite had grown to cover almost everything they touched, and what
they did uniquely they checked by printing it rather than asserting it — a test
that passes because nobody read its output is not a test.

Four things survived that review because nothing in the suite covers them:
raise-sizing hints, hand-history logging, the action history the engine records,
and whether the interactive CLI can play a hand at all. They are here with
assertions this time.

    python -m pytest tests/test_engine_surface.py -q
"""
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from engine import (Action, HandHistoryLogger, PokerGame, get_action_mask,
                    get_raise_sizing_info)


@pytest.fixture
def game():
    """Four-handed, fixed seed, so every assertion below is reproducible."""
    return PokerGame([1000] * 4, small_blind=5, big_blind=10, seed=42)


@pytest.fixture
def logged_game():
    """
    The same game with history recording on.

    It is off by default because training turns it off for speed, which is
    exactly why it needs asking for explicitly here: a test that skips when the
    default is off covers the feature only in the build that does not use it.
    """
    return PokerGame([1000] * 4, small_blind=5, big_blind=10, seed=42,
                     enable_history=True)


# ---------------------------------------------------------------------------
# Raise sizing
# ---------------------------------------------------------------------------

def test_raise_sizing_is_bounded_and_ordered(game):
    """
    The hints an agent sizes a raise from. A maximum below the minimum, or a
    ratio outside [0, 1] once normalised, would push a network toward requesting
    raises the engine then rejects — which the audit found being silently
    converted to folds.
    """
    sizing = get_raise_sizing_info(game, game.state.current_player)

    assert sizing["can_raise"] in (0.0, 1.0), sizing
    if sizing["can_raise"]:
        assert sizing["min_raise_ratio"] <= sizing["max_raise_ratio"], sizing
        assert sizing["min_raise_ratio"] >= 0.0, sizing


def test_raise_sizing_agrees_with_the_action_mask(game):
    """
    Two descriptions of the same fact, from different functions. They must not
    disagree: an agent that reads "raising is legal" from one and gets no sizing
    from the other has no way to act on it.
    """
    player = game.state.current_player
    mask = get_action_mask(game, player)
    sizing = get_raise_sizing_info(game, player)

    assert len(mask) == 5, mask
    assert bool(mask[3]) == bool(sizing["can_raise"]), (mask, sizing)


# ---------------------------------------------------------------------------
# What the engine records
# ---------------------------------------------------------------------------

def test_action_history_records_what_was_played(logged_game):
    """
    History is disabled during training for speed, so the enabled path is the
    one that goes unexercised — and it is the path every hand log and post-hoc
    analysis reads from.
    """
    player = logged_game.state.current_player
    logged_game.apply_action(player, Action("call", 10))

    assert len(logged_game.action_history) == 1, logged_game.action_history
    recorded_player, action_type, _amount, street = logged_game.action_history[0]
    assert recorded_player == player
    assert action_type == "call"
    assert street == "preflop"


def test_history_is_reset_between_hands(logged_game):
    """A hand's history must describe that hand, not accumulate across them."""
    logged_game.apply_action(logged_game.state.current_player, Action("call", 10))
    assert len(logged_game.action_history) == 1

    logged_game.reset_hand()
    assert logged_game.action_history == [], logged_game.action_history


# ---------------------------------------------------------------------------
# Hand history logging
# ---------------------------------------------------------------------------

def test_the_hand_log_contains_the_hand_that_was_played(game, tmp_path):
    """
    The log is the human-readable record of a hand. Asserting it is non-empty
    would pass on a header alone, so this checks that an action actually
    reaches it.
    """
    logger = HandHistoryLogger(log_dir=str(tmp_path), table_name="TestTable")
    logger.start_hand(game, hand_id=1)

    player = game.state.current_player
    logger.log_hole_cards(player, game.players[player].hole_cards, hero=True)
    logger.log_action(player, Action("fold"), 0)
    game.apply_action(player, Action("fold"))

    history = logger.get_hand_history()

    assert "TestTable" in history, history
    assert "fold" in history.lower(), history
    assert str(player) in history, history


# ---------------------------------------------------------------------------
# The interactive CLI
# ---------------------------------------------------------------------------

def test_the_cli_plays_a_hand_to_completion(monkeypatch):
    """
    `engine/cli.py` is the only way a person drives the engine by hand, and
    nothing else imports it — so a crash here reaches a human before it reaches
    a test.

    Two details make this safe to run unattended. The fallback answer is
    ``fold``, because the CLI re-prompts on an illegal action and ``check`` is
    illegal when facing a bet — a fallback that can be rejected is an infinite
    loop. And the prompt budget turns any remaining loop into a failure with a
    message rather than a hung suite, which is how this test first behaved.
    """
    replies = iter(["call", "check", "check", "check"])
    prompts = []
    BUDGET = 40

    def answer(_prompt=""):
        prompts.append(_prompt)
        if len(prompts) > BUDGET:
            raise AssertionError(
                f"CLI asked for input {len(prompts)} times without finishing "
                f"the hand; last prompt {_prompt!r}")
        return next(replies, "fold")

    monkeypatch.setattr(sys, "argv",
                        ["cli.py", "--players", "2", "--stack", "100",
                         "--sb", "5", "--bb", "10", "--seed", "42"])

    captured = StringIO()
    with patch("builtins.input", side_effect=answer), \
            patch("sys.stdout", captured):
        import engine.cli
        engine.cli.main()

    output = captured.getvalue()
    assert output.strip(), "the CLI produced no output at all"
    assert "hand over" in output.lower(), output[-400:]
