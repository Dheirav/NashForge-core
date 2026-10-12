"""
The viewer's loop, checked against the loop it is supposed to be.

Two things were wrong before, and neither would have shown up as an exception.
The pot was never awarded, because the old controller dealt again on
`is_hand_over()` — which means the betting stopped, not that the chips moved —
so a session's chip counts were quietly meaningless. And no solver history
string was kept, so the CFR agent missed every lookup and played as a second
random opponent under its own name.

Both are silent, so both are asserted here rather than left to be noticed.
"""
import os
import pickle

import numpy as np
import pytest

from evaluation.benchmark import cfr_agent
from abstraction.betting import CHECK_CALL
from gui.game_controller import (STARTING_STACK, GameController, NUM_ACTIONS,
                                 _resolve_label)

STRATEGY = os.path.join(os.path.dirname(__file__), "..", "results", "cfr",
                        "nolimit_strategy.pkl")


def _play(controller, hands, on_hand=None):
    """Drive a session with a scripted person who checks and calls."""
    played, guard = 0, 0
    while played < hands:
        guard += 1
        assert guard < 20000, "session did not progress"
        controller.update()
        if controller.awaiting_human:
            legal = [i for i in range(NUM_ACTIONS) if controller.legal_actions()[i]]
            assert legal, "asked to act with no legal action"
            controller.choose(legal[min(1, len(legal) - 1)])
        elif controller.hand_over:
            if on_hand is not None:
                on_hand(controller)
            played += 1
            controller.next_hand()
    return controller


def test_the_pot_is_awarded_every_hand():
    """
    Chips are conserved once the hand is over.

    This is the assertion the old viewer would have failed: it read stacks
    without settling, so the chips sat in the middle and every hand scored as a
    loss of what had been contributed.
    """
    def check(controller):
        total = sum(player.stack for player in controller.game.players)
        assert total == 2 * STARTING_STACK, (
            f"chips not conserved after the hand: {total} != {2 * STARTING_STACK}")

    _play(GameController(opponent="random", seed=5), hands=30, on_hand=check)


def test_the_session_total_is_the_sum_of_its_hands():
    controller = GameController(opponent="random", seed=6)
    seen = []
    _play(controller, hands=20, on_hand=lambda c: seen.append(c.last_result))
    assert len(seen) == 20
    assert controller.hands_played == 20
    assert controller.total_chips == sum(seen)


@pytest.mark.skipif(not os.path.exists(STRATEGY),
                    reason="no solved strategy in results/cfr")
def test_the_cfr_agent_finds_its_strategy_in_this_loop():
    """
    The lookup miss rate is the test of the history string.

    The solver keys off `bucket|history`. If the viewer builds that string
    differently from `_play_hand` — a missing street separator, an unappended
    human action — every lookup misses and the agent falls back to choosing at
    random while still being labelled CFR. A miss rate near zero is what says
    the two loops agree.
    """
    with open(STRATEGY, "rb") as handle:
        saved = pickle.load(handle)

    misses = [0, 0]
    controller = GameController(opponent="random", seed=7)
    controller.agent = cfr_agent(saved["strategy"], saved["abstraction"],
                                 np.random.default_rng(0), misses=misses,
                                 raise_cap=1, probe=controller.policy_probe)
    _play(controller, hands=40)

    consulted = misses[1]
    assert consulted > 0, "the agent was never asked to act"
    assert misses[0] / consulted < 0.02, (
        f"the CFR agent missed {misses[0]}/{consulted} lookups; the viewer's "
        "history string does not match the one the strategy was keyed by")


@pytest.mark.skipif(not os.path.exists(STRATEGY),
                    reason="no solved strategy in results/cfr")
def test_the_shown_policy_is_a_distribution_over_legal_actions():
    """What the panel displays has to be what the agent sampled from."""
    controller = GameController(opponent="cfr", seed=8)
    checked = 0
    for _ in range(4000):
        controller.update()
        policy = controller.agent_policy
        if policy is not None:
            assert policy.shape == (NUM_ACTIONS,)
            assert np.isclose(policy.sum(), 1.0), "policy is not normalised"
            assert (policy >= 0).all()
            chosen = controller.agent_last_action
            assert policy[chosen] > 0, "sampled an action the policy gave no weight"
            checked += 1
        if controller.awaiting_human:
            legal = [i for i in range(NUM_ACTIONS) if controller.legal_actions()[i]]
            controller.choose(legal[min(1, len(legal) - 1)])
        elif controller.hand_over:
            controller.next_hand()
        if checked > 40:
            break
    assert checked > 0, "the agent never exposed a policy"


def test_the_agents_cards_stay_hidden_until_the_pot_is_paid():
    controller = GameController(opponent="random", seed=9)
    for _ in range(3000):
        controller.update()
        assert controller.showdown_visible() == controller.hand_over
        if controller.awaiting_human:
            legal = [i for i in range(NUM_ACTIONS) if controller.legal_actions()[i]]
            controller.choose(legal[min(1, len(legal) - 1)])
        elif controller.hand_over:
            controller.next_hand()


def test_an_illegal_action_is_refused_rather_than_played():
    """
    The buttons already grey out; this is the layer beneath them.

    Keyboard shortcuts reach `choose` without passing a button, so the mask has
    to be enforced here and not only in the drawing code.
    """
    controller = GameController(opponent="random", seed=10)
    for _ in range(500):
        controller.update()
        if controller.awaiting_human:
            break
        if controller.hand_over:
            controller.next_hand()
    assert controller.awaiting_human

    legal = controller.legal_actions()
    illegal = [i for i in range(NUM_ACTIONS) if not legal[i]]
    if illegal:
        before = controller.history
        assert controller.choose(illegal[0]) is False
        assert controller.history == before, "an illegal action changed the game"
        assert controller.awaiting_human, "an illegal action passed the turn"


def test_the_merged_action_is_named_for_the_situation():
    """
    `Check` and `Call` are one abstract action, and the screen has to say which.

    The abstraction merges them because only one is ever legal, so the
    distinction lives in `to_call` rather than in the action space. Left merged
    on screen, a person is asked to press `Check/Call` with nothing to call.
    """
    assert _resolve_label(CHECK_CALL, 0) == "Check"
    assert _resolve_label(CHECK_CALL, 4) == "Call 4"
    # Every other row is the abstraction's own name, unchanged.
    for index in range(NUM_ACTIONS):
        if index != CHECK_CALL:
            assert _resolve_label(index, 0) == _resolve_label(index, 4)


def test_the_policy_panel_is_labelled_from_the_agents_node():
    """
    Not from `to_call`, which by then belongs to whoever acts next.

    The panel shows the distribution the agent produced at the node it acted on.
    Labelling it from the live `to_call` would print `Call 4` beside a
    distribution the agent produced while checking, which is the same class of
    error as reading a strategy against the wrong game.
    """
    controller = GameController(opponent="random", seed=10)
    for _ in range(500):
        controller.update()
        if controller.awaiting_human and controller.agent_last_action is not None:
            break
        if controller.hand_over:
            controller.next_hand()

    if controller.agent_last_action is None:
        pytest.skip("the agent did not act before the person was asked to")

    controller.to_call = controller.agent_to_call + 7   # the next actor's, not the agent's
    assert controller.policy_label(CHECK_CALL) == _resolve_label(
        CHECK_CALL, controller.agent_to_call)
    assert controller.action_label(CHECK_CALL) == _resolve_label(
        CHECK_CALL, controller.to_call)
