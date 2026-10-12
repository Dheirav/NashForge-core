"""
A viewer that plays the hand the benchmark scores.

The previous controller reimplemented the betting loop: its own raise sizing,
its own action-name matching, no solver history string, no raise cap, and no
`finish_hand()`. So the game on screen was not the game `benchmark()` measures,
and the pot was never awarded — `is_hand_over()` means the betting finished,
not that the chips moved, which is the exact trap `_play_hand` documents. It
also loaded a genome from `hall_of_fame/`, which is empty and whose former
contents the audit withdrew.

This drives `evaluation.benchmark`'s own primitives instead — the same mask,
the same constraint onto the solver's tree, the same action conversion, the
same settle. The human occupies one seat; anything with the panel's agent
signature occupies the other.

Stacks reset to the benchmark's depth every hand
------------------------------------------------
Not an oversight, and not the same thing as forgetting the score. The solver
was fitted for one stack depth, and letting stacks drift across hands walks it
out of the tree it knows while the display goes on calling it the CFR agent.
`benchmark()` deals every hand from the same depth for that reason, so this
does too, and keeps the running total separately.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from abstraction.betting import CHECK_CALL
from engine import PokerGame, get_abstract_action_mask
# _constrain is private, and imported anyway: the alternative is a second copy
# of the tree-narrowing rules that silently stops matching the one the agent
# was measured under.
from evaluation.benchmark import (NUM_ACTIONS, RAISE_ACTIONS, _constrain,
                                  always_call_agent, cfr_agent, random_agent)
from training.fitness import abstract_action_to_engine_action, finish_hand

#: Matches `benchmark()`'s defaults. Changing these changes the game the agent
#: was fitted for, not merely its comfort.
STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2
RAISE_CAP = 1

STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "cfr",
                             "nolimit_strategy.pkl")

ACTION_LABELS = ["Fold", "Check/Call", "Raise ½ pot", "Raise pot",
                 "Raise 2× pot", "All-in"]


def _resolve_label(index, to_call):
    """
    `Check` and `Call` are one abstract action; say which one this is.

    The abstraction merges them because only one is ever legal -- `to_call` is
    either zero or it is not -- so the distinction lives in the situation rather
    than in the action space. On screen it has to be put back, or a person is
    asked to press `Check/Call` with nothing to call.
    """
    if index != CHECK_CALL:
        return ACTION_LABELS[index]
    return "Check" if to_call == 0 else f"Call {to_call}"


class AgentUnavailable(RuntimeError):
    """Raised with a reason a viewer can act on, rather than a stack trace."""


def load_cfr_opponent(rng, probe):
    """
    The one agent in this repository validated against something external.

    Kuhn's -1/18 and exact Leduc exploitability are what make it worth showing;
    the evolutionary genomes were bred on the withdrawn metric and the PPO
    policies are not measured yet.
    """
    path = os.path.abspath(STRATEGY_PATH)
    if not os.path.exists(path):
        # It is NOT a committed artifact: `*.pkl` is gitignored, so this file
        # exists only where it was trained. A clone does not have it and no
        # checkout will produce it -- it has to be retrained.
        raise AgentUnavailable(
            f"no solved strategy at {path}.\n"
            "It is gitignored (*.pkl), so a fresh clone will not have it. "
            "Retrain with:\n"
            "  venv/bin/python scripts/cfr/train_nolimit.py --iterations 150000 "
            "--output results/cfr/nolimit_strategy.pkl\n"
            "or play with --opponent random.")
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    return cfr_agent(saved["strategy"], saved["abstraction"], rng,
                     raise_cap=RAISE_CAP, probe=probe)


class GameController:
    """
    One seat is a person, the other is an agent, and the loop between them is
    the benchmark's.
    """

    def __init__(self, opponent="cfr", human_seat=0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.human_seat = human_seat
        self.agent_seat = 1 - human_seat

        #: Filled by the agent with the distribution it actually sampled, so
        #: the panel shows the policy rather than a re-derivation of it.
        self.policy_probe = [None]
        self.opponent_name = opponent
        self.agent = self._build_opponent(opponent)

        self.hands_played = 0
        self.total_chips = 0          # to the human, across the session
        self.last_result = None       # chips won on the hand just finished
        self.hand_over = False
        self.awaiting_human = False
        self.mask = None
        self.to_call = 0
        self.agent_policy = None      # distribution shown in the side panel
        self.agent_last_action = None
        #: What the agent faced when it acted, which is not what the person
        #: faces now. `to_call` moves on to the next actor; the panel is still
        #: showing the agent's node, so it needs its own copy to label from.
        self.agent_to_call = 0
        self.message = None

        self._start_hand()

    # -- setup ----------------------------------------------------------

    def _build_opponent(self, name):
        if name == "cfr":
            return load_cfr_opponent(self.rng, self.policy_probe)
        if name == "random":
            return random_agent(self.rng)
        if name == "always-call":
            return always_call_agent()
        raise AgentUnavailable(f"unknown opponent {name!r}; "
                               "expected cfr, random or always-call")

    def _start_hand(self):
        """A fresh deal at the benchmark's depth. See the module docstring."""
        self.game = PokerGame([STARTING_STACK, STARTING_STACK],
                              small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
                              seed=int(self.rng.integers(0, 2 ** 31 - 1)),
                              enable_history=False)
        self.history = ""
        self.street = self.game.state.betting_round
        self.raises_this_street = 0
        self.hand_over = False
        self.awaiting_human = False
        self.agent_policy = None
        self.agent_last_action = None
        self.message = None
        self.guard = 0

    def next_hand(self):
        """Deal again. Only meaningful once the current hand has been paid."""
        if self.hand_over:
            self._start_hand()

    # -- the loop -------------------------------------------------------

    def update(self):
        """
        Advance until the game needs the person, or the hand ends.

        Mirrors `_play_hand`. The one difference is that it returns to the
        caller instead of calling the human seat, because a person is not a
        function that answers immediately.
        """
        if self.hand_over or self.awaiting_human:
            return

        while not self.game.is_hand_over():
            self.guard += 1
            if self.guard > 200:
                self.message = "hand did not terminate"
                self._settle()
                return

            player = self.game.state.current_player
            if player is None:
                break

            if self.game.state.betting_round != self.street:
                self.street = self.game.state.betting_round
                self.history += "/"
                self.raises_this_street = 0

            actor = self.game.players[player]
            self.to_call = self.game.current_bet - actor.bet
            self.mask = _constrain(get_abstract_action_mask(self.game, player),
                                   self.to_call, self.raises_this_street,
                                   RAISE_CAP, street=self.history.count("/"))

            if player == self.human_seat:
                self.awaiting_human = True
                return

            self.policy_probe[:] = [None]
            choice = self.agent(self.game, player, self.mask, self.history)
            self.agent_policy = self.policy_probe[0]
            self.agent_last_action = choice
            self.agent_to_call = self.to_call
            self._commit(player, choice)

        self._settle()

    def choose(self, action_index):
        """The person's move. Ignored unless it is their turn and it is legal."""
        if not self.awaiting_human or self.mask is None:
            return False
        if not self.mask[action_index]:
            return False
        self.awaiting_human = False
        self._commit(self.human_seat, action_index)
        self.update()
        return True

    def _commit(self, player, choice):
        """
        Apply one abstract action, keeping the solver's key in step with it.

        The history string is what the CFR agent looks itself up by, so it has
        to be appended in the same place `_play_hand` appends it — before the
        engine moves, and for both seats, not just the agent's.
        """
        mask = self.mask
        assert mask is not None, "_commit outside a decision point"
        if not mask[choice]:
            choice = int(np.flatnonzero(mask)[0])
        self.history += str(choice)
        if choice in RAISE_ACTIONS:
            self.raises_this_street += 1
        self.game.apply_action(
            player, abstract_action_to_engine_action(choice, self.game, player))

    def _settle(self):
        """
        Award the pot and score the hand.

        `finish_hand` before reading stacks, for the reason `_play_hand` gives:
        at showdown the chips are still in the middle, and reading around it
        scores every hand as a loss of what was contributed.
        """
        finish_hand(self.game)
        self.last_result = self.game.players[self.human_seat].stack - STARTING_STACK
        self.total_chips += self.last_result
        self.hands_played += 1
        self.hand_over = True
        self.awaiting_human = False
        self.mask = None

    # -- what the renderer asks for --------------------------------------

    @property
    def bb_per_100(self):
        """The session in the units every result in this project is quoted in."""
        if self.hands_played == 0:
            return None
        return 100.0 * self.total_chips / (self.hands_played * BIG_BLIND)

    def legal_actions(self):
        """Six flags, in `ACTION_LABELS` order, or all-false when not to act."""
        if self.mask is None or not self.awaiting_human:
            return [False] * NUM_ACTIONS
        return [bool(self.mask[i]) for i in range(NUM_ACTIONS)]

    def action_label(self, index):
        """The person's buttons. `Check` and `Call` are one abstract action."""
        return _resolve_label(index, self.to_call)

    def policy_label(self, index):
        """
        The same names for the agent's policy panel, from the agent's node.

        Reading `to_call` here instead would label the panel with whatever the
        person is facing now, so the row could read `Call 4` beside a
        distribution the agent produced while checking.
        """
        return _resolve_label(index, self.agent_to_call)

    def showdown_visible(self):
        """The agent's cards, once the hand is paid and not before."""
        return self.hand_over
