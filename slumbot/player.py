"""
This project's solver, playing a hand dealt by Slumbot.

The bridge turns Slumbot's betting string into the solver's own history key and
chip accounting; this puts the two together and produces an action. It reuses
`evaluation.benchmark.cfr_agent` rather than reimplementing the lookup, through a
shim that presents the four things that agent actually reads. A second copy of
the lookup is exactly how the agent measured here would drift from the agent
measured on the panel, and then the two numbers would not be about the same
thing.

What this is honestly measuring
-------------------------------
A 100bb, one-raise-per-street strategy playing a 200bb unlimited-raise opponent.
That is not the solver's strength at Slumbot's game and should never be quoted as
though it were. It is the first number in this project that somebody else's agent
produced, which is the whole of what milestone M1 claims.

Two counters come back with every session for that reason: `misses`, the fraction
of decisions where the strategy had no entry and the agent chose among legal
actions at random, and `off_abstraction`, bets too large for the abstraction to
describe. A high miss rate means the result is largely a measurement of a random
agent wearing the solver's name -- the failure this project has already published
once, at a 74.3% miss rate.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from cfr.flat import load_strategy
from slumbot._policy import fallback_choice, strength_class
from evaluation.benchmark import cfr_agent
from slumbot.api import HandState
from slumbot.bridge import legal_mask, parse_cards, replay, to_slumbot
from abstraction.betting import schedule_from_args  # noqa: E402


@dataclass
class SessionStats:
    """Protocol health, kept apart from anything about winning."""
    hands: int = 0
    decisions: int = 0
    misses: int = 0
    consulted: int = 0
    off_abstraction: int = 0
    actions_sent: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    #: Lookup misses by how many raises were already on the street, so a run
    #: says where the misses live and not only how many there are. That
    #: histogram is what chooses the next schedule, if there is one.
    miss_depths: Dict[int, int] = field(default_factory=dict)
    #: Misses answered by the hand-strength rule rather than a random guess.
    fallbacks: int = 0
    #: Schedule-level misses from the bridge: a re-raise the schedule had no
    #: size for, by depth.
    schedule_misses: Dict[int, int] = field(default_factory=dict)

    @property
    def miss_rate(self) -> float:
        return self.misses / self.consulted if self.consulted else 0.0


def _shim(hole, board, to_call: int):
    """
    The four things `cfr_agent` reads, and nothing else.

    Deliberately not a `PokerGame`: building one would mean dealing a deck to
    match cards Slumbot has already dealt, and a mismatch between the two would
    be a new and entirely self-inflicted class of bug.
    """
    actor = SimpleNamespace(hole_cards=hole, bet=0)
    return SimpleNamespace(
        players=[actor, SimpleNamespace(hole_cards=[], bet=0)],
        state=SimpleNamespace(community_cards=board),
        current_bet=to_call)


def strategy_schedule(saved: dict):
    """The raise schedule a pickle was trained on: an int cap or a taper tuple."""
    args = saved.get("args") or {}
    if not isinstance(args, dict):
        args = vars(args)
    cap = args.get("raise_cap", 1)
    return schedule_from_args(cap)


class SolverPlayer:
    """Answers Slumbot's hands with the strategy in `results/cfr/`."""

    def __init__(self, strategy_path: str, rng: Optional[np.random.Generator] = None,
                 raise_cap=None, purify: str = "none", fallback: bool = True):
        saved = load_strategy(strategy_path)
        self.abstraction = saved["abstraction"]
        self.rng = rng if rng is not None else np.random.default_rng(0)
        if raise_cap is None:
            # The schedule the pickle was trained on, the way `chipzen.player`
            # reads it. Defaulting to one raise measured every deeper solver as
            # a one-raise one, which is how the contender plan's step 1 opened.
            raise_cap = strategy_schedule(saved)
        self.raise_cap = raise_cap
        self.stats = SessionStats()
        #: The miss counter is the agent's own, so what is reported is what the
        #: agent experienced rather than a second count taken alongside it.
        self._misses = [0, 0]
        self.purify = purify
        #: On a lookup miss, the arena's hand-strength rule rather than the
        #: panel agent's uniform guess. The guess is right for a measurement
        #: of the solver alone and wrong for a match: see `fallback_choice`.
        self.fallback = fallback
        self._agent = cfr_agent(saved["strategy"], self.abstraction, self.rng,
                                misses=self._misses, raise_cap=raise_cap, purify=purify)

    def begin_hand(self) -> None:
        """Snapshot the counters, so `hand_record` can say what this hand cost."""
        self._hand_start = (self.stats.decisions, self._misses[0],
                            sum(self.stats.schedule_misses.values()), self.stats.off_abstraction)

    def hand_record(self, state: HandState) -> dict:
        """
        One finished hand, compactly: what it won, where it ended, and whether
        the solver was ever without an entry in it.

        This is the record `scripts/slumbot_split.py` reads to split the loss by
        miss against hit, by final street and by position. The whole-run miss
        rate says how often the agent guessed; only a per-hand record can say
        whether the guesses are where the chips went.
        """
        start = getattr(self, "_hand_start", (0, 0, 0, 0))
        misses = self._misses[0] - start[1]
        return {
            "w": state.winnings, "pos": state.client_pos,
            "street": state.action.count("/"),          # 0 preflop to 3 river
            # Slumbot reveals its cards after a fold too, so the cards do not
            # say whether there was a showdown; the last action does.
            "showdown": not state.action.rstrip("/").endswith("f"),
            "decisions": self.stats.decisions - start[0],
            "miss": misses,
            "sched_miss": sum(self.stats.schedule_misses.values()) - start[2],
            "off": self.stats.off_abstraction - start[3],
        }

    def __call__(self, state: HandState) -> str:
        node = replay(state, self.rng, schedule=self.raise_cap)
        self.stats.off_abstraction += node.misses
        for depth in node.miss_depths:
            self.stats.schedule_misses[depth] = self.stats.schedule_misses.get(depth, 0) + 1

        hole = parse_cards(state.hole_cards)
        board = parse_cards(state.board)
        mask = legal_mask(node, self.raise_cap)

        missed_before = self._misses[0]
        choice = self._agent(_shim(hole, board, node.to_call), 0, mask,
                             node.history)
        if self._misses[0] > missed_before:
            depth = node.raises_this_street
            self.stats.miss_depths[depth] = self.stats.miss_depths.get(depth, 0) + 1
            if self.fallback:
                top = int(getattr(self.abstraction, "postflop_buckets", 6)) - 1
                choice = fallback_choice(strength_class(self.abstraction, hole, board), top,
                                         mask, node.pot, node.to_call)
                self.stats.fallbacks += 1
        if not mask[choice]:
            # The agent is asked only for legal actions, but a mask this layer
            # built and an agent the panel built could disagree; sending an
            # illegal action would be rejected by the server and lose the hand
            # to a protocol error rather than to poker.
            choice = int(np.flatnonzero(mask)[0])

        outgoing = to_slumbot(choice, node)
        self.stats.decisions += 1
        self.stats.actions_sent[outgoing[0]] = \
            self.stats.actions_sent.get(outgoing[0], 0) + 1
        self.stats.misses, self.stats.consulted = self._misses
        return outgoing
