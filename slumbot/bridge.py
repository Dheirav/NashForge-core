"""
Slumbot's hand, expressed as something this project's solver can answer.

The transport is `slumbot/api.py`; this is the layer between it and an agent.
`docs/EXTERNAL_BENCHMARK.md` is explicit that this is most of the engineering and
that a bug in it is indistinguishable from a weak strategy, so it is a module
with its own tests and no network in it.

What it has to bridge
---------------------
Slumbot bets any legal amount. This project plays six abstract actions, and its
solver was fitted over them. So an incoming bet has to be read *as* one of the
six, and an outgoing abstract action has to become a chip amount.

The inbound mapping is not rounding. `abstraction/translation.py` already
implements the pseudo-harmonic mapping of Ganzfried & Sandholm (2013), which
sends a bet probabilistically to its two neighbouring sizes: a deterministic
nearest-size rule has a boundary, and a boundary is a thing an opponent sits just
inside of. That module is used rather than reimplemented.

The two mismatches this cannot fix, and does not pretend to
-----------------------------------------------------------
**Depth.** Slumbot plays 200 big blinds; the solver was fitted for 100. Bets are
translated as fractions of the pot, which insulates the middle sizes, but not
all-in: at 200bb it is twice as far away as anything the strategy was solved for.

**Raises per street.** The solver knows one. Slumbot re-raises. A re-raise puts
the node off-tree, the lookup misses, and `cfr_agent` falls back to choosing
among legal actions at random.

Both are properties of playing a 100bb one-raise strategy against a 200bb
unlimited-raise opponent, and both are reported rather than hidden: the miss rate
comes back with every result, for the same reason the panel reports it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, raise_sizes_at
from abstraction.translation import translate
from engine.cards import Card
from slumbot.api import BIG_BLIND, SMALL_BLIND, STARTING_STACK, HandState

#: The raise sizes the solver knows, as fractions of the pot. Indices 2..5 of the
#: six abstract actions; all-in has no pot fraction and is handled separately.
RAISE_FRACTIONS = (0.5, 1.0, 2.0)

TOKEN = re.compile(r"b(\d+)|([ckf])")


class TranslationError(RuntimeError):
    """A hand that could not be represented. Raised rather than guessed at."""


def parse_card(text: str) -> Card:
    """``"Th"`` to a Card. Slumbot's rank and suit letters are already ours."""
    if len(text) != 2:
        raise TranslationError(f"unreadable card {text!r}")
    return Card(text[0].upper(), text[1].lower())


def parse_cards(items) -> List[Card]:
    return [parse_card(c) for c in items]


@dataclass
class Node:
    """Where a hand stands, in this project's terms rather than Slumbot's."""
    history: str = ""                 # the solver's key: abstract actions, "/" per street
    pot: int = 0                      # chips in the middle, both players
    committed: List[int] = field(default_factory=lambda: [0, 0])   # this street
    #: Chips each player put in on *completed* streets. Slumbot's bet levels are
    #: per street and reset at each one, so a player's ceiling on this street is
    #: the stack less what earlier streets already took -- capping at the full
    #: stack instead asks for more chips than exist and the server rejects it.
    prior: List[int] = field(default_factory=lambda: [0, 0])
    street: int = 0
    raises_this_street: int = 0
    to_act: int = 0
    misses: int = 0                   # bets that fell outside the abstraction entirely
    #: Raise depths at which the schedule had no sized raise to read a bet as.
    #: Under a one-raise cap every re-raise lands here; it is the count the
    #: contender plan gates a deeper tree on, so it is kept by depth.
    miss_depths: List[int] = field(default_factory=list)

    @property
    def to_call(self) -> int:
        return abs(self.committed[0] - self.committed[1])


def _blinds(client_pos: int) -> Tuple[int, int]:
    """
    Who has what in before a card is dealt.

    Heads-up, the button posts the small blind and acts first preflop. Slumbot
    opens the betting in every observed hand, so the client is the big blind and
    the amounts below follow. Returned as (client, bot) so the caller never has
    to remember which way round `client_pos` runs.
    """
    return (BIG_BLIND, SMALL_BLIND) if client_pos == 0 else (SMALL_BLIND, BIG_BLIND)


def replay(state: HandState, rng: np.random.Generator, schedule=1) -> Node:
    """
    Walk Slumbot's betting string, translating each bet into an abstract action.

    The result is the history key the solver is looked up by, plus enough chip
    accounting to price the next decision. Bets arrive as levels bet *to*, so the
    increment is a difference; the blinds are committed before the string starts
    and are not in it.
    """
    client, bot = _blinds(state.client_pos)
    node = Node(pot=client + bot, committed=[client, bot])

    # Index 0 is always the client, so who moves first is a question about the
    # button. Heads-up the button posts the small blind and acts first preflop,
    # and the big blind acts first on every street after it -- the order reverses
    # exactly once. Measured rather than assumed: folding immediately loses 100
    # at client_pos 0 and 50 at client_pos 1, so the client is the big blind in
    # the first case and the button in the second.
    button = 1 if state.client_pos == 0 else 0
    actor = button

    for chunk in state.action.split("/"):
        if node.street > 0:
            node.prior = [node.prior[i] + node.committed[i] for i in (0, 1)]
            node.committed = [0, 0]
            node.raises_this_street = 0
            node.history += "/"
            actor = 1 - button           # the big blind leads after the flop
        for match in TOKEN.finditer(chunk):
            amount, simple = match.group(1), match.group(2)
            if simple == "f":
                node.history += str(FOLD)
            elif simple in ("c", "k"):
                owed = node.to_call
                node.pot += owed
                node.committed[actor] += owed
                node.history += str(CHECK_CALL)
            else:
                level = int(amount)
                increment = level - node.committed[actor]
                # The pot the bet was sized against is what was there before it.
                fraction = increment / node.pot if node.pot else 0.0
                node.history += str(_as_abstract(fraction, level, node, schedule, rng))
                node.pot += increment
                node.committed[actor] = level
                node.raises_this_street += 1
            actor = 1 - actor
        node.street += 1

    node.street -= 1               # the loop counts the last street it entered
    # Always 0: `committed` and `prior` are built client-first, and the decision
    # being priced is always the client's. Indexing by `client_pos` here read the
    # bot's chips for half the hands.
    node.to_act = 0
    return node


def _as_abstract(fraction: float, level: int, node: Node, schedule,
                 rng: np.random.Generator) -> int:
    """
    Which of the six the solver should think it faced.

    An all-in is its own action rather than a very large raise: the solver has a
    slot for it, and calling a 200bb shove "two times pot" would ask the strategy
    a question about a bet it could fold to.

    Translation is over the sizes the schedule has at *this* raise depth, as
    `chipzen.bridge` does. Translating over all three at every depth read a
    re-raise of a third of the pot as "half pot" under a `(4, 2)` taper, a code
    the solver never stored there, so it missed for the wrong reason and the
    miss rate could not say what a deeper tree would buy.
    """
    if level >= STARTING_STACK:
        return ALL_IN
    sizes = raise_sizes_at(schedule, node.raises_this_street, node.street)
    fractions = [RAISE_FRACTIONS[a - 2] for a in sizes if a != ALL_IN]
    if not fractions:
        # No sized raise at this depth: only all-in, or nothing at all. Either
        # way the solver has no sized entry to be asked, and this is the count
        # a deeper schedule would change.
        node.miss_depths.append(node.raises_this_street)
        return ALL_IN
    if fraction >= fractions[-1] * 1.5:
        # Beyond the largest size the abstraction carries and short of a shove.
        # Counted, because a bet this project cannot describe is exactly the
        # thing that should show up as a number rather than as a shrug.
        node.misses += 1
        return ALL_IN
    chosen = translate(fractions, max(fraction, fractions[0]), rng)
    return [a for a in sizes if a != ALL_IN][chosen]


def max_level(node: Node, stack: int = STARTING_STACK) -> int:
    """
    The largest level this player can bet *to* on the current street.

    Levels are per street and reset at each one, so the ceiling is the stack less
    whatever earlier streets already took. Using the full stack asks for chips the
    player no longer has, and the server answers "Bet size too big" -- which costs
    the hand to a protocol error rather than to poker.
    """
    return stack - node.prior[node.to_act]


def to_slumbot(action: int, node: Node, stack: int = STARTING_STACK) -> str:
    """
    An abstract action, as something the server will accept.

    Amounts go out as levels bet *to*, matching the notation coming in. A raise
    is sized off the pot *after* the call that precedes it, which is how the
    abstraction defines its fractions -- sizing off the pot before under-bets
    every raise, and reads as a strategy playing timidly.
    """
    if action == FOLD:
        return "f"
    if action == CHECK_CALL:
        return "c" if node.to_call else "k"

    ceiling = max_level(node, stack)
    if action == ALL_IN:
        return f"b{ceiling}"

    call = node.to_call
    pot_after_call = node.pot + call
    fraction = RAISE_FRACTIONS[action - 2]
    level = node.committed[node.to_act] + call + int(round(pot_after_call * fraction))
    return f"b{min(level, ceiling)}"


def legal_mask(node: Node, raise_cap: int = 1) -> np.ndarray:
    """
    The six actions, narrowed to the tree the solver was trained on.

    Mirrors `evaluation.benchmark._constrain` rather than inventing a second set
    of rules: folding with nothing to call is dominated and off-tree, and raises
    stop at the cap. Both remove options, so nothing illegal is ever offered.
    """
    mask = np.ones(6, dtype=np.float64)
    if node.to_call <= 0:
        mask[FOLD] = 0.0
    allowed = raise_sizes_at(raise_cap, node.raises_this_street, node.street)
    for action in (2, 3, 4, ALL_IN):
        if action not in allowed:
            mask[action] = 0.0
    if not mask.any():
        mask[CHECK_CALL] = 1.0
    return mask
