"""
Chipzen's hand, expressed as something this project's solver can answer.

The arena sends a structured `turn_request`: hole cards and board as strings,
the pot, both stacks, what it costs to call, the legal raise range, and every
action so far as `{seat, action, amount, phase}` with `amount` the **total bet
level on that street** after the action. That is the same "bet to a level"
notation Slumbot uses, so this module is `slumbot/bridge.py` with the string
parser replaced by a walk over a list, and it reuses that module's `Node` and
card parsing rather than growing a second copy of either.

Two things are done differently from the Slumbot bridge, both on purpose.

**Raise fractions are measured the way the engine defines them.** The solver's
raise sizes are fractions of the pot *after the call*; the Slumbot bridge
measured an incoming raise against the pot before it, which agrees for an
opening bet and disagrees for a re-raise. Under a one-raise tree a re-raise was
off-tree anyway, so it never mattered. Under a taper it does.

**Translation respects the raise schedule.** A re-raise under `(4, 2)` may only
be two times pot or all-in, so an incoming re-raise is translated onto those two
and not onto all three sizes. Translating onto a size the solver never had at
that depth produces a key the solver never stored, and the miss would then be
counted as a depth problem when it was a bridge bug.

Stacks and blinds are whatever the arena says they are. Matches start at 10,000
chips with blinds 50 and 100, the blinds rise on a schedule, and the stacks carry
over from hand to hand until one bot has everything. Nothing here assumes a
depth; the player picks the solver from the effective stack in big blinds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, raise_sizes_at
from abstraction.translation import translate
from engine.cards import Card
from slumbot.bridge import Node, RAISE_FRACTIONS, TranslationError, parse_cards

#: The arena's four player actions. Blind and ante posts are synthetic entries
#: that appear in the history but never in `valid_actions`.
BLIND_POSTS = ("post_small_blind", "post_big_blind")
ANTE_POST = "post_ante"
PHASES = ("preflop", "flop", "turn", "river")


@dataclass
class Hand:
    """One decision's worth of context: the node plus what the node cannot hold."""
    node: Node
    seat: int                              # which seat is ours
    big_blind: int
    #: Chips each seat had before this hand's blinds, by seat. What "all-in"
    #: means for a bet is judged against these, not against a constant.
    start_stacks: List[int] = field(default_factory=lambda: [0, 0])
    #: Per-street raise depth at which each miss occurred. The histogram is
    #: what says whether a deeper tree would have helped.
    miss_depths: List[int] = field(default_factory=list)
    #: Positions in `node.history` of raises read as all-in although the
    #: bettor kept chips (a size the schedule cannot name at that depth), and
    #: the depth each was made at. When such a street closes and the hand
    #: goes on, `_close_street` re-reads it; see there.
    pseudo_allins: List[Tuple[int, int]] = field(default_factory=list)
    #: How many streets `_close_street` re-read this hand.
    collapsed: int = 0
    #: The history with those streets re-read, or None when nothing was. The
    #: true history stays the key; this one is asked only after the primary
    #: and its companion have both missed on the true one, since a companion
    #: may hold the real node (a one-raise primary's cap-2 companion does).
    alt_history: str = None
    _edits: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def effective_bb(self) -> float:
        """The shorter stack in big blinds, which is the depth the hand is played at."""
        return min(self.start_stacks) / self.big_blind if self.big_blind else 0.0


def _as_abstract(fraction: float, level: int, ceiling: int, node: Node,
                 schedule, hand: Hand, rng: np.random.Generator) -> int:
    """
    Which of the solver's raises an incoming bet is read as.

    All-in is its own action: a shove is a bet to everything the bettor has, and
    reading it as "two times pot" would ask the strategy about a bet it could
    fold to. A bet beyond the largest size the abstraction carries is read as a
    shove too and counted, because that is a bet this project cannot describe.
    """
    sizes = raise_sizes_at(schedule, node.raises_this_street, node.history.count("/"))
    if level >= ceiling:
        return ALL_IN
    fractions = [RAISE_FRACTIONS[a - 2] for a in sizes if a != ALL_IN]
    if not fractions:
        # The schedule has no sized raise at this depth, only all-in or
        # nothing. Either way the solver has no sized entry to be asked.
        hand.miss_depths.append(node.raises_this_street)
        hand.pseudo_allins.append((len(node.history), node.raises_this_street))
        return ALL_IN
    if fraction >= fractions[-1] * 1.5:
        node.misses += 1
        hand.pseudo_allins.append((len(node.history), node.raises_this_street))
        return ALL_IN
    chosen = translate(fractions, max(fraction, fractions[0]), rng)
    return [a for a in sizes if a != ALL_IN][chosen]


def _close_street(hand: Hand, schedule) -> None:
    """
    Re-read a street that closed on the call of a pseudo all-in.

    A raise the schedule cannot name at its depth is read as all-in, which is
    right for the decision it poses (fold or call) but says the hand ends at
    the call. When the bettor kept chips the arena deals on, and every later
    decision would key a history the tree treats as terminal: in one burst
    of rated matches, three deep-stack hands at about 1,600 chips each,
    every one a preflop third raise under the (4, 2, 1) taper followed by a
    flop. So once such a street has closed with a call, the pseudo all-in is
    re-read as the largest sized raise the schedule has at that depth, and
    where it has none the raise and the call collapse into a single call.
    Amounts are computed from the real pot either way, so only the strategy
    lookup moves, which is what translation does everywhere else. The re-read
    history is kept beside the true one (`Hand.alt_history`), not in its
    place: see that field for why.
    """
    node = hand.node
    if not hand.pseudo_allins:
        return
    street_start = node.history.rfind("/") + 1
    street = node.history[street_start:]
    if not street.endswith(str(ALL_IN) + str(CHECK_CALL)):
        hand.pseudo_allins = [p for p in hand.pseudo_allins if p[0] < street_start]
        return
    position = len(node.history) - 2
    depths = [d for (pos, d) in hand.pseudo_allins if pos == position]
    if not depths:
        return
    sized = [a for a in raise_sizes_at(schedule, depths[0], node.history[:street_start].count("/")) if a != ALL_IN]
    hand._edits.append((position, (str(sized[-1]) if sized else "") + str(CHECK_CALL)))
    hand.pseudo_allins = [p for p in hand.pseudo_allins if p[0] < street_start]
    hand.collapsed += 1


def replay(state: dict, seat: int, rng: np.random.Generator,
           schedule=1) -> Hand:
    """
    Walk the arena's action history into the solver's history key.

    Two passes. The first only moves chips, to learn what each seat started the
    hand with (its stack now plus everything it has put in), because whether a
    bet is a shove depends on that and the history has to be read before the
    stacks at the time can be known. The second builds the key.
    """
    history = state.get("action_history") or []
    opponent = 1 - seat
    now = [0, 0]
    now[seat] = int(state.get("your_stack") or 0)
    now[opponent] = int((state.get("opponent_stacks") or [0])[0])

    contributed = _contributions(history)
    start = [now[s] + contributed[s] for s in (0, 1)]

    big_blind = 0
    small_blind = 0
    for entry in history:
        if entry.get("action") == "post_big_blind":
            big_blind = int(entry.get("amount") or 0)
        elif entry.get("action") == "post_small_blind":
            small_blind = int(entry.get("amount") or 0)
    # A short post (a player with less than a blind) is not the level: 35
    # logged hands read as 1bb deep at any true depth. The small blind's post
    # pins it, except when both are short, where nothing does.
    big_blind = max(big_blind, 2 * small_blind)
    hand = Hand(node=Node(), seat=seat, big_blind=big_blind, start_stacks=start)
    node = hand.node

    phase = "preflop"
    for entry in history:
        actor = int(entry["seat"])
        action = entry["action"]
        amount = int(entry.get("amount") or 0)

        if action in BLIND_POSTS:
            node.pot += amount
            node.committed[actor] += amount
            continue
        if action == ANTE_POST:
            # Dead money: in the pot, not part of anyone's street level.
            node.pot += amount
            node.prior[actor] += amount
            continue

        if entry.get("phase") != phase:
            phase = entry.get("phase")
            _close_street(hand, schedule)
            node.prior = [node.prior[i] + node.committed[i] for i in (0, 1)]
            node.committed = [0, 0]
            node.raises_this_street = 0
            node.history += "/"
            node.street += 1

        if action == "fold":
            node.history += str(FOLD)
        elif action == "check":
            node.history += str(CHECK_CALL)
        elif action == "call":
            # A call's amount is the increment, not the level: 3,128 logged
            # calls carried the increment and none the level (15 September).
            # Reading it as a level undercounted the pot in 60 percent of
            # decisions and keyed a quarter of them at a line never taken.
            node.pot += amount
            node.committed[actor] += amount
            node.history += str(CHECK_CALL)
        elif action == "raise":
            to_call = max(node.committed[1 - actor] - node.committed[actor], 0)
            increment = amount - node.committed[actor]
            pot_after_call = node.pot + to_call
            fraction = (increment - to_call) / pot_after_call if pot_after_call else 0.0
            ceiling = start[actor] - node.prior[actor]
            node.history += str(_as_abstract(fraction, amount, ceiling, node,
                                             schedule, hand, rng))
            node.pot += increment
            node.committed[actor] = amount
            node.raises_this_street += 1
        elif action.startswith("post"):
            # A post this bridge has not seen (an ante under another name):
            # dead money, not a decision. Raising here folded a whole match.
            node.pot += amount
            node.prior[actor] += amount
        else:
            # An unknown verb is read as the passive action rather than as a
            # reason to abandon the hand's history.
            node.history += str(CHECK_CALL)

    # The arena's own phase field is authoritative for where we are, and it
    # can be ahead of the history: a street that opened with our action has no
    # entry on it yet.
    current = state.get("phase") or "preflop"
    while phase != current:
        index = PHASES.index(phase)
        if index + 1 >= len(PHASES):
            break
        phase = PHASES[index + 1]
        _close_street(hand, schedule)
        node.prior = [node.prior[i] + node.committed[i] for i in (0, 1)]
        node.committed = [0, 0]
        node.raises_this_street = 0
        node.history += "/"
        node.street += 1

    node.to_act = seat
    if hand._edits:
        alt = node.history
        for position, replacement in sorted(hand._edits, reverse=True):
            alt = alt[:position] + replacement + alt[position + 2:]
        hand.alt_history = alt
    return hand


def _contributions(history: Sequence[dict]) -> List[int]:
    """Chips each seat has put in over the hand so far, all streets."""
    total = [0, 0]
    level = [0, 0]
    phase = "preflop"
    for entry in history:
        actor = int(entry["seat"])
        action = entry["action"]
        amount = int(entry.get("amount") or 0)
        if action in BLIND_POSTS:
            level[actor] += amount
            total[actor] += amount
            continue
        if action == ANTE_POST:
            total[actor] += amount
            continue
        if entry.get("phase") != phase:
            phase = entry.get("phase")
            level = [0, 0]
        if action == "call":
            total[actor] += amount                  # the increment, see `replay`
            level[actor] += amount
        elif action == "raise":
            total[actor] += amount - level[actor]   # a raise is a level
            level[actor] = amount
        elif action.startswith("post"):
            total[actor] += amount
    return total


def legal_mask(node: Node, valid_actions: Sequence[str], schedule=1,
               tree: bool = True) -> np.ndarray:
    """
    The six actions, narrowed to what the arena allows and, by default, to the
    solver's tree.

    The arena's `valid_actions` is authoritative on legality, so anything it
    omits is removed even if the tree has it: a raise is not offered against an
    all-in opponent, for instance, and sending one would be rejected. The tree
    then removes what the solver never saw, the same way `slumbot.bridge.legal_mask`
    and the panel do. `tree=False` keeps only the arena's view, which is what a
    fallback for an off-tree node needs: at that node the tree has already run
    out, and refusing a raise the arena offers because the tree lacks one would
    make aces call a three-bet.
    """
    mask = np.ones(6, dtype=np.float64)
    if "fold" not in valid_actions or node.to_call <= 0:
        mask[FOLD] = 0.0
    if "check" not in valid_actions and "call" not in valid_actions:
        mask[CHECK_CALL] = 0.0
    allowed = (raise_sizes_at(schedule, node.raises_this_street, node.history.count("/"))
               if tree else (2, 3, 4, ALL_IN))
    for action in (2, 3, 4, ALL_IN):
        if action not in allowed or "raise" not in valid_actions:
            mask[action] = 0.0
    if not mask.any():
        mask[CHECK_CALL if ("check" in valid_actions or "call" in valid_actions)
             else FOLD] = 1.0
    return mask


def to_chipzen(action: int, node: Node, state: dict) -> dict:
    """
    An abstract action, as the arena's `{action, params}`.

    A raise goes out as a total bet level sized off the pot after the call, the
    engine's own definition, clamped into the range the server says is legal.
    There is no all-in action on the wire; a shove is a raise to `max_raise`.
    """
    to_call = int(state.get("to_call") or 0)
    if action == FOLD:
        return {"action": "fold", "params": {}}
    if action == CHECK_CALL:
        return {"action": "check" if to_call <= 0 else "call", "params": {}}

    low = int(state.get("min_raise") or 0)
    high = int(state.get("max_raise") or 0)
    if high <= 0:
        # Nothing to raise with: the shove the solver wanted is a call.
        return {"action": "check" if to_call <= 0 else "call", "params": {}}
    if action == ALL_IN:
        return {"action": "raise", "params": {"amount": high}}

    pot_after_call = int(state.get("pot") or 0) + to_call
    fraction = RAISE_FRACTIONS[action - 2]
    level = node.committed[node.to_act] + to_call + int(round(pot_after_call * fraction))
    # The ceiling wins over the floor: short, min_raise can exceed max_raise
    # (only a shove for less is possible), and max(low, ...) then sent more
    # chips than we had and was rejected.
    return {"action": "raise", "params": {"amount": min(high, max(low, level))}}


def cards(state: dict) -> Tuple[List[Card], List[Card]]:
    """Hole cards and board, as the engine's own type."""
    return (parse_cards(state.get("your_hole_cards") or []),
            parse_cards(state.get("board") or []))
