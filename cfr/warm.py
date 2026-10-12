"""
Warm-start entries from one betting tree, laid out in another tree's action order.

`MCCFR::warm_start` copies an entry onto a new node position by position, which is right only when the
two trees list the same actions at that node. Our new trees insert a size into existing nodes (T1 and T2
add a half-pot raise after a bet: fold, call, 2x, jam becomes fold, call, half, 2x, jam), so a positional
copy would put the old 2x probability on the new half, jam on 2x, and nothing on jam. Here each old
probability is placed under its own action in the new node's list and a size the old tree lacked starts at
zero; when the two trees agree the entry comes back unchanged.

Both action lists come from `evaluation.benchmark._solver_actions`, the reconstruction the bot's lookups
use, which reads the street from the history key; whether the node faces a bet is read from the key too.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from abstraction.betting import CHECK_CALL, FOLD, RAISE_ACTIONS
from evaluation.benchmark import _solver_actions

_RAISES = {str(a) for a in RAISE_ACTIONS}


def facing_a_bet(history: str) -> bool:
    """Preflop's first decision faces the big blind; otherwise, whether the street's last action raised."""
    street = history.split("/")[-1]
    if not street:
        return "/" not in history          # preflop root: the small blind faces the big blind
    return street[-1] in _RAISES


def actions_at(history: str, schedule, width: Optional[int] = None) -> Optional[List[int]]:
    """The node's action list under `schedule`; with `width`, None unless it has that many actions."""
    to_call = 1 if facing_a_bet(history) else 0
    actions = list(_solver_actions(history, to_call, schedule))
    if width is not None and len(actions) != width:
        if width == 2 and to_call:
            return [FOLD, CHECK_CALL]       # stack-capped: fold or call, as cfr_agent reads it
        return None
    return actions


def remap(key: str, probabilities: Sequence[float], old_schedule, new_schedule) -> Optional[List[float]]:
    """One entry in the new tree's order, or None when the old node's actions cannot be reconstructed."""
    history = key.split("|", 1)[1] if "|" in key else ""
    old = actions_at(history, old_schedule, len(probabilities))
    if old is None:
        return None
    # A stack-capped node (fold or call only, where the schedule alone lists more) stays capped: the same
    # history leaves the same stacks in both trees, so the cap does not depend on the sizes the tree offers.
    capped = old == [FOLD, CHECK_CALL] and len(actions_at(history, old_schedule)) != 2
    new = [FOLD, CHECK_CALL] if capped else actions_at(history, new_schedule)
    by_action: Dict[int, float] = {a: float(p) for a, p in zip(old, probabilities)}
    return [by_action.get(a, 0.0) for a in new]


def remap_entries(entries: Iterable[Tuple[str, Sequence[float]]], old_schedule, new_schedule
                  ) -> Tuple[List[Tuple[str, List[float]]], int]:
    """All entries in the new tree's order, and how many had to be dropped."""
    out, dropped = [], 0
    for key, probabilities in entries:
        mapped = remap(key, probabilities, old_schedule, new_schedule)
        if mapped is None:
            dropped += 1
        else:
            out.append((key, mapped))
    return out, dropped
