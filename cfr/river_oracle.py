"""
The opponent's river range when its policy is known: the oracle arm of the
river test.

On 24 September river re-solving (`cfr/river.py`) won 55.4 ± 1.1% against the
blueprint it came from, and nothing against the scripted shapes: bully
53.3 against 55.3 without it, meek 58.7 against 58.4, 1,000 matches each. The
solver takes the opponent's range from the blueprint's own reach, which is
exactly right in the mirror and wrong against anything else. Whether that
assumption is what cost the gain, or the solve's equilibrium objective is, can
be told apart by giving the solver the true range, and against a scripted
shape the true range is computable: the script is code.

So this replays the hand's public history and, at each of the opponent's
decisions, multiplies every hand's reach by the chance the script would have
done what it did (`chipzen.archetypes.action_probabilities`). The script
decides on Monte Carlo equity against a random hand, `samples` runouts, so its
equity is noisy around the true one and a hand near a threshold goes either
way; that noise is modelled rather than ignored, because a hand's chance of
crossing a threshold is most of what separates the ranges of two shapes.

This is an upper bound for the test, not a player: no real opponent's policy
is known. It says how much a perfect read would be worth.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

import numpy as np

from chipzen.archetypes import FOLD, PASSIVE, RAISE, action_probabilities
from engine.cards import Card

#: Runouts behind the equity the oracle treats as the truth. The script uses
#: 200, so this is fifty times tighter; 0.19 ms a hand, measured 24 Sept.
PRECISION = 10_000

#: Points of the script's equity noise the policy is averaged over: the
#: midpoints of equal-probability slices of a normal.
_NODES = np.array([NormalDist().inv_cdf((k + 0.5) / 24) for k in range(24)])

_PREFLOP: Dict[Tuple[int, int], float] = {}
_POSTFLOP: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}

_CATEGORY = {"fold": FOLD, "check": PASSIVE, "call": PASSIVE, "raise": RAISE}
_BOARD_AT = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _equities(pairs: np.ndarray, board: Sequence[int], full_board: Tuple[int, ...]) -> np.ndarray:
    """True equity against a random hand, per hole pair, on this board prefix. The
    pairs are the river's `HandSet`, which the full board fixes, so it keys the cache."""
    import pokerbot_native as native
    if not board:
        out = np.empty(len(pairs))
        for i, (x, y) in enumerate(pairs):
            key = (int(x), int(y))
            if key not in _PREFLOP:
                _PREFLOP[key] = float(native.equity_vs_random([key[0], key[1]], [], PRECISION, 1_000_003 * key[0] + key[1]))
            out[i] = _PREFLOP[key]
        return out
    key = (tuple(int(c) for c in board), full_board)
    cached = _POSTFLOP.get(key)
    if cached is not None:
        return cached
    board_list = [int(c) for c in board]
    out = np.array([native.equity_vs_random([int(x), int(y)], board_list, PRECISION, 7919 * int(x) + int(y))
                    for x, y in pairs])
    if len(_POSTFLOP) >= 64:
        _POSTFLOP.clear()
    _POSTFLOP[key] = out
    return out


def _noisy(p: dict, equity: np.ndarray, samples: int, **context) -> np.ndarray:
    """`action_probabilities` averaged over the script's own sampling error."""
    sd = np.sqrt(np.clip(equity * (1.0 - equity), 1e-6, None) / samples)
    total = np.zeros(equity.shape + (3,))
    for z in _NODES:
        total += action_probabilities(p, np.clip(equity + z * sd, 0.0, 1.0), **context)
    return total / len(_NODES)


def decisions(state: dict, opponent_seat: int) -> List[dict]:
    """
    The opponent's decisions this hand, each with what the script saw when it
    made it: the arena's pot and price, the raises before it on the street,
    whether a raise was legal. Rebuilt from `action_history` by the dealer's
    own accounting (`scripts/chipzen_duel.py`, `Dealer`): a raise's amount is
    the level on the street, a call's and a blind's are chips added.
    """
    history = state.get("action_history") or []
    me = 1 - opponent_seat
    # Stacks at the start of the hand: now, plus everything each seat has put in.
    put_in = [0, 0]
    committed = [0, 0]
    phase = None
    for a in history:
        if a.get("phase") != phase:
            phase, committed = a.get("phase"), [0, 0]
        seat, amount = int(a["seat"]), int(a.get("amount") or 0)
        if a["action"] == "raise":
            put_in[seat] += amount - committed[seat]
            committed[seat] = amount
        elif a["action"] in ("call", "post_small_blind", "post_big_blind"):
            put_in[seat] += amount
            committed[seat] += amount
    stack = [0, 0]
    stack[me] = int(state.get("your_stack") or 0) + put_in[me]
    stack[opponent_seat] = int((state.get("opponent_stacks") or [0])[0]) + put_in[opponent_seat]

    out = []
    committed, pot, phase, raises = [0, 0], 0, None, 0
    for a in history:
        if a.get("phase") != phase:
            phase, committed, raises = a.get("phase"), [0, 0], 0
        seat, action, amount = int(a["seat"]), a["action"], int(a.get("amount") or 0)
        if seat == opponent_seat and action in _CATEGORY:
            to_call = max(min(committed[me] - committed[seat], stack[seat]), 0)
            out.append({"phase": phase, "category": _CATEGORY[action], "facing": to_call > 0,
                        "price": to_call / (pot + to_call) if to_call > 0 else 0.0,
                        "can_raise": stack[seat] > to_call and stack[me] > 0, "raises": raises})
        if action == "raise":
            increment = amount - committed[seat]
            committed[seat] = amount
            stack[seat] -= increment
            pot += increment
            raises += 1
        elif action in ("call", "post_small_blind", "post_big_blind"):
            committed[seat] += amount
            stack[seat] -= amount
            pot += amount
    return out


def script_reach(pairs: np.ndarray, board: Sequence[Card], params: dict, state: dict,
                 opponent_seat: int, samples: int = 200) -> np.ndarray:
    """
    The scripted opponent's reach over `pairs` (a `HandSet`'s hole pairs) at
    this river decision: the product over its decisions this hand of the chance
    it took the action it did. A hand the script would never have played this
    way gets zero, which the blueprint's range never gives.
    """
    board_index = [c.index for c in board]
    reach = np.ones(len(pairs))
    for d in decisions(state, opponent_seat):
        equity = _equities(pairs, board_index[:_BOARD_AT[d["phase"]]], tuple(board_index))
        probabilities = _noisy(params, equity, samples, facing=d["facing"], price=d["price"],
                               preflop=d["phase"] == "preflop", can_raise=d["can_raise"],
                               raises=d["raises"])
        reach *= probabilities[:, d["category"]]
    return reach
