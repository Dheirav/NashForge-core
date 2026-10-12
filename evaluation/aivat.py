"""
AIVAT's practical subset for the arena logs (Burch, Schmid, Moravcik and Bowling, AAAI 2018).

A hand's realised net is replaced by

    net - sum over chance nodes and our decisions of (v(what happened) - E[v over what could have happened])

where v is a value function frozen before the data it scores is seen. Each subtracted term has expectation zero
given everything before it, so the mean is unchanged whatever v is, while a v that tracks the hand's eventual result
cancels the luck of the deal and the board. Three kinds of term are used:

- **the deal of our hole cards**: E over all 1,326 holdings, which is exact because nothing else is known yet;
- **each board card or flop**: E over the cards we have not seen. The opponent's two cards are among them, so this is
  not quite the true conditional distribution (that would need the opponent's range); the error is a card-removal
  effect of order 2/47 of the term, and the simulation in `scripts/aivat_sim.py` bounds what it does to the mean;
- **our own decisions**, only where the distribution we sampled from is known exactly: a decision row that carries
  it (`probs`). A purified set plays one action with certainty, so its term is identically zero and is left out,
  and a zero term is unbiased by construction. An approximate distribution is never used, because a wrong one is
  exactly what makes the term's mean nonzero.

The opponent's decisions get no term, as AIVAT allows for a player whose strategy is unknown, and neither does the
deal of their cards, which we see only at showdown and on our folds: a term that exists only when the opponent's
cards happen to be shown would be selected on what came after it. A called all-in before the river keeps the exact
treatment of `--allin-adjust` (its base value is the expectation over the runouts, which is AIVAT with a perfect
value function on those chance nodes), and only the chance nodes up to the all-in street get a heuristic term.

The value function is v = g_street(pot, effective stack) * (equity against a random hand - 1/2) for a state where
the contributions are equal, and minus our contribution after a fold. Equity is exact (every opponent holding and
every runout) after the flop and a frozen table preflop. Because equity against a random hand from the unseen cards
is a martingale over the next card, E over the next card of v is v with the previous street's equity, so no chance
expectation has to be enumerated: a chance term is g * (equity now - equity before). With g = pot, v is the
check-down value against a random hand (pot * equity less our half of it); the frozen file fits g's two coefficients per street on matches played
before the bursts it is used on (an arena-side fit script, not in this repository).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

VALUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aivat_value.json")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_AT = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
#: The bridge's raise sizes, fractions of the pot after the call (chipzen.bridge.to_chipzen), so a slot's value is
#: priced at the chips that slot would have sent.
RAISE_FRACTIONS = (0.5, 1.0, 2.0)
FOLD, CALL, ALL_IN = 0, 1, 5
_RANKS = "23456789TJQKA"
_SUITS = "hdcs"


def card_index(card: str) -> int:
    """'Kh' to the native deck index, suit * 13 + rank (engine.cards.Card.index)."""
    return _SUITS.index(card[1]) * 13 + _RANKS.index(card[0])


def hand_class(hole: Sequence[int]) -> str:
    """'AKs', 'AKo' or 'AA' for two native indices: preflop equity against a random hand depends on nothing else."""
    (ra, sa), (rb, sb) = sorted(((c % 13, c // 13) for c in hole), reverse=True)
    if ra == rb:
        return _RANKS[ra] * 2
    return _RANKS[ra] + _RANKS[rb] + ("s" if sa == sb else "o")


def preflop_table(samples: int = 4_000_000, seed: int = 20261005) -> Dict[str, float]:
    """Equity against a random hand for each of the 169 classes, by the native sampler (about 0.4 s a class)."""
    import pokerbot_native as native
    out = {}
    for a, b in itertools.combinations(range(52), 2):
        key = hand_class((a, b))
        if key not in out:
            out[key] = float(native.equity_vs_random([a, b], [], samples, seed + len(out)))
    return out


@lru_cache(maxsize=200_000)
def _board_equity(hole: tuple, board: tuple) -> float:
    """
    Exact equity against a uniformly random holding from the cards we have not seen, ties counted half.

    `allin_edge` enumerates the runouts in C for one opposing holding and returns P(win) - P(lose); averaging it over
    every holding gives the same thing against a random hand. About 0.08 s on the flop, 4 ms on the turn.
    """
    import pokerbot_native as native
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    edges = [native.allin_edge(list(hole), list(o), list(board), True) for o in itertools.combinations(deck, 2)]
    return (sum(edges) / len(edges) + 1.0) / 2.0


class ValueFunction:
    """g_street(pot, eff) = a * pot + b * eff, times (equity - 1/2); see the module docstring."""

    def __init__(self, coefficients: Dict[str, Sequence[float]], preflop: Dict[str, float], name: str = "",
                 provenance: Optional[dict] = None):
        self.coefficients = {s: (float(coefficients[s][0]), float(coefficients[s][1])) for s in STREETS}
        self.preflop = preflop
        self.name = name
        self.provenance = provenance or {}

    @property
    def digest(self) -> str:
        """What the ledger records: a value function is fixed before it scores, and this says which one it was."""
        payload = json.dumps({"coefficients": self.coefficients, "preflop": self.preflop}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: str = VALUE_FILE) -> "ValueFunction":
        with open(path) as handle:
            data = json.load(handle)
        return cls(data["coefficients"], data["preflop_equity"], data.get("name", os.path.basename(path)),
                   data.get("provenance"))

    @classmethod
    def checkdown(cls, preflop: Dict[str, float]) -> "ValueFunction":
        """g = pot: the hand checked down from here against a random hand (pot * equity - our half). Nothing fitted."""
        return cls({s: (1.0, 0.0) for s in STREETS}, preflop, "checkdown")

    def equity(self, hole: Sequence[int], board: Sequence[int]) -> float:
        if not board:
            return self.preflop[hand_class(hole)]
        return _board_equity(tuple(sorted(hole)), tuple(sorted(board)))

    def scale(self, street: str, pot: float, eff: float) -> float:
        a, b = self.coefficients[street]
        return a * pot + b * eff


def _phase_order(phase: str) -> int:
    return STREETS.index(phase)


def trace_hand(result: dict, before: Sequence[int], decisions: List[dict], seat: int, hole: Sequence[str]) -> dict:
    """
    The chance nodes and our decision nodes of one logged hand, with the pot and stacks at each.

    A raise's amount in the arena's history is the seat's total for the street and a call's the increment, the same
    reading as chipzen_decompose.contributions. The board of a street is the one on our first decision there; every
    street the betting reached has one (checked on all 44,617 logged decisions on 5 October), and a street reached
    with nobody left to act is the all-in case, which the caller treats exactly.
    """
    history = result.get("action_history") or []
    total, street_total, street = [0, 0], [0, 0], None
    chance = [{"street": "preflop", "board": [], "pot": None, "eff": float(min(before))}]
    ours = []
    boards = {}
    # Rows are paired with our actions street by street, so a row the log lost costs only its own term. With one
    # queue for the hand, every later row would shift onto the wrong street's action and be dropped by the phase
    # check, and those lost terms would not be counted anywhere.
    by_street: Dict[str, List[dict]] = {}
    for d in decisions:
        boards.setdefault(d["phase"], [card_index(c) for c in (d.get("board") or [])])
        by_street.setdefault(d["phase"], []).append(d)
    queues = {p: iter(rows) for p, rows in by_street.items()}
    unlogged = 0
    for a in history:
        if a["phase"] != street:
            if street is not None:
                chance.append({"street": a["phase"], "board": boards.get(a["phase"]), "pot": sum(total),
                               "eff": float(min(before[0] - total[0], before[1] - total[1]))})
            street, street_total = a["phase"], [0, 0]
        s = a["seat"]
        if s == seat and not a["action"].startswith("post"):
            d = next(queues.get(a["phase"], iter(())), None)
            if d is None:
                unlogged += 1
            else:
                ours.append({"row": d, "street": a["phase"], "board": boards.get(a["phase"]),
                             "our": total[s], "opp": total[1 - s],
                             "our_stack": before[s] - total[s], "opp_stack": before[1 - s] - total[1 - s]})
        if a["action"] in ("raise", "bet", "all_in"):
            add = a["amount"] - street_total[s]
            street_total[s] = a["amount"]
        elif a["action"] in ("call", "post_small_blind", "post_big_blind"):
            add = a["amount"]
            street_total[s] += a["amount"]
        else:
            add = 0
        total[s] += add
        if a["action"].startswith("post"):
            chance[0]["pot"] = sum(total)
    if chance[0]["pot"] is None:
        chance[0]["pot"] = 0
    return {"hole": [card_index(c) for c in hole], "chance": chance, "ours": ours, "unlogged": unlogged}


def action_values(vf: ValueFunction, street: str, eq: float, node: dict) -> np.ndarray:
    """
    v after each of the six abstract actions at one of our decisions; nan where the slot is illegal.

    A fold is terminal and worth minus what we have put in. Anything else is valued as if the opponent matched it
    as far as their stack allows, at that street's g and the equity we hold now: a bet's chance of taking the pot
    is not modelled, which costs variance reduction and nothing else.
    """
    d = node["row"]
    pot, to_call = int(d["pot"]), int(d["to_call"])
    our, opp, our_stack, opp_stack = node["our"], node["opp"], node["our_stack"], node["opp_stack"]
    legal = d["legal"]
    out = np.full(6, np.nan)
    for a in range(6):
        if not legal[a]:
            continue
        if a == FOLD:
            out[a] = -our
            continue
        if a == CALL:
            inc = min(to_call, our_stack)
        elif a == ALL_IN:
            inc = our_stack
        else:
            inc = min(our_stack, to_call + int(round(RAISE_FRACTIONS[a - 2] * (pot + to_call))))
        matched = min(our + inc, opp + opp_stack)
        eff = min(our + our_stack, opp + opp_stack) - matched
        out[a] = vf.scale(street, 2 * matched, eff) * (eq - 0.5)
    return out


def decision_distribution(node: dict) -> Optional[np.ndarray]:
    """
    The distribution our seat sampled from at this decision, if the log says exactly what it was.

    Only a row's own `probs` qualifies. A purified set's choice is certain, so its term would be zero anyway; a
    distribution recovered from a solver that did not decide (a companion, the rule, a read) would be wrong.
    """
    probs = node["row"].get("probs")
    if probs is None:
        return None
    p = np.asarray(probs, dtype=float) * np.asarray(node["row"]["legal"], dtype=float)
    return p / p.sum() if p.sum() > 0 else None


def hand_terms(vf: ValueFunction, trace: dict, last_street: Optional[str] = None) -> dict:
    """
    The control terms of one hand: one per chance node and one per decision with a known distribution.

    `last_street` stops the chance terms after that street, for a called all-in whose later cards the caller has
    already taken the exact expectation over. Returns the terms by kind, the chance features (for fitting g) and
    counts of what was left out. `decisions_without` counts logged rows that get no term, while
    `decisions_unlogged` counts our actions the log has no row for, which cannot get one either.
    """
    hole = trace["hole"]
    stop = _phase_order(last_street) if last_street else 3
    out = {"chance": {}, "features": {}, "decision": 0.0, "decisions_with_term": 0, "decisions_without": 0,
           "decisions_unlogged": trace.get("unlogged", 0), "board_missing": 0}
    previous = 0.5
    for node in trace["chance"]:
        street = node["street"]
        if _phase_order(street) > stop:
            break
        board = node["board"]
        if board is None or len(board) != BOARD_AT[street]:
            out["board_missing"] += 1
            break                       # the equities after a missing street are not chained to a known one
        eq = vf.equity(hole, board)
        delta = eq - previous
        out["features"][street] = (node["pot"] * delta, node["eff"] * delta)
        out["chance"][street] = vf.scale(street, node["pot"], node["eff"]) * delta
        previous = eq
    for node in trace["ours"]:
        sigma = decision_distribution(node)
        if sigma is None or node["board"] is None or not node["row"]["legal"][int(node["row"]["choice"])]:
            out["decisions_without"] += 1
            continue
        if last_street and _phase_order(node["street"]) > stop:
            continue
        eq = vf.equity(hole, node["board"])
        v = action_values(vf, node["street"], eq, node)
        choice = int(node["row"]["choice"])
        expected = float(np.nansum(sigma * np.nan_to_num(v)))
        out["decision"] += float(v[choice]) - expected
        out["decisions_with_term"] += 1
    out["total"] = sum(out["chance"].values()) + out["decision"]
    return out
