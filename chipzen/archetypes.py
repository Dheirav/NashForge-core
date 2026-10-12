"""
Scripted opponents for the duel: a handful of crude, parameterised playing styles (a station that calls a lot, a nit that
folds a lot, a maniac that raises a lot, and a few others), each deciding on Monte Carlo equity against a random hand
plus a few thresholds. They are crude on purpose: the point is that they are wrong in the ways real opponents are
wrong, so a change that only wins against a solver can be told from one that also wins against them.

The interface is the duel's: `decide(state, valid, seat)` returning the arena's action dict, and a `stats` object with
the counters the duel prints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from chipzen.bridge import parse_cards

ARCHETYPES = ("station", "nit", "maniac", "foldraise",
              "sticky", "nofold3bet", "folder", "wildpassive", "meek", "bully", "reraiser", "thirdraiser")

PARAMS = {
    "station":  dict(open_eq=0.56, limp_eq=0.30, threebet_eq=0.58, fold_margin=0.12, raise_eq=0.78,
                     raise_p=0.5, bluff_p=0.0, call_p=0.75, defend_eq=0.30, defend3_eq=0.48),
    "nit":      dict(open_eq=0.47, limp_eq=0.55, threebet_eq=0.60, fold_margin=0.04, raise_eq=0.58,
                     raise_p=0.6, bluff_p=0.05, call_p=0.05, defend_eq=0.55, defend3_eq=0.58),
    "maniac":   dict(open_eq=0.50, limp_eq=0.45, threebet_eq=0.53, fold_margin=0.02, raise_eq=0.52,
                     raise_p=0.6, bluff_p=0.15, call_p=0.50, defend_eq=0.45, defend3_eq=0.78),
    "foldraise": dict(open_eq=0.54, limp_eq=0.68, threebet_eq=0.58, fold_margin=0.20, raise_eq=0.55,
                      raise_p=0.7, bluff_p=0.05, call_p=0.30, defend_eq=0.62, defend3_eq=0.58),


    # Agents 04, 08, 01, 05: enters 67 to 78% of hands, folds to a bet 0 to
    # 11% of the time, calls 94 to 97% of its answers, shows down 82 to 96%
    # of hands. Our "station" folds to a bet 35% of the time, which is three
    # times as willing; against a bot that never folds the right answer is to
    # stop bluffing and value bet much thinner, and that is the answer the
    # exploiter has never been asked for.
    "sticky":   dict(open_eq=0.52, limp_eq=0.20, threebet_eq=0.80, fold_margin=-0.30, raise_eq=0.85,
                     raise_p=0.3, bluff_p=0.0, call_p=0.98, defend_eq=0.18, defend3_eq=0.25),

    # Nine of the eleven fold to a three-bet 0 or 1% of the time, where every
    # shape above folds 46 to 62%. So a value three-bet is worth far more
    # against the real field than our panel says, and a three-bet bluff far
    # less; the read that three-bets a folder must never fire here.
    "nofold3bet": dict(open_eq=0.50, limp_eq=0.35, threebet_eq=0.62, fold_margin=-0.10, raise_eq=0.70,
                       raise_p=0.4, bluff_p=0.05, call_p=0.85, defend_eq=0.22, defend3_eq=0.20),

    # Agent 07: enters 0% of hands, folds 100% of blinds to an open, 3,188
    # hands in the corpus. Trivial to beat, and in the panel because a set
    # that fails to raise every hand against it is broken.
    "folder":   dict(open_eq=0.95, limp_eq=0.95, threebet_eq=0.95, fold_margin=0.45, raise_eq=0.95,
                     raise_p=0.1, bluff_p=0.0, call_p=0.9, defend_eq=0.90, defend3_eq=0.92),

    # Agent 06: three-bets 46% of hands and then folds to a bet 0% of the
    # time, calling 81% of its answers. Wild before the flop, immovable
    # after it; our "maniac" is aggressive throughout and never plays this.
    "wildpassive": dict(open_eq=0.36, limp_eq=0.20, threebet_eq=0.40, fold_margin=-0.25, raise_eq=0.82,
                        raise_p=0.3, bluff_p=0.02, call_p=0.95, defend_eq=0.20, defend3_eq=0.24, open_frac=1.0),

    "meek":     dict(open_eq=0.44, limp_eq=0.40, threebet_eq=0.82, fold_margin=0.12, raise_eq=0.85,
                     raise_p=0.25, bluff_p=0.0, call_p=0.85, defend_eq=0.42, defend3_eq=0.46),

    "bully":    dict(open_eq=0.47, limp_eq=0.60, threebet_eq=0.56, fold_margin=0.16, raise_eq=0.45,
                     raise_p=0.80, bluff_p=0.45, call_p=0.25, defend_eq=0.52, defend3_eq=0.60,
                     open_frac=1.0),
    "reraiser": dict(open_eq=0.50, limp_eq=0.45, threebet_eq=0.38, fold_margin=0.02, raise_eq=0.52,
                     raise_p=0.6, bluff_p=0.15, call_p=0.20, defend_eq=0.45, defend3_eq=0.78),
    "thirdraiser": dict(open_eq=0.50, limp_eq=0.45, threebet_eq=0.38, fold_margin=0.02, raise_eq=0.52,
                        raise_p=0.6, bluff_p=0.15, call_p=0.20, defend_eq=0.45, defend3_eq=0.78,
                        third_raise_frac=0.5),
}


@dataclass
class ArchetypeStats:
    decisions: int = 0
    misses: int = 0
    companion_hits: int = 0
    fallbacks: int = 0


class Archetype:
    """A scripted opponent; `kind` is one of ARCHETYPES, `params` a row of PARAMS."""

    def __init__(self, kind: str, rng: np.random.Generator, samples: int = 200, params: dict = None):
        if kind not in ARCHETYPES:
            raise ValueError(f"archetype must be one of {ARCHETYPES}, got {kind!r}")
        self.kind = kind
        self.p = dict(PARAMS[kind], **(params or {}))
        self.rng = rng
        self.samples = samples
        self.stats = ArchetypeStats()
        self.label = kind
        self.opponent = None
        self.profiles = None

    def _equity(self, state: dict) -> float:
        import pokerbot_native as native
        hole = [c.index for c in parse_cards(state["your_hole_cards"])]
        board = [c.index for c in parse_cards(state.get("board") or [])]
        return float(native.equity_vs_random(hole, board, self.samples, int(self.rng.integers(0, 2 ** 62))))

    @staticmethod
    def _raise_to(state: dict, fraction: float, allin: bool = False) -> dict:
        """A raise to about `fraction` of the pot after the call, clipped to the arena's bounds."""
        if allin:
            return {"action": "raise", "params": {"amount": int(state["max_raise"])}}
        pot_after_call = int(state["pot"]) + int(state["to_call"])
        level = int(state["min_raise"]) + int(fraction * pot_after_call)
        level = max(int(state["min_raise"]), min(level, int(state["max_raise"])))
        return {"action": "raise", "params": {"amount": level}}

    @staticmethod
    def _passive(valid: Sequence[str]) -> dict:
        return {"action": "check" if "check" in valid else "call", "params": {}}

    @staticmethod
    def _fold(valid: Sequence[str]) -> dict:
        return {"action": "fold" if "fold" in valid else "check", "params": {}}

    @staticmethod
    def _blind(state: dict) -> int:
        return max((int(a.get("amount") or 0) for a in state.get("action_history", [])
                    if a.get("action") == "post_big_blind"), default=100)

    def _short(self, state: dict) -> bool:
        return int(state["your_stack"]) + int(state["to_call"]) <= 12 * self._blind(state)

    @staticmethod
    def _raises_before(state: dict) -> int:
        """Raises so far on the current street, from the arena's history."""
        hist = state.get("action_history") or []
        phase = state.get("phase") or "preflop"
        return sum(1 for a in hist if a.get("phase") == phase and a.get("action") == "raise")

    def decide(self, state: dict, valid: Sequence[str], seat: int) -> dict:  # noqa: ARG002 (the duel passes it)
        self.stats.decisions += 1
        p = self.p
        e = self._equity(state)
        to_call = int(state["to_call"])
        pot = int(state["pot"])
        facing = to_call > 0
        price = to_call / (pot + to_call) if facing else 0.0
        preflop = not state.get("board")
        can_raise = "raise" in valid
        raises = self._raises_before(state)
        u = float(self.rng.random())

        if preflop:
            if raises == 0:
                # Unopened, or the big blind facing a limp: open, limp, or fold.
                if can_raise and e >= p["open_eq"]:
                    return self._raise_to(state, p.get("open_frac", 0.5))
                if e >= p["limp_eq"] or not facing:
                    return self._passive(valid)
                return self._fold(valid)
            # Facing a raise (or more).
            if can_raise and e >= p["threebet_eq"] and u > p["call_p"] * 0.5:
                if raises >= 2 and p.get("third_raise_frac") is not None and not self._short(state):
                    # A sized third raise, the one the (4,2,1) tree cannot express; jams only once short.
                    return self._raise_to(state, p["third_raise_frac"])
                return self._raise_to(state, 1.0, allin=self._short(state) or raises >= 2)
            defend = p["defend3_eq"] if raises >= 2 else p["defend_eq"]
            if e >= defend and e >= price + p["fold_margin"]:
                return self._passive(valid)
            return self._fold(valid)

        # Postflop.
        if facing:
            if e < price + p["fold_margin"]:
                return self._fold(valid)
            if can_raise and e >= p["raise_eq"] and u > p["call_p"]:
                return self._raise_to(state, 1.0, allin=self._short(state))
            return self._passive(valid)
        if can_raise and ((e >= p["raise_eq"] and u < p["raise_p"]) or u < p["bluff_p"]):
            return self._raise_to(state, 0.66)
        return self._passive(valid)


#: Columns of `action_probabilities`: the three things `decide` can do, by kind.
FOLD, PASSIVE, RAISE = 0, 1, 2


def action_probabilities(p: dict, equity: np.ndarray, facing: bool, price: float, preflop: bool,
                         can_raise: bool, raises: int) -> np.ndarray:
    """
    The chance `Archetype.decide` folds, checks or calls, or raises, for each
    of an array of equities, over its uniform draw `u`. It mirrors `decide`
    branch for branch; `tests/test_river_oracle.py` holds the two together by
    sampling `decide`, so a change to one that is not made to the other fails.

    The river oracle (`cfr/river_oracle.py`) needs the policy as a function
    of the hand rather than one sampled action, which `decide` cannot give.
    """
    e = np.asarray(equity, dtype=float)
    raise_ = np.zeros_like(e)
    if preflop:
        if raises == 0:
            if can_raise:
                raise_ = (e >= p["open_eq"]).astype(float)
            passive = (1.0 - raise_) * ((e >= p["limp_eq"]) | (not facing))
        else:
            if can_raise:
                raise_ = (e >= p["threebet_eq"]) * (1.0 - p["call_p"] * 0.5)
            defend = p["defend3_eq"] if raises >= 2 else p["defend_eq"]
            passive = (1.0 - raise_) * ((e >= defend) & (e >= price + p["fold_margin"]))
    elif facing:
        stays = (e >= price + p["fold_margin"]).astype(float)
        if can_raise:
            raise_ = stays * (e >= p["raise_eq"]) * (1.0 - p["call_p"])
        passive = stays - raise_
    else:
        if can_raise:
            # (e >= raise_eq and u < raise_p) or u < bluff_p
            raise_ = np.where(e >= p["raise_eq"], max(p["raise_p"], p["bluff_p"]), p["bluff_p"])
        passive = 1.0 - raise_
    fold = 1.0 - raise_ - passive
    return np.stack([fold, passive, raise_], axis=-1)


def build_archetype(kind: str, rng: np.random.Generator, params: dict = None) -> Archetype:
    return Archetype(kind, rng, params=params)
