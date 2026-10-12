"""
Strategy purification: play a cleaner version of a solved strategy.

An abstracted, sampled solve leaves small probabilities on actions it has not finished ruling out, and at
play time those are noise. Ganzfried and Sandholm (2012) found removing them often helps against strong
opponents: *purification* plays the most probable action, *thresholding* drops actions below a share and
keeps the rest mixed. Both raise worst-case exploitability, so each mode is judged by head to head, the
field duels and LBR together. One function, so the agent that plays and LBR that measures see the same thing.

Modes: "none"; "postflop" and "all" (the most probable action, after the flop or everywhere, first on a tie);
"allx" (as "all", except facing an all-in, where the mix is kept: see evaluation/benchmark.py);
"t10", "t20", ... (drop actions under that percentage of the node's mass, renormalise, keep sampling).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

BASE_MODES = ("none", "postflop", "all", "allx")


def valid(mode: str) -> bool:
    return mode in BASE_MODES or (mode.startswith("t") and mode[1:].isdigit() and 0 < int(mode[1:]) < 100)


def row(probabilities: Sequence[float], mode: str, postflop: bool) -> np.ndarray:
    """One node's distribution under `mode`; `postflop` says whether the node is after the flop."""
    p = np.asarray(probabilities, dtype=np.float64)
    total = p.sum()
    if mode == "none" or total <= 0 or (mode == "postflop" and not postflop):
        return p
    if mode == "allx":
        mode = "all"          # the facing-all-in exception needs the history: apply_to_table decides it
    if mode in ("postflop", "all"):
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = total
        return out
    if not valid(mode):
        raise ValueError(f"purify mode {mode!r}: none, postflop, all or tNN")
    share = int(mode[1:]) / 100.0
    kept = np.where(p >= share * total, p, 0.0)
    if kept.sum() <= 0:                      # nothing clears the bar: keep the largest
        kept[int(np.argmax(p))] = p.max()
    return kept * (total / kept.sum())


def apply_to_table(strategy, mode: str):
    """The whole strategy under `mode`: a FlatStrategy with new values, or a dict of new rows."""
    if mode == "none" or getattr(strategy, "is_compact", False):
        return strategy
    from cfr.flat import FlatStrategy
    if isinstance(strategy, FlatStrategy):
        values = np.array(strategy._values, dtype=np.float64, copy=True)
        offsets = strategy._offsets
        for i, key in enumerate(strategy._keys):
            a, b = int(offsets[i]), int(offsets[i + 1])
            history = key.decode().split("|", 1)[-1]
            if mode == "allx" and history.endswith("5"):
                continue                                  # facing an all-in: the mix stays
            values[a:b] = row(values[a:b], mode, "/" in history)
        return FlatStrategy(strategy._keys, offsets, values.astype(strategy._values.dtype, copy=False))
    return {k: (v if mode == "allx" and k.split("|", 1)[-1].endswith("5") else row(v, mode, "/" in k.split("|", 1)[-1]))
            for k, v in strategy.items()}
