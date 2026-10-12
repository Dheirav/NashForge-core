"""
Per-information-set visit counts from the native solver, on disk and back.

The solver's budget rule was equal iterations per reached decision point, which
is an average: iterations over `information_sets_reached`. The nodes behind a
three-bet or an overbet get a small fraction of it, and until the counter
existed that tail had never been measured (docs/research/2026-10-05-visit-counter.md).

Two counts per node, both written only while `set_count_visits(True)` is on:

- `avg_visits`: samples that went into the node's average strategy. Under
  external sampling the owner's node is entered as the non-traverser along one
  sampled path, so this is T times (chance reach x the owner's own reach), the
  number of draws the exported strategy is an average of.
- `regret_visits`: regret updates, made with the owner as the traverser, so T
  times (chance reach x the opponent's reach).

The file sits beside the flat pair as `<name>.visits.npz` and its keys are the
flat export's keys in the same sorted order, so row i of one is row i of the
other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from cfr.flat import KEY_WIDTH

#: Action symbols that are raises in a history string (native/src/nolimit.hpp:
#: 0 fold, 1 check or call, 2 to 5 the raise sizes and all-in).
RAISE_SYMBOLS = frozenset("2345")
STREETS = ("preflop", "flop", "turn", "river")


def visits_path(output: str, iteration: Optional[int] = None) -> str:
    """`<name>.visits.npz` beside `<name>.pkl`; a mid-run snapshot adds its iteration."""
    # Strip only ".pkl", as cfr.flat.flat_paths does: splitext would turn rung_0.5 into rung_0 and let rung_0.5
    # and rung_0.7 overwrite each other's counts, and the counts must sit beside the flat pair they describe.
    base = output[:-len(".pkl")] if output.endswith(".pkl") else output
    return f"{base}.visits.npz" if iteration is None else f"{base}.visits.{iteration}.npz"


@dataclass
class Visits:
    keys: np.ndarray            # fixed-width bytes, sorted
    avg: np.ndarray             # uint32
    regret: np.ndarray          # uint32
    iterations: int             # the solver's counter when the snapshot was taken
    averaged_iterations: int    # of those, the ones that fed the average
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.keys))


def from_solver(solver, averaged_from: int = 0, meta: Optional[Dict] = None) -> Visits:
    key_bytes, avg, regret = solver.visits_flat(KEY_WIDTH)
    keys = np.asarray(key_bytes).view(f"S{KEY_WIDTH}").reshape(-1)
    iterations = int(solver.iterations())
    return Visits(keys, np.asarray(avg), np.asarray(regret), iterations,
                  max(0, iterations - int(averaged_from)), dict(meta or {}))


def write_visits(path: str, visits: Visits) -> None:
    # The metadata goes in as a JSON string: an object array would need
    # allow_pickle to read back, and a counts file should not need that.
    np.savez(path, keys=visits.keys, avg_visits=visits.avg, regret_visits=visits.regret,
             iterations=np.int64(visits.iterations),
             averaged_iterations=np.int64(visits.averaged_iterations),
             meta=np.array(json.dumps(visits.meta, default=str)))


def load_visits(path: str) -> Visits:
    with np.load(path, allow_pickle=False) as data:
        return Visits(data["keys"], data["avg_visits"], data["regret_visits"],
                      int(data["iterations"]), int(data["averaged_iterations"]),
                      json.loads(str(data["meta"])))


def street_and_depth(key) -> tuple:
    """
    (street index, raises so far on that street) for a key `bucket|history`.

    Depth is per street because that is what the raise cap counts and what the
    rare tail is made of: the node after a third raise on the flop is rare
    whatever happened preflop.
    """
    if isinstance(key, bytes):
        key = key.rstrip(b"\0").decode()
    history = key.split("|", 1)[1] if "|" in key else key
    street = history.count("/")
    current = history.rsplit("/", 1)[-1]
    return street, sum(1 for c in current if c in RAISE_SYMBOLS)


def classify(keys: np.ndarray) -> tuple:
    """Street and depth arrays for every key, in one pass."""
    streets = np.empty(len(keys), dtype=np.int8)
    depths = np.empty(len(keys), dtype=np.int8)
    for i, k in enumerate(keys):
        streets[i], depths[i] = street_and_depth(k)
    return streets, depths
