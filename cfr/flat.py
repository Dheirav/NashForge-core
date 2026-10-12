"""
A solved strategy as three flat arrays, read like the dict it came from.

The pickle a solver writes holds its strategy as a dict of small numpy arrays,
one per information set. That is about 200 bytes of Python object around a
few bytes of data: the 100bb cap-2 rung is 160 MB on disk, 5.5 s to load and
1,465 MB resident, and the sixteen-rung ladder the arena bot carries is 2.4 GB,
which is what has forced the bot and every other job on this machine to take
turns. Stored as a sorted array of fixed-width keys, an array of row offsets
and one contiguous array of probabilities, the same rung is under 120 MB and
loads in well under a second, and the ladder fits beside a training run.

`FlatStrategy` is a read-only mapping: `get`, `in`, `len` and iteration answer
exactly as the dict does, and a value comes back as a view of the values
array, so `cfr_agent`, the arena and Slumbot players, the play-off script and
the river solver read it unchanged. The pickle stays the source of truth; the
flat file is derived from it by `scripts/cfr/flatten_strategy.py` and sits
beside it as `<name>.flat.npz`, with the abstraction and the run's arguments in
`<name>.flat.pkl`. `load_strategy` prefers the flat pair when it exists.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Dict, Iterator, Optional

import numpy as np

#: Longest information-set key the fixed-width array can hold. The deepest
#: history seen so far is 20 characters with a three-digit bucket; the
#: converter refuses a longer one rather than truncating it.
KEY_WIDTH = 32


class FlatStrategy(Mapping):
    """The dict's contract over three arrays."""

    def __init__(self, keys: np.ndarray, offsets: np.ndarray, values: np.ndarray):
        if keys.dtype.kind != "S":
            raise TypeError("keys must be a fixed-width bytes array")
        self._keys = keys
        self._offsets = offsets
        self._values = values

    def _find(self, key) -> int:
        if isinstance(key, str):
            key = key.encode()
        probe = np.array(key, dtype=self._keys.dtype)
        i = int(np.searchsorted(self._keys, probe))
        if i < len(self._keys) and self._keys[i] == probe:
            return i
        return -1

    def get(self, key, default=None):
        i = self._find(key)
        if i < 0:
            return default
        return self._values[self._offsets[i]:self._offsets[i + 1]]

    def __getitem__(self, key):
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key) -> bool:
        return self._find(key) >= 0

    def __len__(self) -> int:
        return int(len(self._keys))

    def __iter__(self) -> Iterator[str]:
        for k in self._keys:
            yield k.decode()

    @property
    def nbytes(self) -> int:
        return int(self._keys.nbytes + self._offsets.nbytes + self._values.nbytes)


def flatten(strategy: Dict[str, np.ndarray], dtype=np.float64) -> FlatStrategy:
    """The dict as flat arrays, keys sorted so a lookup is a binary search."""
    keys = sorted(strategy)
    if keys and max(len(k) for k in keys) > KEY_WIDTH:
        longest = max(keys, key=len)
        raise ValueError(f"key {longest!r} is longer than KEY_WIDTH={KEY_WIDTH}")
    key_array = np.array([k.encode() for k in keys], dtype=f"S{KEY_WIDTH}")
    lengths = np.fromiter((np.asarray(strategy[k]).size for k in keys), dtype=np.int64, count=len(keys))
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.empty(int(offsets[-1]), dtype=dtype)
    for k, start, stop in zip(keys, offsets[:-1], offsets[1:]):
        values[start:stop] = np.asarray(strategy[k], dtype=dtype)
    return FlatStrategy(key_array, offsets, values)


def flat_paths(pickle_path: str):
    stem = pickle_path[:-4] if pickle_path.endswith(".pkl") else pickle_path
    return stem + ".flat.npz", stem + ".flat.pkl"


def write_flat(pickle_path: str, saved: dict, dtype=np.float64) -> str:
    """Write the flat pair beside the pickle; returns the .npz path."""
    import pickle   # here rather than at the top: the uploaded bot's sandbox refuses the import
    npz, side = flat_paths(pickle_path)
    flat = flatten(saved["strategy"], dtype)
    np.savez(npz, keys=flat._keys, offsets=flat._offsets, values=flat._values)
    with open(side, "wb") as handle:
        pickle.dump({k: v for k, v in saved.items() if k != "strategy"}, handle)
    return npz


def load_strategy(pickle_path: str, prefer_flat: bool = True) -> dict:
    """
    The saved solver: the pickle's dict, or the flat pair when it exists.

    The flat pair is used only when it is newer than the pickle, so a retrained
    rung is never read through a stale cache of its predecessor.
    """
    # A compact table (cfr/pure.py) is what the uploaded bot ships in place of the flat pair.
    from cfr.pure import load_compact
    compact = load_compact(pickle_path)
    if compact is not None:
        return compact
    import pickle
    npz, side = flat_paths(pickle_path)
    if prefer_flat and os.path.exists(npz) and os.path.exists(side) \
            and os.path.getmtime(npz) >= os.path.getmtime(pickle_path):
        with open(side, "rb") as handle:
            saved = pickle.load(handle)
        data = np.load(npz)
        saved["strategy"] = FlatStrategy(data["keys"], data["offsets"], data["values"])
        return saved
    with open(pickle_path, "rb") as handle:
        return pickle.load(handle)
