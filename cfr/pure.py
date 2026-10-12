"""
A strategy stored compactly for the uploaded bot: hashed keys and 32-bit probabilities.

The upload track gives a bot 256 MB and a 200 MB image. a playing ladder's rungs run to millions of
information sets, and the flat format spends about 60 bytes on each (a 32-byte key, an 8-byte offset
and 8-byte probabilities), which with Python and numba put the bot at 900 MB. Here a node costs an
8-byte hash of its key, a 1-byte width and 4 bytes a probability, compressed on disk to 24 MB for the
ladder.

Storing only the purified action was tried first, and 22 of 4,799 decisions in a shadow test then
differed from the full bot: purification picks the most probable action among those the arena
allows at that moment, and when the stored action was not allowed there was nothing else to pick.
Keeping the whole row keeps every such choice, and lets the image play mixed or purified exactly as
the main bot does. The row is kept as 32-bit floats, not 16-bit fixed point: rounding to 1/65,535
turned a tiny call probability into zero, and at 1bb, where the arena allows only fold or call, the
row then had nothing legal left and the bot fell back to a rule. Floats keep small values and their
order, and the shadow test then agrees with the full bot on every decision.

The rest of a rung, its card abstraction and training arguments, is a JSON file beside the table
(`write_rung`), because the upload sandbox refuses `pickle`.

The hash replaces the key. The converter refuses a table where two keys share a 64-bit hash, so a
lookup cannot land on another node; a key that is not in the table can still match one by chance, at
1.8e6 / 2^64 per lookup. `CompactTable` answers `get` and `in` as `FlatStrategy` does, so
`cfr_agent` and the player read it unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping

import numpy as np



def key_hash(key) -> int:
    """The stable 64-bit hash a table is indexed by; Python's own `hash` is salted per process."""
    if isinstance(key, str):
        key = key.encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def compact_path(pickle_path: str) -> str:
    stem = pickle_path[:-4] if pickle_path.endswith(".pkl") else pickle_path
    return stem + ".compact.npz"


class CompactTable(Mapping):
    """A Mapping from key to its row of probabilities, over hashed keys and 16-bit values."""

    #: Read by `cfr.purify.apply_to_table`, which must not expand 1.8 million rows into a dict.
    is_compact = True

    def __init__(self, hashes: np.ndarray, offsets: np.ndarray, values: np.ndarray):
        self._hashes, self._offsets, self._values = hashes, offsets, values

    def _find(self, key) -> int:
        h = np.uint64(key_hash(key))
        i = int(np.searchsorted(self._hashes, h))
        return i if i < len(self._hashes) and self._hashes[i] == h else -1

    def get(self, key, default=None):
        i = self._find(key)
        if i < 0:
            return default
        return self._values[self._offsets[i]:self._offsets[i + 1]].astype(np.float64)

    def __getitem__(self, key):
        row = self.get(key)
        if row is None:
            raise KeyError(key)
        return row

    def __contains__(self, key) -> bool:
        return self._find(key) >= 0

    def __len__(self) -> int:
        return len(self._hashes)

    def __iter__(self):
        raise TypeError("a CompactTable holds hashes, not keys; it cannot be iterated")


def compact(strategy) -> CompactTable:
    """Every row of `strategy`, as 32-bit floats, indexed by key hash."""
    hashes, rows = [], []
    for key, probs in strategy.items():
        hashes.append(key_hash(key))
        rows.append(np.asarray(probs, dtype=np.float32))
    hashes = np.array(hashes, dtype=np.uint64)
    order = np.argsort(hashes, kind="stable")
    hashes = hashes[order]
    if len(hashes) > 1 and np.any(hashes[1:] == hashes[:-1]):
        raise ValueError("two keys share a 64-bit hash; this table cannot be stored by hash")
    rows = [rows[i] for i in order]
    offsets = np.zeros(len(rows) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(r) for r in rows])
    values = np.concatenate(rows) if rows else np.zeros(0, dtype=np.float32)
    return CompactTable(hashes, offsets, values)


def _offsets(widths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(widths) + 1, dtype=np.uint32)
    np.cumsum(widths, out=offsets[1:], dtype=np.uint32)
    return offsets


def write_compact(pickle_path: str, table: CompactTable) -> str:
    path = compact_path(pickle_path)
    # Widths rather than offsets (a byte a node against four) and compressed: the built image is capped
    # at 200 MB and this takes a ladder's tables from 41 to 24 MB. The offsets are rebuilt when it loads.
    widths = np.diff(table._offsets).astype(np.uint8)
    np.savez_compressed(path, hashes=table._hashes, widths=widths, values=table._values)
    return path


def rung_path(pickle_path: str) -> str:
    stem = pickle_path[:-4] if pickle_path.endswith(".pkl") else pickle_path
    return stem + ".rung.json"


# The rest of a rung (its card abstraction and training arguments) goes to JSON rather than a pickle:
# the upload sandbox refuses `pickle`, and a JSON file is also one a reviewer can read. Tuples, tuple
# keys and arrays are tagged so they come back as they went in.
def _encode(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {"__dict__": {k: _encode(v) for k, v in value.items()}}
        return {"__pairs__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot store a {type(value).__name__} in a rung file")


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if "__array__" in value:
            return np.array(value["__array__"], dtype=value["dtype"])
        if "__tuple__" in value:
            return tuple(_decode(v) for v in value["__tuple__"])
        if "__dict__" in value:
            return {k: _decode(v) for k, v in value["__dict__"].items()}
        if "__pairs__" in value:
            return {_decode(k): _decode(v) for k, v in value["__pairs__"]}
    return value


def write_rung(pickle_path: str, saved: dict) -> str:
    """The abstraction's attributes and the training arguments, as JSON beside the compact table."""
    path = rung_path(pickle_path)
    abstraction = saved["abstraction"]
    doc = {"abstraction_class": type(abstraction).__name__,
           "abstraction": _encode(dict(vars(abstraction))),
           "args": _encode(dict(saved["args"]) if isinstance(saved["args"], dict) else dict(vars(saved["args"])))}
    with open(path, "w") as handle:
        json.dump(doc, handle)
    return path


def load_compact(pickle_path: str):
    """The rung's abstraction and arguments with its compact table, or None without one."""
    path, side = compact_path(pickle_path), rung_path(pickle_path)
    if not (os.path.exists(path) and os.path.exists(side)):
        return None
    with open(side) as handle:
        doc = json.load(handle)
    from abstraction.buckets import CardAbstraction
    if doc["abstraction_class"] != "CardAbstraction":
        raise ValueError(f"{side}: a {doc['abstraction_class']} cannot be rebuilt here")
    abstraction = CardAbstraction.__new__(CardAbstraction)
    abstraction.__dict__.update(_decode(doc["abstraction"]))
    data = np.load(path, allow_pickle=False)
    return {"abstraction": abstraction, "args": _decode(doc["args"]),
            "strategy": CompactTable(data["hashes"], _offsets(data["widths"]), data["values"])}
