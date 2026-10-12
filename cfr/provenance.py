"""
Where a solve came from, written beside it when it is trained.

The 170 solves behind our old ladders (7 Oct) each keep their full training
arguments, but none says which code trained it, while the native solver, the
abstraction and the trainer changed 25 times from 15 September; 92 of them were
built on another solve (its buckets or a warm start) that is no more permanent
than they are; and most ran on 2 to 6 threads, so the same code and seed do not
give the same file. A solve that cannot be rebuilt cannot be safely deleted, and
a disk that filled on 7 Oct made that a real question. So every training run now
records, at its start, the commit it ran on (and whether the tree was dirty), the
command, the machine, and a digest of every solve it was built on; and
`scripts/solve_manifest.py` keeps one entry per solve with its files' digests,
its sources and the ladders that use it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import subprocess
import sys
from typing import Dict, Optional

#: The arguments that name another solve this one was built on.
SOURCE_ARGS = ("warm_start", "abstraction_from")


def _git(root: str, *args: str, strip: bool = True) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Porcelain status is column-aligned: stripping the whole output eats the first line's status column.
    return out.stdout.strip() if strip else out.stdout


def code_version(root: str) -> Dict:
    """The commit `root` is at, its branch, and the tracked files that differ from it (a dirty tree trains code
    no commit holds, so the list says exactly what)."""
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None}
    changed = [line[3:] for line in (_git(root, "status", "--porcelain", "--untracked-files=no", strip=False)
                                     or "").splitlines() if line.strip()]
    # The match ledger and the docs change under every run without changing what trains; only code makes it dirty.
    dirty = [f for f in changed if not f.startswith(("results/", "docs/")) and not f.endswith(".md")]
    return {"commit": commit, "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty), "dirty_files": dirty[:50], "data_changed": [f for f in changed if f not in dirty][:20]}


def file_digest(path: str, chunk: int = 1 << 22) -> str:
    """blake2b of a file's bytes, read in chunks (a 1 GB solve in about a second)."""
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sources(args: Dict) -> Dict[str, Dict]:
    """Each solve this one was built on, by the argument that named it: path, size and digest, so a rebuild can
    check it found the same file and not a retrained one under the same name."""
    found = {}
    for name in SOURCE_ARGS:
        path = args.get(name)
        if not path:
            continue
        entry = {"path": os.path.abspath(path)}
        if os.path.isfile(path):
            entry.update(size=os.path.getsize(path), blake2b=file_digest(path))
        else:
            entry["missing"] = True
        found[name] = entry
    return found


def provenance(args: Dict, root: str) -> Dict:
    """Everything a rebuild needs that the arguments do not hold, taken at the start of the run."""
    return {"code": code_version(root), "command": [sys.executable] + sys.argv,
            "cwd": os.getcwd(), "python": platform.python_version(), "host": platform.node(),
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "sources": sources(args)}
