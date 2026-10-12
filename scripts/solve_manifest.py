"""
One entry per solve: its files and their digests, its training arguments, the code that trained it, what it was
built on and what was built on it, and the ladders that use it.

    venv/bin/python scripts/solve_manifest.py                  # every solve under results/cfr, incremental
    venv/bin/python scripts/solve_manifest.py --only results/cfr/experiments/x.pkl     # one solve, as the trainer does
    venv/bin/python scripts/solve_manifest.py --deletable      # the leaves: solves nothing live or built on uses

It never deletes anything, and nothing should delete a solve on its own: `--deletable` is a list to show the
user, and a solve goes only with their explicit go-ahead for that deletion (CLAUDE.md).

Why (7 Oct): the 170 solves behind our old ladders kept their arguments but not their code, 92 were built on a
solve that could be deleted with them, and C: was full. A solve may be deleted only if it can be rebuilt, so the
manifest says, for each one, what a rebuild needs, and which deletions would strand another solve's chain.

Solves trained since `cfr/provenance.py` carry their commit; older ones get the last commit on any branch before
their run started (file time minus the run's seconds), marked `inferred`, which is the best the record allows and
is wrong for a run from a dirty tree. Digests are reused while a file's size and time are unchanged, so a rerun
hashes only what changed; the first full run reads every file once and prints its progress with an ETA.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cfr.provenance import SOURCE_ARGS, file_digest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFR = os.path.join(ROOT, "results", "cfr")
OUT = os.path.join(CFR, "MANIFEST.json")
SUFFIXES = (".pkl", ".json", ".flat.pkl", ".flat.npz", ".visits.npz", ".partial.pkl", ".partial.json")
VISITS = re.compile(r"\.visits\.\d+\.npz$")
#: Ladders the bot plays or a lane measures today; a solve under one of these is never deletable.
LIVE_DEFAULT = ("ladder_live",)


def stem_of(path: str) -> str:
    path = VISITS.sub("", path)
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def rel(path: str) -> str:
    path = os.path.realpath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def solve_files(stem: str):
    return sorted(f for f in glob.glob(glob.escape(stem) + ".*")
                  if os.path.isfile(f) and not os.path.islink(f) and stem_of(f) == stem)


def all_stems():
    """Every solve stored under results/cfr as a real file (links in ladders point at these)."""
    found = set()
    for path in glob.glob(os.path.join(CFR, "**", "*.pkl"), recursive=True):
        if os.path.islink(path) or path.endswith((".flat.pkl", ".partial.pkl")):
            continue
        found.add(stem_of(os.path.realpath(path)))
    return found


def inferred_commit(started: float):
    when = dt.datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds")
    out = subprocess.run(["git", "-C", ROOT, "log", "--all", f"--before={when}", "-1", "--format=%H %cI %s"],
                         capture_output=True, text=True)
    line = out.stdout.strip()
    if not line:
        return None
    commit, date, subject = line.split(" ", 2)
    return {"commit": commit, "committed": date, "subject": subject[:100], "inferred": True,
            "basis": "the last commit on any branch before the run started (file time minus its seconds)"}


def ladders_by_target():
    """Which ladder (results/cfr/*/ and the scratch lanes' lad_* dirs) links each solve file."""
    users = {}
    dirs = glob.glob(os.path.join(CFR, "*", "")) + glob.glob(os.path.expanduser("~/pokerbot-scratch/*/lad_*/"))
    for directory in dirs:
        name = rel(directory.rstrip("/")) if directory.startswith(CFR) else directory.rstrip("/")
        for link in glob.glob(os.path.join(directory, "*")):
            if os.path.islink(link):
                users.setdefault(stem_of(os.path.realpath(link)), set()).add(name)
    return users


def entry_for(stem: str, old: dict, progress) -> dict:
    files = {}
    for f in solve_files(stem):
        st = os.stat(f)
        key = f[len(stem):]
        prev = (old.get("files") or {}).get(key) or {}
        if prev.get("size") == st.st_size and prev.get("mtime") == int(st.st_mtime) and prev.get("blake2b"):
            digest = prev["blake2b"]
        else:
            digest = file_digest(f)
            progress(st.st_size)
        files[key] = {"size": st.st_size, "mtime": int(st.st_mtime), "blake2b": digest}
    summary = {}
    if os.path.exists(stem + ".json"):
        with open(stem + ".json") as handle:
            summary = json.load(handle)
    args = summary.get("args") or {}
    seconds = summary.get("seconds") or 0
    made = files.get(".pkl", files.get(".json", {})).get("mtime") or 0
    code = (summary.get("provenance") or {}).get("code")
    if not code or not code.get("commit"):
        code = inferred_commit(made - seconds) if made else None
    sources = {}
    for name in SOURCE_ARGS:
        if args.get(name):
            path = args[name] if os.path.isabs(args[name]) else os.path.join(ROOT, args[name])
            sources[name] = {"stem": rel(stem_of(path)), "present": os.path.exists(path)}
    return {"files": files, "bytes": sum(v["size"] for v in files.values()), "args": args,
            "results": {"information_sets_reached": summary.get("information_sets_reached"),
                        "iterations_completed": summary.get("iterations_completed"), "seconds": seconds},
            "code": code, "recorded": bool((summary.get("provenance") or {}).get("code", {}).get("commit")),
            "command": (summary.get("provenance") or {}).get("command"), "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="solve files (any of their files) to (re)write, instead of all")
    parser.add_argument("--live", nargs="+", default=list(LIVE_DEFAULT), help="ladders whose solves are never deletable")
    parser.add_argument("--deletable", action="store_true", help="list the leaves that could go, and stop")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    manifest = {}
    if os.path.exists(args.out):
        with open(args.out) as handle:
            manifest = json.load(handle).get("solves", {})
    stems = {stem_of(os.path.realpath(p)) for p in args.only} if args.only else all_stems()
    if not args.deletable:
        todo = sum(os.path.getsize(f) for s in stems for f in solve_files(s)
                   if not (((manifest.get(rel(s)) or {}).get("files") or {}).get(f[len(s):], {}).get("size")
                           == os.path.getsize(f)
                           and ((manifest.get(rel(s)) or {}).get("files") or {}).get(f[len(s):], {}).get("mtime")
                           == int(os.path.getmtime(f))))
        start, done = time.time(), [0]

        def progress(n):
            done[0] += n
            rate = done[0] / max(time.time() - start, 1e-6)
            left = (todo - done[0]) / rate if rate else 0
            print(f"  hashed {done[0] / 1e9:6.1f} of {todo / 1e9:6.1f} GB, {rate / 1e6:5.0f} MB/s, "
                  f"about {left / 60:4.1f} min left (from the measured rate)", flush=True)
        for i, stem in enumerate(sorted(stems)):
            manifest[rel(stem)] = entry_for(stem, manifest.get(rel(stem)) or {}, progress)
        for key in [k for k in manifest if not args.only and not os.path.exists(os.path.join(ROOT, k + ".pkl"))
                    and not os.path.exists(k + ".pkl")]:
            manifest[key]["gone"] = True
    users = ladders_by_target()
    for key, entry in manifest.items():
        full = os.path.join(ROOT, key) if not os.path.isabs(key) else key
        entry["ladders"] = sorted(users.get(os.path.realpath(full), ()))
        entry["built_on_by"] = []
    for key, entry in manifest.items():
        for source in (entry.get("sources") or {}).values():
            if source["stem"] in manifest:
                manifest[source["stem"]]["built_on_by"].append(key)
    live = set(args.live)
    for entry in manifest.values():
        entry["live"] = any(os.path.basename(l) in live for l in entry["ladders"])
    if args.deletable:
        leaves = [(k, e) for k, e in manifest.items() if not e.get("gone") and not e["live"] and not e["built_on_by"]]
        print(f"{len(leaves)} leaves, {sum(e['bytes'] for _, e in leaves) / 1e9:.1f} GB (keep each .json):")
        for k, e in sorted(leaves, key=lambda kv: -kv[1]["bytes"]):
            print(f"  {e['bytes'] / 1e9:5.2f} GB  {k}  ladders: {', '.join(os.path.basename(l) for l in e['ladders']) or '-'}")
        return
    with open(args.out + ".tmp", "w") as handle:
        json.dump({"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "live_ladders": sorted(live), "solves": manifest}, handle, indent=1)
    os.replace(args.out + ".tmp", args.out)
    present = [e for e in manifest.values() if not e.get("gone")]
    print(f"{args.out}: {len(present)} solves, {sum(e['bytes'] for e in present) / 1e9:.1f} GB; code recorded for "
          f"{sum(e['recorded'] for e in present)}, inferred for {sum(not e['recorded'] for e in present)}; "
          f"live {sum(e['live'] for e in present)}, built on by another {sum(bool(e['built_on_by']) for e in present)}")


if __name__ == "__main__":
    main()
