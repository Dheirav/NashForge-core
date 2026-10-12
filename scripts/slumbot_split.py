"""
Where the Slumbot loss comes from: by miss, by final street, by position.

    venv/bin/python scripts/slumbot_split.py results/slumbot/contender_cap2_200bb_1k.json
    venv/bin/python scripts/slumbot_split.py results/slumbot/m1.json --min-hands 50

Always-fold loses 750 mbb/hand heads-up and the record on file is −997 ± 396, so
at the mean the bot is worse than folding every hand: it is bleeding chips with
the hands it plays. A whole-run win rate cannot say which hands. This reads the
per-hand records the pilot and the measure now write (`hand_records`, or the
pilot's `hands`) and splits the mean by whether the solver ever missed a lookup
in the hand, by the street the hand ended on, by position and by showdown.

Each split carries its own interval. A cell with a few dozen hands has an
interval of thousands of mbb/hand and says nothing; the point is the
comparison between cells with hundreds each, and even that is a lead to chase
rather than a result until the 10,000-hand run repeats it.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

BIG_BLIND = 100
STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def interval(values):
    series = np.asarray(values, dtype=float)
    if len(series) < 2:
        return series.mean() / BIG_BLIND * 1000 if len(series) else float("nan"), float("nan")
    mean = series.mean() / BIG_BLIND * 1000
    half = 1.96 * series.std(ddof=1) / np.sqrt(len(series)) / BIG_BLIND * 1000
    return mean, half


def rows_of(path):
    with open(path) as handle:
        saved = json.load(handle)
    rows = saved.get("hand_records") or saved.get("hands") or []
    rows = [r for r in rows if r.get("w") is not None or r.get("winnings") is not None]
    for r in rows:
        r.setdefault("w", r.get("winnings"))
        r.setdefault("pos", r.get("client_pos"))
        if "street" not in r and "action" in r:
            r["street"] = r["action"].count("/")
    return saved, rows


def split(rows, key, label, min_hands):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r["w"])
    lines = [f"| {label} | hands | mbb/hand | ± | share of the loss |", "|---|---|---|---|---|"]
    total = sum(r["w"] for r in rows)
    for name, values in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(values) < min_hands:
            lines.append(f"| {name} | {len(values)} | too few | | |")
            continue
        mean, half = interval(values)
        share = sum(values) / total if total else float("nan")
        lines.append(f"| {name} | {len(values)} | {mean:+.0f} | {half:.0f} | {100 * share:.0f}% |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result")
    parser.add_argument("--min-hands", type=int, default=30, help="cells smaller than this are not averaged")
    parser.add_argument("--out", help="write the tables here as markdown as well")
    args = parser.parse_args()

    saved, rows = rows_of(args.result)
    if not rows:
        sys.exit(f"{args.result} has no per-hand records; it predates the split "
                 "(runs from 15 September 2026 on carry them)")
    mean, half = interval([r["w"] for r in rows])
    lines = [f"# Slumbot loss split: {os.path.basename(args.result)}", "",
             f"{len(rows)} hands, {mean:+.0f} ± {half:.0f} mbb/hand overall; "
             f"strategy {saved.get('strategy')}, schedule {saved.get('schedule')}, "
             f"purify {saved.get('purify', 'none')}. Always-fold is −750.", ""]
    has_miss = any("miss" in r for r in rows)
    if has_miss:
        lines += split(rows, lambda r: "missed a lookup" if r["miss"] else "no miss", "lookup", args.min_hands) + [""]
        lines += split(rows, lambda r: "schedule miss" if r.get("sched_miss") else "on schedule", "re-raises", args.min_hands) + [""]
    lines += split(rows, lambda r: STREETS.get(r.get("street"), "?"), "ended on", args.min_hands) + [""]
    lines += split(rows, lambda r: "big blind (pos 0)" if r["pos"] == 0 else "button (pos 1)", "position", args.min_hands) + [""]
    if any("showdown" in r for r in rows):
        lines += split(rows, lambda r: "showdown" if r.get("showdown") else "no showdown", "ending", args.min_hands) + [""]
    if has_miss:
        lines += split(rows, lambda r: f"{STREETS.get(r.get('street'), '?')}, {'miss' if r['miss'] else 'hit'}",
                       "street x lookup", args.min_hands) + [""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
