"""
Pool paired LBR chunks: the arm off, the arm on, and the difference.

The arm is whichever option the chunks were paired on, --offtree-third-raise or
--bridge-translation, so the columns name the arm rather than the probes.

    venv/bin/python scripts/lbr_paired_report.py results/cfr/lbr_thirdraise_2026-10-06/*.json

Each chunk comes from `scripts/lbr_ladder.py --offtree-third-raise ... --paired` (or
`--bridge-translation --paired`)
and holds both arms over the same per-hand seeds. The difference is pooled from
its per-hand sums, not by averaging chunk means, so the standard error is the
paired one: hands where no probe was taken differ by exactly zero and add no
noise. That is why the difference can be tighter than either arm's own error.

The arms themselves are pooled the usual way, as the 25 Sept chunks were.
A positive difference clear of zero says the all-in-only hole is worth that
much to LBR; like every LBR number it is a lower bound, never a ceiling.
"""
import argparse
import json
import os
import re
import math
from collections import defaultdict


def pooled(chunks, key):
    """Mean and stderr over equal-or-unequal chunks, weighting each by its hands."""
    total = sum(c["hands"] for c in chunks)
    mean = sum(c[key]["mean"] * c["hands"] for c in chunks) / total
    se = math.sqrt(sum((c[key]["stderr"] * c["hands"]) ** 2 for c in chunks)) / total
    return mean, se, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    by_rung = defaultdict(list)
    for path in args.paths:
        with open(path) as handle:
            for rung, row in json.load(handle).items():
                if not row.get("paired"):
                    print(f"{path}: {rung} is not a paired run, skipped")
                    continue
                row["on"] = {"mean": row["mean"], "stderr": row["stderr"]}
                # Grouped by the chunk's condition name as well: every ladder
                # calls its rung cap2_70bb.pkl, so the rung alone merges two solves.
                condition = re.sub(r"_\d+\.json$", "", os.path.basename(path))
                by_rung[(condition, rung, row.get("resolved", row["path"]))].append(row)

    print("| condition | rung | solve | chunks | hands | arm off, BB/100 | arm on, BB/100 | on minus off, BB/100 "
          "| hands changed | probes | bot folds to a probe | re-read lookups hit / missed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (condition, rung, solve), chunks in sorted(by_rung.items()):
        big_blind = chunks[0]["big_blind"]
        per100 = 100.0 / big_blind
        off_mean, off_se, hands = pooled(chunks, "off")
        on_mean, on_se, _ = pooled(chunks, "on")
        n = sum(c["difference"]["n"] for c in chunks)
        s = sum(c["difference"]["sum"] for c in chunks)
        ss = sum(c["difference"]["sumsq"] for c in chunks)
        d_mean = s / n
        d_se = math.sqrt(max(ss / n - d_mean ** 2, 0.0) * n / (n - 1) / n) if n > 1 else float("nan")
        changed = sum(c["difference"]["hands_changed"] for c in chunks)
        stats = defaultdict(int)
        for c in chunks:
            for k, v in c["probe_stats"].items():
                stats[k] += v
        answered = stats["folded"] + stats["called"]
        folds = f"{100 * stats['folded'] / answered:.0f}%" if answered else "n/a"
        print(f"| {condition} | {rung} | {solve} | {len(chunks)} | {hands:,} | {off_mean * per100:+.1f} ± {off_se * per100:.1f} "
              f"| {on_mean * per100:+.1f} ± {on_se * per100:.1f} | **{d_mean * per100:+.1f} ± {d_se * per100:.1f}** "
              f"| {changed:,} | {stats['probes']:,} | {folds} | {stats['alt_hits']:,} / {stats['alt_misses']:,} |")


if __name__ == "__main__":
    main()
