"""
How many samples each information set's strategy is made of, and whether a rung is done.

    venv/bin/python scripts/cfr/visit_report.py RUN.visits.npz
    venv/bin/python scripts/cfr/visit_report.py RUN_2M.visits.npz RUN_4M.visits.npz
    venv/bin/python scripts/cfr/visit_report.py T.visits.npz 2T.visits.npz \\
        --h2h-delta 0.4 --h2h-se 0.7 --lbr-delta -1.2 --lbr-se 4.5

The counts come from `scripts/cfr/train_nolimit.py --count-visits`. The budget
rule until now was iterations over reached points, an average; this prints the
distribution behind it, by street and by raise depth on the street, because the
rare tail is where an average hides the nodes that never converged.

`avg` is the number of samples in a node's average strategy (the owner's own
reach times chance, times T). `regret` is the number of regret updates (the
opponent's reach times chance, times T). The reach-weighted columns weight
each node by avg x regret, which is proportional to its probability of being
reached in play times the chance probability of the owner's bucket: a proxy,
off by that factor, so it ranks the tail by how often it is played without
pretending to be exact. Node-weighted columns count every reached node once,
which is the pessimist's view: most of the deep tail is almost never played.

With two snapshots of the same tree (T and 2T) it also prints the ratio of
counts on the nodes both reached. Where reach has settled the ratio is the
ratio of iterations; where it is still moving the ratio drifts, which is a
free second signal that a street has not settled.

The stopping rule (docs/research/2026-10-05-visit-counter.md) is checked on
the later snapshot when its instruments are given:
  (a) on the worst street-and-depth line, the reach-weighted 10th percentile
      of avg visits at or above --bar;
  (b) doubling the iterations moves same-tree head to head by under 1 point
      (`scripts/chipzen_duel.py --arena-matches 5000`, match-win percentage
      points, about +/- 0.7 at 5,000), given as --h2h-delta and --h2h-se;
  (c) paired LBR, T against 2T on the same seeds (`scripts/lbr_ladder.py`,
      pooled with `scripts/lbr_paired_report.py`), moves by under one standard
      error, given as --lbr-delta and --lbr-se.
A criterion with no measurement is reported as not measured and the verdict is
not "done": a rule with a missing leg is not a passed rule.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from cfr.visits import STREETS, classify, load_visits  # noqa: E402

PERCENTILES = (1, 10, 25, 50, 90)
THRESHOLDS = (10, 100, 1000)
#: Provisional bar for (a), in average-strategy samples at the reach-weighted
#: 10th percentile. An average of n draws has a per-action standard error up to
#: 0.5 / sqrt(n); at 100 that is 0.05, about the smallest mixing error the
#: head to head has ever resolved. To be replaced by the level the mid rungs had
#: at 20M, where polishing to 60M tied (the 5 Oct finding).
DEFAULT_BAR = 100


def weighted_percentile(values, weights, q):
    """The smallest value v with at least q percent of the weight at or below v."""
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    at = np.searchsorted(cumulative, q / 100.0 * cumulative[-1], side="left")
    return float(values[order][min(at, len(values) - 1)])


def summarise(avg, regret):
    """One row: the distribution of avg visits over a set of nodes."""
    avg = avg.astype(np.float64)
    regret = regret.astype(np.float64)
    row = {"nodes": int(len(avg))}
    if len(avg) == 0:
        return row
    for q in PERCENTILES:
        row[f"p{q}"] = float(np.percentile(avg, q, method="lower"))
    for t in THRESHOLDS:
        row[f"under_{t}"] = float(np.mean(avg < t))
    row["zero"] = float(np.mean(avg == 0))
    row["regret_p10"] = float(np.percentile(regret, 10, method="lower"))
    row["regret_p50"] = float(np.percentile(regret, 50, method="lower"))
    weight = avg * regret
    row["rw_p10"] = weighted_percentile(avg, weight, 10)
    row["rw_p50"] = weighted_percentile(avg, weight, 50)
    total = weight.sum()
    for t in THRESHOLDS:
        row[f"rw_under_{t}"] = float(weight[avg < t].sum() / total) if total > 0 else float("nan")
    return row


def breakdown(visits):
    """Rows for all nodes, by street, and by street and raise depth."""
    streets, depths = classify(visits.keys)
    rows = {"all": summarise(visits.avg, visits.regret)}
    for s, name in enumerate(STREETS):
        on = streets == s
        if not on.any():
            continue
        rows[name] = summarise(visits.avg[on], visits.regret[on])
        for d in sorted(set(depths[on].tolist())):
            sel = on & (depths == d)
            rows[f"{name} d{d}"] = summarise(visits.avg[sel], visits.regret[sel])
    return rows


def ratios(a, b):
    """B over A on the nodes both reached, by street: median and interquartile range."""
    common, ia, ib = np.intersect1d(a.keys, b.keys, assume_unique=True, return_indices=True)
    streets, _ = classify(common)
    out = {"common": int(len(common)), "only_in_b": int(len(b) - len(common)),
           "only_in_a": int(len(a) - len(common)),
           "iteration_ratio": (b.averaged_iterations / a.averaged_iterations) if a.averaged_iterations else float("nan")}
    va = a.avg[ia].astype(np.float64)
    vb = b.avg[ib].astype(np.float64)
    keep = va > 0
    for s, name in [(-1, "all")] + list(enumerate(STREETS)):
        sel = keep if s < 0 else keep & (streets == s)
        if not sel.any():
            continue
        r = vb[sel] / va[sel]
        out[name] = {"median": float(np.median(r)), "q25": float(np.percentile(r, 25)),
                     "q75": float(np.percentile(r, 75)), "nodes": int(sel.sum())}
    return out


def rule(rows, args):
    """The three legs of the stopping rule on the latest snapshot's rows."""
    legs = {}
    # Judged on the worst street-and-depth line, not on the whole tree: over
    # all nodes the reach weight sits on the first decisions, and on the 12bb
    # demo the whole-tree figure was 26,022 while the river after two raises
    # read 812. A rule that the busy lines pass for the rare ones is the
    # average this counter was built to look past.
    lines = {name: r["rw_p10"] for name, r in rows.items()
             if " d" in name and r.get("nodes", 0) and not np.isnan(r.get("rw_p10", float("nan")))}
    worst = min(lines, key=lines.get) if lines else "all"
    p10 = lines.get(worst, rows["all"].get("rw_p10", float("nan")))
    legs["a_visits"] = {"value": p10, "bar": args.bar, "line": worst, "pass": bool(p10 >= args.bar),
                        "what": f"worst line ({worst}) reach-weighted p10 of avg visits {p10:,.0f} "
                                f"against a bar of {args.bar:,.0f}"}
    if args.h2h_delta is None or args.h2h_se is None:
        # A delta without its standard error says nothing about whether a sub-point move was resolvable, so it
        # is not measured, as leg (c) treats a missing SE; otherwise "done" could rest on a width nobody gave.
        legs["b_h2h"] = {"pass": None, "value": args.h2h_delta,
                         "what": "head to head T against 2T not measured (scripts/chipzen_duel.py; "
                                 "--h2h-delta and --h2h-se are both needed)"}
    else:
        ok = abs(args.h2h_delta) < 1.0
        note = "" if args.h2h_se <= 0.7 else f"; its SE {args.h2h_se} is wider than 5,000 matches give"
        legs["b_h2h"] = {"pass": None if note else bool(ok), "value": args.h2h_delta, "se": args.h2h_se,
                         "what": f"2T moved head to head by {args.h2h_delta:+.1f} points (limit 1){note}"}
    if args.lbr_delta is None or args.lbr_se is None:
        legs["c_lbr"] = {"pass": None, "what": "paired LBR T against 2T not measured (scripts/lbr_ladder.py --paired seeds)"}
    else:
        ok = abs(args.lbr_delta) < args.lbr_se
        legs["c_lbr"] = {"pass": bool(ok), "value": args.lbr_delta, "se": args.lbr_se,
                         "what": f"paired LBR moved {args.lbr_delta:+.1f} against one SE of {args.lbr_se:.1f}"}
    passes = [leg["pass"] for leg in legs.values()]
    verdict = ("done" if all(p is True for p in passes)
               else "not done" if any(p is False for p in passes) else "not measured")
    return legs, verdict


def _fmt(v):
    if isinstance(v, float) and np.isnan(v):
        return "-"
    return f"{v:,.0f}"


def _pct(v):
    return "-" if np.isnan(v) else f"{100 * v:.2f}%"


def print_rows(title, visits, rows):
    print(f"\n{title}: {len(visits):,} reached nodes, {visits.averaged_iterations:,} averaged iterations "
          f"({visits.iterations:,} on the solver's counter)")
    head = (f"  {'':<14}{'nodes':>9}" + "".join(f"{'p' + str(q):>8}" for q in PERCENTILES)
            + "".join(f"{'<' + str(t):>7}" for t in THRESHOLDS) + f"{'zero':>7}{'rw p10':>9}{'rw p50':>9}"
            + "".join(f"{'rw<' + str(t):>8}" for t in THRESHOLDS))
    print(head)
    for name, r in rows.items():
        if r["nodes"] == 0:
            continue
        line = (f"  {name:<14}{r['nodes']:>9,}" + "".join(f"{_fmt(r['p' + str(q)]):>8}" for q in PERCENTILES)
                + "".join(f"{100 * r['under_' + str(t)]:>6.1f}%" for t in THRESHOLDS)
                + f"{100 * r['zero']:>6.1f}%{_fmt(r['rw_p10']):>9}{_fmt(r['rw_p50']):>9}"
                + "".join(f"{_pct(r['rw_under_' + str(t)]):>8}" for t in THRESHOLDS))
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", help="a .visits.npz (the earlier snapshot when two are given)")
    parser.add_argument("second", nargs="?", help="a later snapshot of the same tree")
    parser.add_argument("--bar", type=float, default=DEFAULT_BAR,
                        help=f"rule (a): reach-weighted p10 of avg visits must reach this (default {DEFAULT_BAR}, provisional)")
    parser.add_argument("--h2h-delta", type=float, help="rule (b): 2T minus T, match-win points, scripts/chipzen_duel.py")
    parser.add_argument("--h2h-se", type=float, help="its standard error")
    parser.add_argument("--lbr-delta", type=float, help="rule (c): paired LBR, 2T minus T, scripts/lbr_ladder.py")
    parser.add_argument("--lbr-se", type=float, help="its paired standard error")
    parser.add_argument("--json", help="write every row and the verdict here")
    args = parser.parse_args()

    a = load_visits(args.first)
    out = {"first": {"path": args.first, "rows": breakdown(a)}}
    print_rows(args.first, a, out["first"]["rows"])
    latest = out["first"]["rows"]
    if args.second:
        b = load_visits(args.second)
        out["second"] = {"path": args.second, "rows": breakdown(b)}
        print_rows(args.second, b, out["second"]["rows"])
        latest = out["second"]["rows"]
        r = ratios(a, b)
        out["ratios"] = r
        print(f"\nSecond over first on the {r['common']:,} nodes both reached ({r['only_in_b']:,} only in the second, "
              f"{r['only_in_a']:,} only in the first); iterations rose {r['iteration_ratio']:.2f}x:")
        for name in ["all"] + list(STREETS):
            if name in r:
                q = r[name]
                print(f"  {name:<8} median {q['median']:.2f}x  (middle half {q['q25']:.2f} to {q['q75']:.2f})  "
                      f"over {q['nodes']:,} nodes")
    legs, verdict = rule(latest, args)
    out["rule"] = {"legs": legs, "verdict": verdict}
    print("\nStopping rule on the " + ("second" if args.second else "only") + " snapshot:")
    for name, leg in legs.items():
        state = {True: "pass", False: "FAIL", None: "not measured"}[leg["pass"]]
        print(f"  ({name[0]}) {state:<13} {leg['what']}")
    print(f"  verdict: {verdict}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
