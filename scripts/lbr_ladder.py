"""
Local Best Response against the rungs we actually play.

    venv/bin/python scripts/lbr_ladder.py --rungs 70bb 100bb --hands 2000

Exploitability is the measure that matters and we have never had one for the
arena ladder. `cfr/lbr.py` has implemented LBR (Lisy and Bowling 2017) since
early September and the only run on record, `results/cfr/lbr_valuation_remeasure.json`,
was against solvers with 25,089 information sets. a 70bb rung of the current ladder has over a million.
So every improvement argued about this week was argued without knowing where
the blueprint leaks.

Read the sign carefully, because it is asymmetric. A clearly positive number
proves the rung is exploitable by at least that much, and the bet sizes that
produced it say where. A number near zero proves nothing at all: it means this
particular greedy exploiter failed, not that the rung is near equilibrium. A
negative number means the bound is slack and the run was wasted.

The opponent bets sizes that sit between ours on purpose, which is the attack
our pseudo-harmonic translation is supposed to absorb and which nothing has
ever pointed at us.

    venv/bin/python scripts/lbr_ladder.py --rungs 70bb --offtree-third-raise 0.4 0.6 1.0 --paired

`--offtree-third-raise` lets LBR raise those pot fractions where the tree offers
only all-in (the third raise of a street on a (4, 2, 1) rung), read the way the
live bridge reads them. `--paired` plays the run twice in one process, probes
off then on, with every hand's cards and decisions drawn from generators seeded
by the hand's index, so the two differ only on hands where a probe was taken.
The difference is reported hand by hand, which is far tighter than two
independent means; `scripts/lbr_paired_report.py` pools it over chunks.

    venv/bin/python scripts/lbr_ladder.py --rungs 70bb --between-sizes --bridge-translation --paired

`--bridge-translation` reads LBR's own off-tree sizes the way chipzen/bridge.py
reads them, so an overbet at 1.5 times the largest tree size or more is heard as
the pseudo all-in the bridge hears, not as 2x pot. With `--paired` the off arm
drops every option that changes what the bot hears (this one and the probes), so
the difference is what those options are worth together.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.betting import schedule_from_args  # noqa: E402
from cfr.flat import load_strategy  # noqa: E402
from cfr.lbr import LocalBestResponse, lbr_value  # noqa: E402
from games.nolimit import NoLimitHoldem  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#: Sizes that fall between the abstraction's, the way Modicum's LBR opponent
#: bet 0.33 x 2^x of the pot to sit inside the gaps.
BETWEEN = (0.33, 0.66, 1.33, 2.66, 5.33)


def summary(result, hands, big_blind):
    """One arm of a paired run, in the fields a plain run reports."""
    return {"mean": result.mean, "stderr": result.stderr, "ci95": list(result.ci95),
            "proves_exploitable": bool(result.proves_exploitable), "hands": hands,
            "bb_per_100": 100.0 * result.mean / big_blind}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", default=os.path.join(ROOT, "results", "cfr", "ladder_a"))
    parser.add_argument("--rungs", nargs="+", default=["70bb"])
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--rollout-samples", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--between-sizes", action="store_true",
                        help="give the exploiter sizes that sit between ours, the translation attack")
    parser.add_argument("--sizes", nargs="+", type=float,
                        help="the exploiter's bet sizes as pot fractions, e.g. 0.5 1 2 for the taper tree's own; "
                             "a node that lacks one still translates it")
    parser.add_argument("--purify", default="none",
                        help="measure the strategy as played under this purification (cfr/purify.py): none, postflop, all, tNN")
    parser.add_argument("--offtree-third-raise", nargs="+", type=float, default=(),
                        metavar="FRACTION",
                        help="pot fractions LBR may raise where the tree's menu is all-in only, "
                             "translated as chipzen/bridge.py translates them")
    parser.add_argument("--bridge-translation", action="store_true",
                        help="translate LBR's off-tree sizes as chipzen/bridge.py does, pseudo all-in included "
                             "(see cfr/lbr.py); off reproduces every earlier number")
    parser.add_argument("--paired", action="store_true",
                        help="with --offtree-third-raise or --bridge-translation: run without and with them "
                             "over the same per-hand seeds and report the paired difference")
    parser.add_argument("--trace", help="write one record per sized bet LBR makes to this .jsonl (see cfr/lbr.py)")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "cfr", "lbr_ladder.json"))
    args = parser.parse_args()
    treated = bool(args.offtree_third_raise) or args.bridge_translation
    if args.paired and not treated:
        parser.error("--paired compares an option off and on, so it needs --offtree-third-raise "
                     "or --bridge-translation")
    options = {}
    if args.offtree_third_raise:
        options["offtree_third_raise"] = args.offtree_third_raise
    if args.bridge_translation:
        options["bridge_translation"] = True

    rows = {}
    for rung in args.rungs:
        matches = [p for p in glob.glob(os.path.join(args.ladder, f"*_{rung}.pkl")) if ".flat." not in p]
        if not matches:
            print(f"{rung}: not in {args.ladder}")
            continue
        # A ladder directory holds both the cap-2 rungs the bot plays under
        # --deep-primary and the older one-raise `nolimit_*` solves beside them.
        # Taking the first match measured nolimit_70bb.pkl, 131,693 information
        # sets, when the 70bb rung has over a million: an exploitability number for
        # a solver we retired. Prefer what the bot actually plays, and say so.
        matches.sort(key=lambda q: (not os.path.basename(q).startswith("cap2_"), q))
        path = matches[0]
        if len(matches) > 1:
            others = ", ".join(os.path.basename(q) for q in matches[1:])
            print(f"{rung}: using {os.path.basename(path)} (also present: {others})")
        saved = load_strategy(path)
        if args.purify != "none":
            from cfr.purify import apply_to_table
            saved["strategy"] = apply_to_table(saved["strategy"], args.purify)
        saved_args = saved["args"]
        # Through schedule_from_args: a per-street dict read with tuple() became its street names.
        cap = schedule_from_args(saved_args["raise_cap"])
        game = NoLimitHoldem(saved["abstraction"], starting_stack=int(saved_args["stack"]),
                             small_blind=int(saved_args["big_blind"]) // 2,
                             big_blind=int(saved_args["big_blind"]), raise_cap=cap,
                             equity_samples=int(saved_args.get("equity_samples", 200)))
        started = time.perf_counter()
        kwargs = {}
        if args.between_sizes:
            kwargs["bet_sizes"] = BETWEEN
        if args.sizes:
            kwargs["bet_sizes"] = tuple(args.sizes)
        trace = [] if args.trace else None
        if trace is not None:
            kwargs["trace"] = trace
        paired = None
        if args.paired:
            # The trace is the treated arm's: its records are the ones that
            # say what each probe or bridge-read overbet won.
            off_kwargs = {k: v for k, v in kwargs.items() if k != "trace"}
            off = LocalBestResponse(game, saved["strategy"], args.rollout_samples, args.candidates,
                                    **off_kwargs).play(args.hands, paired_seed=args.seed)
            print(f"{rung} options off: {off.summary()}")
            probing = LocalBestResponse(game, saved["strategy"], args.rollout_samples, args.candidates,
                                        **options, **kwargs)
            result = probing.play(args.hands, paired_seed=args.seed)
            print(f"{rung} options on:  {result.summary()}")
            difference = result.values - off.values
            paired = {"off": summary(off, args.hands, int(saved_args["big_blind"])),
                      "difference": {"n": int(difference.size), "sum": float(difference.sum()),
                                     "sumsq": float((difference ** 2).sum()),
                                     "mean": float(difference.mean()),
                                     "stderr": float(difference.std(ddof=1) / np.sqrt(difference.size)),
                                     "hands_changed": int((difference != 0).sum())},
                      "probe_stats": probing.probe_stats}
        elif treated:
            probing = LocalBestResponse(game, saved["strategy"], args.rollout_samples, args.candidates,
                                        **options, **kwargs)
            result = probing.play(args.hands, np.random.default_rng(args.seed))
            paired = {"probe_stats": probing.probe_stats}
        else:
            result = lbr_value(game, saved["strategy"], hands=args.hands,
                               rng=np.random.default_rng(args.seed),
                               rollout_samples=args.rollout_samples,
                               candidates=args.candidates, **kwargs)
        elapsed = time.perf_counter() - started
        if trace is not None:
            with open(args.trace, "a") as handle:
                for record in trace:
                    handle.write(json.dumps(dict(record, rung=rung, seed=args.seed)) + "\n")
        big_blind = int(saved_args["big_blind"])
        rows[rung] = {"path": os.path.basename(path), "mean": result.mean, "stderr": result.stderr,
                      "ci95": list(result.ci95), "proves_exploitable": bool(result.proves_exploitable),
                      "hands": args.hands, "bb_per_100": 100.0 * result.mean / big_blind,
                      "between_sizes": bool(args.between_sizes), "sizes": kwargs.get("bet_sizes"), "seconds": round(elapsed, 1),
                      "information_sets": len(saved["strategy"])}
        if treated:
            rows[rung]["offtree_third_raise"] = list(args.offtree_third_raise)
            rows[rung]["bridge_translation"] = bool(args.bridge_translation)
            rows[rung]["paired"] = bool(args.paired)
            rows[rung]["big_blind"] = big_blind
            rows[rung]["resolved"] = os.path.basename(os.path.realpath(path))
            rows[rung].update(paired)
            if args.paired:
                d = paired["difference"]
                print(f"   on minus off: {100 * d['mean'] / big_blind:+.1f} +/- "
                      f"{100 * d['stderr'] / big_blind:.1f} BB/100, {d['hands_changed']:,} of "
                      f"{d['n']:,} hands changed; {paired['probe_stats']}")
        print(f"{rung}: {result.summary()}")
        print(f"   {100.0 * result.mean / big_blind:+.1f} BB/100, {len(saved['strategy']):,} information sets, "
              f"{elapsed / 60:.1f} min")
        if not result.proves_exploitable:
            print("   the bound is slack: this says LBR failed, not that the rung is sound")

    if rows:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
