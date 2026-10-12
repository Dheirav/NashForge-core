"""
M1 — the first number in this project that somebody else's agent produced.

Step 5 of `docs/EXTERNAL_BENCHMARK.md`. Every strength figure here was computed by
this project about itself, and item 1 closed with no usable bound, so no-limit has
no exploitability figure at all. This does not fix that. It produces one measured
result against a fixed, external, published opponent, which is the whole of what
M1 claims: *10,000 hands with a confidence interval, any result*.

Two estimators, and why the headline is the dull one
-----------------------------------------------------
Slumbot returns `baseline_winnings` beside the actual winnings on every hand. On
the 300-hand pilot the two correlate at 0.85, and differencing them cuts the
spread from 2169 chips to 1362 — a real variance reduction, worth about 1.6x in
hands.

It is not the headline, because **the two estimators disagree in sign**: the pilot
read +130.7 chips/hand raw and −80.5 differenced. What `baseline_winnings` means
is not documented anywhere this project can check, so treating the difference as
the answer is an unverified modelling choice that moves the result across zero.
Item 1's cap-2 row is the precedent — +2.783 one afternoon and −2.900 that
evening, same untouched strategy, only the exploiter's valuation changed.

So the raw win rate is reported as the result, the differenced one is reported
beside it as an alternative whose interpretation is unverified, and neither is
quietly preferred to the other.

What is being measured, stated plainly
---------------------------------------
A 100bb, one-raise-per-street solver playing a 200bb unlimited-raise opponent. It
is not this project's strength at Slumbot's game and must not be quoted as though
it were. The lookup miss rate comes back with the result for the same reason the
panel reports it.

Usage
-----
    venv/bin/python scripts/slumbot_measure.py --hands 10000
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import numpy as np

from slumbot import SlumbotError, play_hand
from slumbot.api import BIG_BLIND
from slumbot.player import SolverPlayer

STRATEGY = os.path.join(ROOT, "results", "cfr", "nolimit_strategy.pkl")
OUT = os.path.join(ROOT, "results", "slumbot")

#: M1's bar. Below it the interval is wide enough that the number says nothing.
M1_HANDS = 10_000


def interval(values):
    """Mean and 95% half-width in mbb/hand, from chips per hand."""
    series = np.asarray(values, dtype=float)
    mean = series.mean() / BIG_BLIND * 1000
    half = 1.96 * series.std(ddof=1) / np.sqrt(len(series)) / BIG_BLIND * 1000
    return mean, half


def _strategy_depth(path):
    """
    Big blinds of stack the strategy was fitted for, read from the file itself.

    This label was hardcoded as "100bb" until 9 September. That was true of
    every strategy which had ever been passed here, and false the first time one
    was not. It is written into the JSON as `caveat`, so a stale one mislabels
    the record permanently rather than visibly.
    """
    with open(path, "rb") as handle:
        trained = pickle.load(handle).get("args") or {}
    stack, blind = trained.get("stack"), trained.get("big_blind")
    if not stack or not blind:
        return "unknown-depth"
    return f"{stack // blind}bb"


def _caveat(strategy):
    """
    What is playing whom, in one line.

    A function rather than a local because it is printed from `main` and again
    from `report`, and on 10 September it was a local of `main` that `report`
    referred to. That crashed with a NameError after 739 minutes of API calls,
    on the last line before the result would have been written.
    """
    return (f"{_strategy_depth(strategy)} {_strategy_tree(strategy)} solver "
            "against Slumbot's 200bb unlimited-raise game")


def _strategy_tree(path):
    """The betting tree, read from the file, so a deeper solver is labelled as one."""
    from slumbot.player import strategy_schedule
    with open(path, "rb") as handle:
        schedule = strategy_schedule(pickle.load(handle))
    if schedule == 1:
        return "one-raise-per-street"
    if isinstance(schedule, int):
        return f"raise-cap-{schedule}"
    return f"taper-{'-'.join(str(n) for n in schedule)}"


def _partial_path(out):
    """Beside the result, named so it cannot be mistaken for one."""
    return os.path.splitext(out)[0] + ".partial.json"


def _save_partial(path, args, player, winnings, baseline, positions, errors,
                  attempts, elapsed, hands=()):
    """
    Every hand so far, written atomically every 500.

    Ten thousand hands against Slumbot is nine hours of API round-trips, and on
    9 September WSL went down at hand 1,500 and took the lot -- no traceback, no
    partial, eighty-five minutes gone. The rate is fixed by the opponent's
    latency, so the only way to make a run that long survivable is to be able to
    resume it.

    Written to a temporary name and renamed, because the failure this exists for
    is the process disappearing mid-write.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({
            "strategy": os.path.basename(args.strategy),
            "seed": args.seed,
            "hands": args.hands,
            "attempts": attempts,
            "winnings": winnings,
            "baseline": baseline,
            "positions": dict(positions),
            "errors": errors,
            "elapsed_seconds": elapsed,
            "segments": args.segments,
            # Without this a resumed run reports the miss rate of its last
            # segment under the whole run's name. This project has already
            # published +60.8 BB/100 beside a 74.3% miss rate it could not see;
            # a rate that silently describes 500 hands of 10,000 is the same
            # failure wearing a checkpoint.
            "stats": dataclasses.asdict(player.stats),
            "hand_records": list(hands),
            "provenance": getattr(args, "provenance", None),
            "note": "PARTIAL. Not a result -- resume with --resume.",
        }, handle)
    os.replace(tmp, path)


def _load_partial(path, args):
    """
    Pick a run back up, or refuse to.

    The three fields checked are the ones that would silently change what is
    being estimated: a different strategy, a different seed or a different
    target makes the two halves samples of different things, and pooling them
    would produce a number nothing could reproduce.
    """
    with open(path) as handle:
        saved = json.load(handle)
    for field, want in (("strategy", os.path.basename(args.strategy)),
                        ("seed", args.seed), ("hands", args.hands)):
        if saved.get(field) != want:
            raise SystemExit(
                f"cannot resume {path}: {field} was {saved.get(field)!r}, "
                f"this run is {want!r}. Delete it or fix the arguments.")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=M1_HANDS)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out", default=os.path.join(OUT, "m1.json"))
    #: Which solver is playing. There are two now -- the 4,000-iteration one
    #: that shipped and a 150,000-iteration one that beats it head to head by
    #: 185 BB/100 -- and a result that does not say which produced it cannot be
    #: compared with the other.
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--purify", default="none", choices=["none", "postflop", "all"],
                        help="play the most probable action instead of sampling; off by default")
    parser.add_argument("--no-fallback", action="store_true",
                        help="guess uniformly at random on a lookup miss, as before 15 September")
    parser.add_argument("--report-partial", action="store_true",
                        help="write the report from a checkpoint that already holds "
                             "every hand, without playing; for a run that finished "
                             "its hands and then failed in the report")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the .partial.json beside --out")
    args = parser.parse_args()

    if args.hands < M1_HANDS:
        parser.error(f"M1 is {M1_HANDS:,} hands; fewer is a pilot, and "
                     "scripts/slumbot_pilot.py is the script that does not "
                     "report a result")

    player = SolverPlayer(args.strategy, np.random.default_rng(args.seed), purify=args.purify,
                          fallback=not args.no_fallback)
    caveat = _caveat(args.strategy)
    winnings, baseline, positions = [], [], Counter()
    hands, errors, token, started = [], [], None, time.time()
    partial, first, prior_elapsed, args.segments = _partial_path(args.out), 0, 0.0, 1

    if args.resume and os.path.exists(partial):
        saved = _load_partial(partial, args)
        winnings, baseline, errors = saved["winnings"], saved["baseline"], saved["errors"]
        hands = saved.get("hand_records", [])
        # JSON turns integer keys into strings; a Counter keyed on "0" and one
        # keyed on 0 both look right and never sum.
        positions = Counter({int(k): v for k, v in saved["positions"].items()})
        first, prior_elapsed = saved["attempts"], saved["elapsed_seconds"]
        args.segments = saved["segments"] + 1
        args.provenance = saved.get("provenance")
        # Protocol health carries across the join, or the miss rate printed at
        # the end describes only the hands this process happened to play.
        for name, value in (saved.get("stats") or {}).items():
            # The same JSON string-key trap as positions: the depth dicts are
            # keyed by int and the report sorts them, and a "2" beside a 2
            # crashed the report at the end of a nine-hour run on 16 Sept.
            if isinstance(value, dict):
                value = {int(k) if str(k).lstrip("-").isdigit() else k: v
                         for k, v in value.items()}
            setattr(player.stats, name, value)
    elif args.resume:
        print(f"no checkpoint at {partial}; starting from hand 1\n", flush=True)
    if args.report_partial:
        if len(winnings) + len(errors) < args.hands:
            raise SystemExit(f"{partial} holds {len(winnings)} hands of {args.hands}; "
                             "not a finished run, resume it instead")
        report(player, winnings, baseline, positions, errors, args, prior_elapsed, hands)
        os.remove(partial)
        return

    print(f"M1 — {args.hands:,} hands against Slumbot, seed {args.seed}")
    print(f"strategy: {os.path.basename(args.strategy)}")
    print(caveat, flush=True)
    if first:
        print(f"resuming at hand {first:,} of {args.hands:,} "
              f"({prior_elapsed / 60:.0f} min already spent, "
              f"segment {args.segments})")
    print(flush=True)

    for index in range(first, args.hands):
        try:
            player.begin_hand()
            state = play_hand(player, token)
            token = state.token
            winnings.append(state.winnings)
            baseline.append(state.baseline_winnings)
            positions[state.client_pos] += 1
            hands.append(player.hand_record(state))
        except SlumbotError as error:
            errors.append(f"hand {index + 1}: {error}")
        if (index + 1) % 500 == 0:
            done = index + 1
            # Rate over this segment only. Dividing the whole count by this
            # process's clock would credit a resumed run with hands another
            # process played and put the ETA hours early.
            segment = time.time() - started
            rate = (done - first) / segment
            mean, half = interval(winnings)
            print(f"  {done:>6,}/{args.hands:,}  {rate:4.1f} hands/s  "
                  f"eta {(args.hands - done) / rate / 60:5.1f} min  "
                  f"[{mean:+7.0f} ± {half:.0f} mbb/hand so far]", flush=True)
            _save_partial(partial, args, player, winnings, baseline, positions,
                          errors, done, prior_elapsed + segment, hands)

    report(player, winnings, baseline, positions, errors, args,
           prior_elapsed + time.time() - started, hands)

    # Only once the result exists. The checkpoint is the fallback, and removing
    # it before the thing it falls back to is written would recreate the hole.
    if os.path.exists(partial):
        os.remove(partial)


def report(player, winnings, baseline, positions, errors, args, elapsed, hands=()):
    stats, caveat = player.stats, _caveat(args.strategy)
    raw_mean, raw_half = interval(winnings)

    usable = [w - b for w, b in zip(winnings, baseline) if b is not None]
    adj_mean, adj_half = interval(usable) if len(usable) > 1 else (None, None)

    print(f"\n{'=' * 64}")
    print(f"M1 — {len(winnings):,} hands in {elapsed / 60:.0f} min\n")
    print(f"  win rate            {raw_mean:+8.1f} ± {raw_half:.0f} mbb/hand")
    print(f"                      95% interval "
          f"[{raw_mean - raw_half:+.0f}, {raw_mean + raw_half:+.0f}]")
    if adj_mean is not None:
        print(f"\n  differenced against Slumbot's baseline, interpretation")
        print(f"  unverified — NOT the result:")
        print(f"                      {adj_mean:+8.1f} ± {adj_half:.0f} mbb/hand")
        if (raw_mean > 0) != (adj_mean > 0):
            print("    The two estimators disagree in sign. Neither is quoted")
            print("    alone, and the raw figure above is the reported one.")

    print(f"\n  lookup miss rate    {stats.miss_rate:.1%} "
          f"({stats.misses}/{stats.consulted})")
    print(f"  misses by raises already on the street  "
          f"{dict(sorted(stats.miss_depths.items()))}")
    print(f"  re-raises the schedule had no size for  "
          f"{dict(sorted(stats.schedule_misses.items()))}")
    print(f"  off-abstraction     {stats.off_abstraction}")
    print(f"  seats               {dict(positions)}")
    print(f"  protocol errors     {len(errors)}")
    print(f"\n  Measures a {caveat}.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({
            "milestone": "M1",
            "strategy": os.path.basename(args.strategy),
            "hands": len(winnings),
            "seed": args.seed,
            "units": "mbb/hand",
            "win_rate": raw_mean, "ci95": raw_half,
            "baseline_differenced": adj_mean,
            "baseline_differenced_ci95": adj_half,
            "baseline_note": "Slumbot's baseline_winnings differenced from "
                             "actual. Interpretation unverified; not the "
                             "reported result.",
            "lookup_miss_rate": stats.miss_rate,
            "lookup_misses_by_raise_depth": {str(k): v for k, v in sorted(stats.miss_depths.items())},
            "schedule_misses_by_raise_depth": {str(k): v for k, v in sorted(stats.schedule_misses.items())},
            "schedule": str(player.raise_cap),
            "purify": player.purify,
            "fallback": player.fallback, "fallbacks": player.stats.fallbacks,
            "off_abstraction": stats.off_abstraction,
            "seats": dict(positions),
            "protocol_errors": errors,
            "caveat": caveat,
            "segments": getattr(args, "segments", 1),
            "provenance": getattr(args, "provenance", None),
            "elapsed_seconds": elapsed,
            # Per hand, for scripts/slumbot_split.py: winnings, position, final
            # street, showdown, and whether the solver missed in the hand.
            "hand_records": list(hands),
        }, handle, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
