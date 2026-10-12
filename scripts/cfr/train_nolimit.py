"""
Milestone 5 — train a CFR agent on abstracted heads-up no-limit Hold'em.

Fits a card abstraction, runs external-sampling MCCFR over the abstracted game,
and measures the resulting strategy against baselines by playing hands.

Two honest limits on what this can report, both stated rather than papered over:

* **Exploitability is not computable here.** Kuhn and Leduc allow an exact best
  response; no-limit does not. Local Best Response gives a lower bound and is
  the next piece of work. Until then the only measure available is head-to-head
  chips, which is a weaker claim: beating a passive baseline shows the strategy
  is not broken, not that it is near equilibrium.

* **Head-to-head results need error bars.** Poker results over a few thousand
  hands are dominated by variance. Every figure below carries a standard error
  and a 95% interval, and seats alternate so position is not a confound.

Usage
-----
    python scripts/cfr/train_nolimit.py
    python scripts/cfr/train_nolimit.py --iterations 20000 --buckets 8
    python scripts/cfr/train_nolimit.py --raise-cap 2 --output strategy.npz
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.betting import (RAISE_ACTIONS, STREETS, StreetSchedule, measure,  # noqa: E402
                                 normalise_schedule, parse_level, schedule_from_args)
from abstraction.buckets import CardAbstraction, preflop_key
from abstraction.equity import FULL_DECK  # noqa: E402
from cfr import ALL_RULES, MCCFRSolver  # noqa: E402
from cfr.flat import KEY_WIDTH, FlatStrategy, flat_paths  # noqa: E402
from cfr.provenance import provenance  # noqa: E402

#: The native path's flat export, written beside the pickle by `_report_and_write`.
_FLAT_EXPORT = None
from cfr.play import (always_call_policy, play_hands, strategy_policy,
                      uniform_policy)  # noqa: E402
from games.nolimit import NoLimitHoldem  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--buckets", type=int, default=6)
    parser.add_argument("--preflop-buckets", type=int, default=None,
                        help="preflop classes; defaults to --buckets. 169 keeps every "
                             "starting hand apart (abstraction.buckets.PREFLOP_HANDS)")
    #: An int is the uniform schedule (N raises, all four sizes). Two or more
    #: values taper: `--raise-cap 4 1` allows four sizes for the opening bet and
    #: only all-in for the raise, which buys a re-raise for 347,136 information
    #: sets where carrying all four sizes to depth 2 costs 7,560,240.
    parser.add_argument("--raise-cap", type=parse_level, nargs="+", default=[1],
                        help="uniform depth, or a tapered schedule of sizes per raise "
                             "depth (e.g. --raise-cap 4 1). A level may name its sizes "
                             "instead of counting them from the largest end: "
                             "--raise-cap half,pot,jam half,pot,jam jam keeps the "
                             "pot-sized re-raise and drops the 2x-pot one, at the same "
                             "tree size")
    #: A schedule for one street, which then differs from --raise-cap there:
    #: `--street-raise-cap preflop half,pot,2x,jam half,2x,jam jam`. Repeat per
    #: street. On 26 Sept the traced LBR leak was 41 to 44% preflop sizes the
    #: tree lacks; adding them preflop only costs x1.4 to x1.9, everywhere x4 to x12.
    parser.add_argument("--street-raise-cap", nargs="+", action="append", default=None,
                        metavar=("STREET", "LEVEL"),
                        help="a street (preflop/flop/turn/river) and its levels, as --raise-cap takes them")
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--abstraction-samples", type=int, default=800)
    parser.add_argument("--equity-samples", type=int, default=40)
    parser.add_argument("--update-rule", default="vanilla",
                        type=lambda v: v if v in ("vanilla", "linear", "cfr+", "dcfr") or v.startswith("dcfr:") else
                        (_ for _ in ()).throw(argparse.ArgumentTypeError(f"vanilla, linear, cfr+, dcfr or dcfr:A,B,G, not {v}")),
                        help="regret and averaging schedule (cfr/updates.py). vanilla weighs "
                             "every iteration's strategy equally, which leaves a rarely "
                             "reached node's average at its early near-uniform visits")
    parser.add_argument("--common-random-numbers", action="store_true",
                        help="one deal per iteration shared across every branch (native only); "
                             "off by default until its head-to-head passes")
    parser.add_argument("--exact-terminals", action="store_true",
                        help="score flop and turn all-ins over every runout instead of one sample "
                             "(native only); off by default until its head-to-head passes")
    parser.add_argument("--current-when-empty", action="store_true",
                        help="export regret matching's current strategy at nodes whose average is empty, "
                             "instead of a uniform (native only); off by default until its head-to-head passes")
    parser.add_argument("--average-from", type=float, default=0.0,
                        help="fraction of the run before the average strategy starts accumulating "
                             "(Pluribus skipped the early part; 0 is the classical average)")
    #: The card feature. `histogram` clusters on the distribution of river
    #: equity over sampled runouts with earth mover's distance
    #: (abstraction/histogram.py; Johanson et al. 2013), which separates draws
    #: from made hands where the scalar E[HS] cannot. Fitting costs
    #: `--hist-runouts` x `--hist-opponents` evaluations per sampled situation.
    parser.add_argument("--strength", default="equity", choices=["equity", "histogram"],
                        help="postflop card feature: E[HS] (default) or the equity histogram")
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument("--hist-runouts", type=int, default=100)
    parser.add_argument("--hist-opponents", type=int, default=50)
    parser.add_argument("--fit-samples", type=int, default=None,
                        help="histogram mode: fit the postflop classes on this many native-sampled "
                             "situations a street (default: --abstraction-samples through the Python "
                             "histogram, 800; 20,000 here costs eight seconds)")
    parser.add_argument("--abstraction-from", metavar="PICKLE",
                        help="reuse the fitted abstraction (and its bucket tables) of an existing "
                             "solve instead of fitting one, so a ladder shares one clustering and one "
                             "set of tables; the stack and tree may differ, the card classes do not")
    # The list comes from chipzen.archetypes, not from a copy here. On 24
    # September a copy that still held four shapes rejected `bully` seven times
    # in two seconds, the lane assembled a ladder with no exploiter rungs in
    # it, and every measurement that followed was one solver against itself.
    from chipzen.archetypes import ARCHETYPES
    parser.add_argument("--opponent-archetype", choices=sorted(ARCHETYPES),
                        help="native only: train a best response to this scripted field shape "
                             "(chipzen/archetypes.py's calibrated parameters) instead of an equilibrium; "
                             "the result is an exploiter, played only behind a confident read")
    parser.add_argument("--opponent-share", type=float, default=0.5,
                        help="how often the scripted opponent plays its script rather than its learned strategy: "
                             "1.0 is a pure best response (overfits the script), 0.5 a restricted Nash response")
    parser.add_argument("--opponent-mix", nargs="+", metavar="NAME:SHARE",
                        help="native only: train against several scripted shapes at once, e.g. station:0.25 maniac:0.15; "
                             "at each opponent node a script plays with the shares' total probability, split by share "
                             "(a solve trained against one shape alone overfits it)")
    parser.add_argument("--table-threads", type=int, default=None,
                        help="threads for building the bucket tables (default: --threads); the build is "
                             "embarrassingly parallel and the tables are built once")
    parser.add_argument("--no-hist-tables", action="store_true",
                        help="histogram mode: compute buckets at the lookup instead of precomputing the "
                             "flop and turn tables (measured 21 Sept: 5.0 ms/it without them, 0.3 with)")
    parser.add_argument("--texture", action="store_true",
                        help="fold the board's flush and straight texture into the "
                             "postflop bucket (abstraction.buckets.board_texture)")
    parser.add_argument("--eval-hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the strategy and summary here")
    #: The C++ core, roughly 32x faster and answer-changing: bucketing is seeded
    #: from the cards rather than from Python's tuple hash. Default, with
    #: --no-native to fall back to the Python solver, which stays as the
    #: reference `tests/test_native.py` pins the port against.
    parser.add_argument("--native", action=argparse.BooleanOptionalAction,
                        default=True, help="use the C++ solver core")
    #: Worker threads sharing one node table (native only). One is the exact
    #: single-threaded path the golden test pins; more split the iterations
    #: between workers with their own deals and random streams, which is not
    #: reproducible bit for bit, only in distribution. Item 11 of NEXT.md.
    parser.add_argument("--threads", type=int, default=1,
                        help="worker threads for the native solver (1: the golden path)")
    #: Warm start (native only): seed every node whose key the given pickle
    #: holds with that strategy, as if `--warm-weight` iterations had already
    #: been played at `--warm-scale` chips of regret each (default: one big
    #: blind). Meant for a cap-2 solve started from the one-raise solve at the
    #: same depth, whose histories are a subset with the same bucket scheme;
    #: the re-raise actions start at zero regret. See MCCFR::warm_start.
    parser.add_argument("--warm-start", metavar="PICKLE",
                        help="seed the shared nodes from this saved strategy")
    parser.add_argument("--warm-weight", type=int, default=1_000_000,
                        help="iterations the warm start counts for")
    parser.add_argument("--warm-scale", type=float, default=None,
                        help="chips of regret per warm iteration (default: the big blind)")
    #: Regret-based pruning (native only), Pluribus's form: after `--prune-after`
    #: of the run, 95% of iterations skip the traverser's actions whose regret
    #: is below `--prune-stacks` starting stacks (Pluribus: -300M chips on
    #: 10,000-chip stacks, i.e. 30,000 stacks), never on the river and never an
    #: action that ends the hand. Off by default; the golden test pins that.
    parser.add_argument("--prune-after", type=float, default=None,
                        help="fraction of the run before pruning starts (e.g. 0.1); off if absent")
    parser.add_argument("--prune-stacks", type=float, default=30000.0,
                        help="prune an action whose regret is below minus this many starting stacks")
    parser.add_argument("--warm-mode", default="proportional", choices=["proportional", "frozen"],
                        help="proportional: regrets set to the prior; frozen: play the prior for "
                             "--warm-weight iterations while regrets accumulate (substitute regrets)")
    #: Per-node visit counts (native only), written as `<name>.visits.npz`
    #: beside the flat pair; read with scripts/cfr/visit_report.py. Off by
    #: default: off, the counters are never written and the golden path is
    #: unchanged. The counters live in the node's padding, so on costs no memory.
    parser.add_argument("--count-visits", action="store_true",
                        help="count visits per information set and write <name>.visits.npz")
    #: Mid-run snapshots of the counts, so one run gives the T and 2T pair the
    #: stopping rule compares instead of two runs. Counted in --iterations'
    #: units, after any frozen warm phase.
    parser.add_argument("--visit-snapshots", type=int, nargs="+", default=[],
                        help="also write <name>.visits.<N>.npz after N iterations (needs --count-visits)")
    return parser.parse_args()


def _resident_mb():
    """This process's resident size, or 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


#: Where this run came from (cfr/provenance.py), taken once at the start of `main` so a commit made while it trains
#: is not mistaken for the code that trained it; written into every summary this run writes.
_PROVENANCE = None


def _write(path, strategy, abstraction, args, results, iterations, seconds):
    """The strategy and its summary, written together so neither outlives the other."""
    with open(path, "wb") as handle:
        pickle.dump({"strategy": strategy, "abstraction": abstraction,
                     "args": vars(args), "results": results}, handle)
    with open(os.path.splitext(path)[0] + ".json", "w") as handle:
        json.dump({"args": vars(args), "results": results,
                   "information_sets_reached": len(strategy),
                   "iterations_completed": iterations,
                   "seconds": seconds, "provenance": _PROVENANCE}, handle, indent=2)


def _native_tables(abstraction):
    """
    The fitted abstraction as flat tables the C++ solver takes.

    Handed over rather than refitted on the other side: k-means over sampled
    equities happens once and is cheap, and refitting in C++ would be a second
    clustering that could silently disagree with this one.
    """
    preflop = [0] * (52 * 52)
    for first in range(52):
        for second in range(52):
            if first != second:
                preflop[first * 52 + second] = abstraction._preflop[
                    preflop_key([FULL_DECK[first], FULL_DECK[second]])]
    return (preflop,
            list(abstraction._centroid_list["flop"]),
            list(abstraction._centroid_list["turn"]),
            list(abstraction._centroid_list["river"]))


def _train_native(args, abstraction, projected):
    """
    Train with the C++ core, then write the same pickle the Python path writes.

    Everything downstream -- `evaluation.benchmark`, the Slumbot bridge, the GUI
    -- reads that file and must not be able to tell which solver produced it.

    The native path CHANGES ANSWERS: bucketing is seeded from the cards rather
    than from Python's tuple hash, so a hand near a boundary can fall either
    side. The two were measured playing indistinguishably, 16.2 and 23.1 BB/100
    apart against random and always-call, but strategies from the two paths are
    not comparable with each other. See docs/retrain-plan.md.
    """
    import pokerbot_native

    # Counts go in `schedule`, named levels in `raise_sizes`; the native side
    # takes the named form when it is given one. A schedule is one or the
    # other, never a mixture, so the saved args describe one game.
    cap = schedule_from_args(args.raise_cap)
    street_raise_sizes = []
    if isinstance(cap, StreetSchedule):
        street_raise_sizes = [[list(level) for level in normalise_schedule(spec)] for spec in cap.streets]
        schedule, raise_sizes = [], street_raise_sizes[0]
    elif isinstance(args.raise_cap, int):
        schedule, raise_sizes = [4] * args.raise_cap, []
    elif any(isinstance(level, (tuple, list)) for level in args.raise_cap):
        schedule = []
        raise_sizes = [list(level) if isinstance(level, (tuple, list))
                       else list(RAISE_ACTIONS[len(RAISE_ACTIONS) - int(level):])
                       for level in args.raise_cap]
    else:
        schedule, raise_sizes = list(args.raise_cap), []
    preflop, flop, turn, river = _native_tables(abstraction)

    print(f"\nTraining MCCFR (native) for {args.iterations:,} iterations"
          + (f" on {args.threads} threads..." if args.threads > 1 else "..."))
    hist = {}
    if getattr(abstraction, "strength", "equity") == "histogram":
        hist = dict(hist_centroids=[abstraction._hist_centroids[street].tolist()
                                    for street in ("flop", "turn", "river")],
                    hist_bins=abstraction.hist_bins, hist_runouts=abstraction.hist_runouts,
                    hist_opponents=abstraction.hist_opponents)
        tables = abstraction.native_tables()
        hist["flop_table"] = tables.get("flop")
        hist["turn_table"] = tables.get("turn")
    solver = pokerbot_native.NoLimitSolver(
        preflop, flop, turn, river, args.equity_samples, args.stack,
        args.big_blind // 2, args.big_blind, schedule, args.seed,
        texture=bool(getattr(abstraction, "texture", False)),
        rule=args.update_rule, raise_sizes=raise_sizes, street_raise_sizes=street_raise_sizes, **hist)
    solver.set_average_from(int(args.average_from * args.iterations))
    averaged_from = int(args.average_from * args.iterations)
    if args.visit_snapshots and not args.count_visits:
        raise SystemExit("--visit-snapshots needs --count-visits")
    if args.count_visits and not args.output:
        raise SystemExit("--count-visits writes beside --output; give one")
    solver.set_count_visits(bool(args.count_visits))
    if args.opponent_mix and args.opponent_archetype:
        raise SystemExit("--opponent-mix and --opponent-archetype are alternatives; give one")
    if args.opponent_mix:
        from chipzen.archetypes import PARAMS
        mix = []
        for entry in args.opponent_mix:
            name, _, share = entry.partition(":")
            if name not in PARAMS or not share:
                raise SystemExit(f"--opponent-mix: NAME:SHARE with NAME one of {sorted(PARAMS)}, not {entry}")
            mix.append(({k: float(v) for k, v in PARAMS[name].items()}, float(share)))
        solver.set_opponent_mix(mix, args.big_blind)
        print("restricted best response against a mix: " + ", ".join(f"{e.partition(':')[0]} {100 * float(e.partition(':')[2]):.0f}%"
                                                                      for e in args.opponent_mix) + " of opponent decisions", flush=True)
    if args.opponent_archetype:
        from chipzen.archetypes import PARAMS
        solver.set_opponent_archetype({k: float(v) for k, v in PARAMS[args.opponent_archetype].items()}, args.big_blind,
                                      args.opponent_share)
        print(f"restricted best response: the opponent plays the {args.opponent_archetype} archetype "
              f"{100 * args.opponent_share:.0f}% of the time", flush=True)
    solver.set_common_random_numbers(bool(args.common_random_numbers))
    solver.set_exact_terminals(bool(args.exact_terminals))
    solver.set_current_when_empty(bool(args.current_when_empty))
    if args.warm_start:
        from cfr.flat import load_strategy
        prior = load_strategy(args.warm_start)
        entries = [(key, [float(p) for p in prior["strategy"][key]]) for key in prior["strategy"]]
        # By action, not by position (cfr/warm.py): a tree that inserts a size into existing nodes would
        # otherwise inherit each probability one slot to the left. Identity when the trees agree.
        from cfr.warm import remap_entries
        prior_args = prior.get("args") or {}
        prior_cap = prior_args.get("raise_cap", 1) if isinstance(prior_args, dict) else getattr(prior_args, "raise_cap", 1)
        entries, dropped = remap_entries(entries, schedule_from_args(prior_cap), schedule_from_args(args.raise_cap))
        print(f"warm start mapped by action onto this tree: {len(entries):,} entries, {dropped:,} dropped "
              f"(no action list of that width at their node)", flush=True)
        scale = args.big_blind if args.warm_scale is None else args.warm_scale
        solver.warm_start(entries, args.warm_weight, scale, args.warm_mode)
        if args.warm_mode == "frozen":
            # The frozen iterations are measurement, not play: keep them out of
            # the average, and out of the run's own count of iterations.
            solver.set_average_from(args.warm_weight + int(args.average_from * args.iterations))
            averaged_from = args.warm_weight + int(args.average_from * args.iterations)
            args.iterations += args.warm_weight
        print(f"warm start ({args.warm_mode}): {len(entries):,} entries from {args.warm_start}, "
              f"weight {args.warm_weight:,} iterations at {scale:g} chips", flush=True)
        del prior, entries
    if args.prune_after is not None:
        solver.set_pruning(int(args.prune_after * args.iterations), -args.prune_stacks * args.stack, 0.95)
        print(f"pruning from iteration {int(args.prune_after * args.iterations):,}: regret below "
              f"{-args.prune_stacks * args.stack:,.0f} chips, 95% of iterations", flush=True)
    start = time.perf_counter()

    # Trained in chunks so the run reports progress.
    #
    # The native path printed nothing until it finished, which cost a run: the
    # [4,2] taper on 11 September spent 66 minutes swapping at 50% CPU and there
    # was no way to see it was in trouble, or how far along it was. The Python
    # path has had a progress callback since the 500,000-iteration run that was
    # killed at six hours for the same reason.
    step = max(1, args.iterations // 50)
    done = 0
    # The proportional warm start begins the solver's counter at the prior's
    # weight; those iterations were never played, so they fed no average.
    averaged_from = max(averaged_from, solver.iterations())
    frozen = args.warm_weight if (args.warm_start and args.warm_mode == "frozen") else 0
    snapshots = sorted(n + frozen for n in args.visit_snapshots if 0 < n + frozen < args.iterations)
    skipped = [n for n in args.visit_snapshots if not 0 < n + frozen < args.iterations]
    if skipped:
        # Said aloud because the T and 2T pair is the point of asking: a typo here would otherwise surface
        # only when visit_report.py is handed a file that was never written.
        print(f"  warning: visit snapshots {skipped} skipped: a snapshot must fall inside the run's "
              f"{args.iterations - frozen:,} iterations, and the end of the run is written anyway", flush=True)
    while done < args.iterations:
        chunk = min(step, args.iterations - done)
        if snapshots:
            chunk = min(chunk, snapshots[0] - done)   # land exactly on the snapshot
        solver.train(chunk, args.threads)
        done += chunk
        if snapshots and done == snapshots[0]:
            from cfr.visits import from_solver, visits_path, write_visits
            label = done - frozen
            path = visits_path(args.output, label)
            write_visits(path, from_solver(solver, averaged_from, {"args": vars(args), "run_iterations": label}))
            print(f"  visit snapshot at {label:,}: {path}", flush=True)
            snapshots.pop(0)
        taken = time.perf_counter() - start
        rss = _resident_mb()
        # ETA from the measured rate, never estimated up front.
        eta = (args.iterations - done) * taken / done / 60
        print(f"  {done:>10,}/{args.iterations:,}  {taken / done * 1000:6.3f} ms/it  "
              f"{solver.information_sets():>9,} infosets  {rss:6.0f} MB  "
              f"eta {eta:6.1f} min", flush=True)
    elapsed = time.perf_counter() - start
    print(f"  {elapsed:.1f}s ({elapsed / args.iterations * 1000:.3f} ms/iteration)")
    if args.opponent_archetype or args.opponent_mix:
        # The exploiter's value against the exact opponent it was solved
        # against, in the training game's chips per hand; the number the duel
        # against the Python copy of that opponent cannot separate from the
        # copy's mismatch.
        value = solver.evaluate_against_policy(200000, args.seed + 7)
        print(f"  against the {args.opponent_archetype or ' + '.join(args.opponent_mix)} in the training game: {value:+.3f} chips/hand "
              f"({100.0 * value / args.big_blind:+.1f} BB/100) over 200,000 hands", flush=True)
    print(f"  information sets reached: {solver.information_sets():,} "
          f"of {projected.information_sets:,} in the abstraction")
    if args.warm_start:
        print(f"  warm start: {solver.warm_hits():,} of {solver.warm_entries():,} entries "
              f"landed on a node that was reached")
    if args.prune_after is not None:
        print(f"  pruning: {solver.pruned():,} action visits skipped")

    if args.count_visits:
        from cfr.visits import from_solver, visits_path, write_visits
        counts = from_solver(solver, averaged_from, {"args": vars(args), "run_iterations": done - frozen})
        write_visits(visits_path(args.output), counts)
        reached = counts.avg
        print(f"  visits: median {np.median(reached):,.0f} average-strategy samples a node, 10th percentile "
              f"{np.percentile(reached, 10):,.0f}; {np.mean(reached < 100):.1%} of nodes under 100 "
              f"(scripts/cfr/visit_report.py {visits_path(args.output)})", flush=True)
        del counts, reached

    # Flat export: three arrays rather than a tree, a dict and millions of
    # small arrays. The pickle keeps its dict shape (views into the values
    # array), and the flat pair is written beside it from the same arrays.
    key_bytes, offsets, values = solver.average_strategy_flat(KEY_WIDTH)
    keys = np.asarray(key_bytes).view(f"S{KEY_WIDTH}").reshape(-1)
    flat = FlatStrategy(keys, np.asarray(offsets), np.asarray(values))
    strategy = {k.decode(): flat._values[flat._offsets[i]:flat._offsets[i + 1]]
                for i, k in enumerate(keys)}
    global _FLAT_EXPORT
    _FLAT_EXPORT = flat

    # Scored and written exactly as the Python path does, through the same
    # Python game, so the file is indistinguishable downstream and the printed
    # baselines are comparable with every previous run's.
    game = NoLimitHoldem(abstraction, starting_stack=args.stack,
                         big_blind=args.big_blind, raise_cap=schedule_from_args(args.raise_cap),
                         equity_samples=args.equity_samples)
    _report_and_write(args, game, abstraction, strategy,
                      solver.information_sets(), elapsed)


def main():
    global _PROVENANCE
    args = parse_args()
    _PROVENANCE = provenance(vars(args), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    # One value is the int the rest of the codebase has always taken; more than
    # one is a taper. Normalised here so `vars(args)` in the output JSON records
    # which game produced the strategy.
    args.raise_cap = (args.raise_cap[0] if len(args.raise_cap) == 1
                      else tuple(args.raise_cap))
    if isinstance(args.raise_cap, (tuple, list)):
        args.raise_cap = tuple(tuple(level) if isinstance(level, (tuple, list)) else int(level)
                               for level in args.raise_cap)
    if args.street_raise_cap:
        specs = [args.raise_cap] * len(STREETS)
        for entry in args.street_raise_cap:
            street, levels = entry[0].lower(), [parse_level(level) for level in entry[1:]]
            if street not in STREETS or not levels:
                raise SystemExit(f"--street-raise-cap: a street ({', '.join(STREETS)}) and its levels, not {entry}")
            specs[STREETS.index(street)] = tuple(tuple(l) if isinstance(l, (tuple, list)) else int(l) for l in levels)
        # Saved as {street: named levels}, which `schedule_from_args` reads back.
        args.raise_cap = StreetSchedule(tuple(specs)).for_saving()
        args.street_raise_cap = None
    rng = np.random.default_rng(args.seed)
    if args.preflop_buckets is None:
        args.preflop_buckets = args.buckets
    if (args.warm_start or args.prune_after is not None) and not args.native:
        raise SystemExit("--warm-start and --prune-after are native solver features; drop --no-native")
    if args.count_visits and not args.native:
        raise SystemExit("--count-visits is a native solver feature; drop --no-native")

    projected = measure({street: (args.preflop_buckets if street == "preflop" else args.buckets)
                         for street in STREETS},
                        raise_cap=schedule_from_args(args.raise_cap))
    print(f"Abstract game: {projected.summary()}")
    print(f"  (raise cap is the parameter that decides feasibility — see "
          f"measure_abstraction.py)\n")

    start = time.perf_counter()
    if args.abstraction_from:
        from cfr.flat import load_strategy
        abstraction = load_strategy(args.abstraction_from)["abstraction"]
        for name in ("buckets", "preflop_buckets", "strength", "texture"):
            own = {"buckets": abstraction.postflop_buckets, "preflop_buckets": abstraction.preflop_buckets,
                   "strength": getattr(abstraction, "strength", "equity"),
                   "texture": bool(getattr(abstraction, "texture", False))}[name]
            if getattr(args, name) != own:
                raise SystemExit(f"--abstraction-from: the pickle's {name} is {own!r}, the run asks {getattr(args, name)!r}")
        print(f"Card abstraction reused from {args.abstraction_from}"
              + (" with its bucket tables" if getattr(abstraction, "_hist_tables", None) else ""))
    else:
        print(f"Fitting card abstraction ({args.fit_samples or args.abstraction_samples} situations/street"
              + (", native histograms" if args.fit_samples else "") + ")...")
        abstraction = CardAbstraction(
            preflop_buckets=args.preflop_buckets, postflop_buckets=args.buckets,
            samples=args.abstraction_samples, equity_samples=args.equity_samples,
            texture=args.texture, strength=args.strength,
            hist_bins=args.hist_bins, hist_runouts=args.hist_runouts, hist_opponents=args.hist_opponents,
        ).fit(rng, fit_samples=args.fit_samples)
        print(f"  fitted in {time.perf_counter() - start:.1f}s")
    print(abstraction.describe())
    if args.strength == "histogram" and not args.no_hist_tables and not getattr(abstraction, "_hist_tables", None):
        # Once per abstraction: every canonical flop and turn situation's
        # bucket, so neither the solve nor the bot computes a histogram on
        # those streets again. Minutes for the flop, tens of minutes for the
        # turn on the given threads.
        start = time.perf_counter()
        table_threads = max(args.table_threads or args.threads, 1)
        print(f"Building bucket tables on {table_threads} threads...", flush=True)
        abstraction.build_tables(threads=table_threads,
                                 progress=lambda street, n: print(f"  {street}: {n:,} classes at "
                                                                  f"{time.perf_counter() - start:.0f}s", flush=True))
        print(f"  tables built in {time.perf_counter() - start:.0f}s")

    game = NoLimitHoldem(abstraction, starting_stack=args.stack,
                         big_blind=args.big_blind, raise_cap=schedule_from_args(args.raise_cap),
                         equity_samples=args.equity_samples)

    if args.native:
        return _train_native(args, abstraction, projected)

    print(f"\nTraining MCCFR for {args.iterations:,} iterations...")
    rule = next((r for r in ALL_RULES if r.name == args.update_rule), None)
    if rule is None:
        raise SystemExit(f"--update-rule {args.update_rule}: custom settings exist only on the native path")
    solver = MCCFRSolver(game, rule=rule, seed=args.seed,
                         average_from=int(args.average_from * args.iterations))
    start = time.perf_counter()

    def checkpoint(done, total, live):
        """
        Report, and save what exists so far.

        Both halves are here because their absence cost a run: a 500,000
        iteration attempt was killed at six hours having printed nothing to
        size it by and written nothing to keep. Memory is the ceiling rather
        than time -- the abstraction's table is 4.7 MB and the solver's
        bookkeeping reached 4.9 GB -- so the resident size is reported too,
        being the number that predicts the ending.
        """
        taken = time.perf_counter() - start
        rss = _resident_mb()
        eta = (total - done) * taken / done / 60
        print(f"  {done:>8,}/{total:,}  {taken / done * 1000:5.1f} ms/it  "
              f"{len(live.nodes):>7,} infosets  {rss:6.0f} MB  "
              f"eta {eta:5.1f} min", flush=True)
        if args.output:
            # `<base>.partial.pkl`, so its summary lands beside it as
            # `<base>.partial.json` rather than colliding with the final one.
            _write(os.path.splitext(args.output)[0] + ".partial.pkl",
                   live.average_strategy(), abstraction, args, {}, done, taken)

    solver.train(args.iterations, on_progress=checkpoint,
                 progress_every=max(1, args.iterations // 50))
    elapsed = time.perf_counter() - start
    print(f"  {elapsed:.1f}s ({elapsed / args.iterations * 1000:.1f} ms/iteration)")
    print(f"  information sets reached: {len(solver.nodes):,} "
          f"of {projected.information_sets:,} in the abstraction")

    strategy = solver.average_strategy()
    _report_and_write(args, game, abstraction, strategy, len(solver.nodes), elapsed)


def _report_and_write(args, game, abstraction, strategy, information_sets, elapsed):
    """Score against the baselines and write the pickle, for either solver."""
    trained = strategy_policy(strategy)

    print(f"\nHead-to-head over {args.eval_hands:,} hands, seats alternating.")
    print("Positive means the trained strategy is winning.\n")

    results = {}
    for name, opponent in (("uniform random", uniform_policy()),
                           ("always call", always_call_policy())):
        outcome = play_hands(game, [trained, opponent], args.eval_hands,
                             np.random.default_rng(args.seed + 1))
        per_100_bb = outcome.mean / args.big_blind * 100
        print(f"  vs {name:<15} {outcome.summary()}")
        print(f"  {'':<18} = {per_100_bb:+.1f} BB/100")
        results[name] = {
            "chips_per_hand": outcome.mean,
            "stderr": outcome.stderr,
            "ci95": list(outcome.ci95),
            "bb_per_100": per_100_bb,
            "separated_from_zero": outcome.separated_from_zero,
        }

    print("\nNote: head-to-head chips is a weaker claim than exploitability.")
    print("Beating a passive baseline shows the strategy is not broken; it does")
    print("not show it is near equilibrium. LBR is the next piece of work.")

    if args.output:
        with open(args.output, "wb") as handle:
            pickle.dump({"strategy": strategy, "abstraction": abstraction,
                         "args": vars(args), "results": results}, handle)
        if _FLAT_EXPORT is not None:
            # The flat pair from the same arrays, so the loaders never touch
            # the dict form of a big rung again.
            npz, side = flat_paths(args.output)
            np.savez(npz, keys=_FLAT_EXPORT._keys, offsets=_FLAT_EXPORT._offsets, values=_FLAT_EXPORT._values)
            with open(side, "wb") as handle:
                pickle.dump({"abstraction": abstraction, "args": vars(args)}, handle)
        finished = dict(_PROVENANCE or {}, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
            json.dump({"args": vars(args), "results": results,
                       "information_sets_reached": information_sets,
                       "seconds": elapsed, "provenance": finished}, handle, indent=2)
        print(f"\nWrote {args.output}")
        # Its manifest entry (scripts/solve_manifest.py) now, while the files are fresh. Never fatal: the solve is
        # written either way, and a later full run of the manifest picks up anything this missed.
        try:
            import subprocess
            kept = os.path.realpath(args.output).startswith(
                os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "cfr")) + os.sep)
            if kept:     # a scratch or test solve is not part of the record
                subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "solve_manifest.py"),
                               "--only", args.output], timeout=600, check=False)
        except Exception as error:   # noqa: BLE001
            print(f"manifest not updated ({error}); run scripts/solve_manifest.py")


if __name__ == "__main__":
    main()
