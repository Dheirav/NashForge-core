"""
How many card buckets should the solver use, given N seconds?

Six buckets is what every figure in this project was produced with, and it was
never chosen. Heads-up no-limit has 169 distinct starting hands before a board
is dealt, and this abstraction sorts all of them, plus every flop, turn and
river, into six classes. `abstraction/betting.py` states the consequence
directly: **action and card abstraction upper-bound achievable exploitability
regardless of how well the solver runs.** After the 200bb retrain, training and
stack depth are both spent as levers against Slumbot, which leaves the
abstraction as the binding constraint and this as the cheaper of its two
dimensions.

Cheap in memory, that is. From `results/cfr/abstraction_size.json` at raise cap 1:

    6 buckets      49,200 information sets     4.7 MB
    20 buckets    164,000 information sets    15.7 MB
    50 buckets    410,000 information sets    39.4 MB

Nothing there is a constraint on this machine. The cost is elsewhere: the same
traversals now have to cover eight times as many decisions, so a finer
abstraction is a better map that the solver has less time to read.

Why the budget axis is wall-clock
---------------------------------
A finer abstraction covers a smaller share of a larger tree per iteration, and
may also cost more per iteration to read. Wall-clock charges both together, and
"given N seconds, how many buckets" is the question a practitioner actually
faces. Budgeting by iterations would waive whichever part of the cost is
per-iteration and hand the finer arm more CPU for the same nominal budget.

**The per-iteration part is negative here, which was worth measuring rather than
assuming.** Across six rungs and three seeds on 11 September, 20 buckets ran
cheaper per iteration than 6 at every rung and bought 7 to 13 percent MORE
traversals at equal wall-clock. So the budget axis favours the finer arm, and a
finer arm that still loses is losing on the partition rather than on compute.
This is why every rung records `{left,right}_ms_per_iteration` per arm across
every seed: the sweep answers it as a side effect instead of leaving it as an
assumption in a docstring.

**This makes the axis sensitive to the machine, which is a real hazard here.**
This project already withdrew a wall-clock axis once: an identical 8M-hand run
read 4.81 h on a quiet machine and 10.42 h sharing cores with another job. Every
rung records the load average sampled *during* its own window, per arm, and
flags an imbalance above 1.0. A sweep run under heavy or varying contention is
measuring the machine and its rows should not be pooled with a quiet one's.

It is the axis `crossover.py` and `head_to_head.py` already use, which keeps this
result comparable to the abstraction crossover rather than making it a separate
thing.

Why head to head rather than exploitability
-------------------------------------------
Local Best Response is confined to the same abstraction as the strategy it
measures, so it evaluates each arm on its own terms and goes slack. That
investigation was closed with no usable bound. Playing the arms against each
other sidesteps it, at the cost of the weaker claim: whoever wins has answered
"which should I use", not "how good is it".

**Each arm is asked the question its own bucketing poses.** Two bucket counts
produce different information-set keys, so handing one arm the other's key would
make it look like a bad player rather than a mismatched lookup. `strategy_policy`
takes an abstraction for exactly this reason. Nothing about the *game* is
abstracted: cards are dealt for real and showdowns settled on real hands, so
every arm plays the same poker and only its view of it differs.

What was expected, and what happened
------------------------------------
Written before the first run: if bucket count behaves the way the strength signal
did, the fine arms lose at the short budgets and cross above the coarse ones
later, and **if 6 buckets wins at every budget, that is a real finding** pointing
the next effort at the raise cap instead.

It won at every budget. Six against twenty over a 128-fold range of budgets,
40s to 5120s, three seeds: the gap declines from +3.307 to about +0.5 and then
plateaus, separated from zero at the top rung (+0.624 +/- 0.259 chips/hand, about
31 BB/100). Fifty buckets lost to twenty at every rung with no closing trend and
was dropped after the first sweep. The card abstraction is not the constraint.

The top rung is 134,500 iterations against the shipped solver's 250,000, so this
is 55 percent of production budget. The plateau across the last three rungs is
what carries the conclusion, not arrival at production scale.

**And the result is conditional on the equity estimator, which was measured
afterwards and is noisier than the partition it feeds.** At `equity_samples=40`
the estimate's standard deviation is 0.0667 while adjacent bucket centroids sit
0.101 to 0.170 apart, so 42 percent of situations change bucket on a re-roll.
Fifty buckets would space centroids about 0.02 apart, three times inside that
noise, which means assignment at fifty was close to random. This sweep therefore
answers how many buckets the estimator can resolve, not how many are worth
having. Re-run it once `equity_samples` is raised.

Usage
-----
    python scripts/cfr/bucket_sweep.py
    python scripts/cfr/bucket_sweep.py --buckets 6 20 --budgets 40 160 --seeds 2
"""
import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.buckets import CardAbstraction  # noqa: E402
from cfr import MCCFRSolver, VANILLA  # noqa: E402
from cfr.play import play_hands, strategy_policy  # noqa: E402
from games.nolimit import NoLimitHoldem  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buckets", type=int, nargs="+", default=[6, 20, 50],
                        help="bucket counts to compare; 6 is what shipped")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[40, 160, 640, 2560],
                        help="cumulative training seconds at which to play")
    parser.add_argument("--seeds", type=int, default=3)
    #: Which seeds specifically, when --seeds is not the right shape of answer.
    #: A seed whose budgets were spent on a contended machine bought a fraction
    #: of the work at the same nominal budget, and because budgets are
    #: cumulative it never catches up. Re-running that seed alone is the repair;
    #: re-running all of them throws away the clean ones.
    parser.add_argument("--seed-list", type=int, nargs="+",
                        help="explicit seeds, overriding --seeds")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="pool separately-run seeds into one report")
    parser.add_argument("--raise-cap", type=int, default=1)
    parser.add_argument("--equity-samples", type=int, default=40)
    parser.add_argument("--abstraction-samples", type=int, default=800)
    parser.add_argument("--hands", type=int, default=3000,
                        help="hands per matchup; seats alternate within each")
    parser.add_argument("--output", default="results/cfr/bucket_sweep.json")
    return parser.parse_args()


def save(path, payload):
    """Written atomically, because this run outlasts a Windows Update window."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def load_average():
    try:
        with open("/proc/loadavg") as handle:
            return float(handle.read().split()[0])
    except Exception:
        return float("nan")


def build(buckets, seed, args):
    """A fitted abstraction, its game, and a solver ready to train."""
    rng = np.random.default_rng(seed)
    abstraction = CardAbstraction(
        preflop_buckets=buckets, postflop_buckets=buckets,
        samples=args.abstraction_samples,
        equity_samples=args.equity_samples).fit(rng)
    game = NoLimitHoldem(abstraction, raise_cap=args.raise_cap,
                         equity_samples=args.equity_samples)
    return abstraction, game, MCCFRSolver(game, rule=VANILLA, seed=seed)


def train_to(solver, seconds, sample_every=10.0):
    """
    Train until ``seconds`` more have elapsed.

    Load is sampled *during* the window rather than read once afterwards. The
    arms are trained one after another, so a spike landing in one arm's window
    and not another's biases that rung directly, and a single reading taken once
    every arm has finished cannot show that it happened. This machine shares
    cores with other training jobs, so it is not a hypothetical.

    Returns (elapsed, iterations gained, mean load over the window).
    """
    before = solver.iterations
    started = time.perf_counter()
    deadline = started + seconds

    samples = []
    next_sample = started
    while time.perf_counter() < deadline:
        solver.train(25)
        now = time.perf_counter()
        if now >= next_sample:
            samples.append(load_average())
            next_sample = now + sample_every

    elapsed = time.perf_counter() - started
    load = statistics.fmean(samples) if samples else load_average()
    return elapsed, solver.iterations - before, load


def run_seed(seed, args, on_measurement=None):
    """Train every arm in lockstep and play each pair off at each budget."""
    built = {b: build(b, seed, args) for b in args.buckets}
    solvers = {b: built[b][2] for b in args.buckets}

    # Any arm's game will do: the dynamics are identical, and each agent is
    # asked through its own abstraction rather than through the game's.
    arena = built[args.buckets[0]][1]

    measurements = []
    spent = 0.0
    for budget in args.budgets:
        elapsed, gained, load = {}, {}, {}
        for buckets in args.buckets:
            elapsed[buckets], gained[buckets], load[buckets] = train_to(
                solvers[buckets], budget - spent)
        spent = budget

        policies = {b: strategy_policy(solvers[b].average_strategy(), built[b][0])
                    for b in args.buckets}

        for left, right in itertools.combinations(args.buckets, 2):
            outcome = play_hands(arena, [policies[left], policies[right]],
                                 args.hands, np.random.default_rng(9000 + seed))
            row = {
                "seed": seed,
                "budget": budget,
                "left_buckets": left,
                "right_buckets": right,
                #: Positive favours `left`, which is the coarser arm, because
                #: --buckets is given in increasing order. A positive sweep is
                #: therefore "the coarse abstraction is still winning".
                "chips_per_hand_to_left": outcome.mean,
                "stderr": outcome.stderr,
                "ci95": list(outcome.ci95),
                "separated_from_zero": outcome.separated_from_zero,
                "hands": outcome.hands,
                "left_iterations": solvers[left].iterations,
                "right_iterations": solvers[right].iterations,
                "left_information_sets": len(solvers[left].nodes),
                "right_information_sets": len(solvers[right].nodes),
                "left_ms_per_iteration": elapsed[left] / max(1, gained[left]) * 1000,
                "right_ms_per_iteration": elapsed[right] / max(1, gained[right]) * 1000,
                "left_load_avg": load[left],
                "right_load_avg": load[right],
                # A rung whose arms trained under noticeably different load is a
                # rung where the budget was not really equal.
                "load_imbalance": abs(load[left] - load[right]),
            }
            measurements.append(row)

            winner = (f"{left}" if outcome.separated_from_zero and outcome.mean > 0
                      else f"{right}" if outcome.separated_from_zero
                      else "not separated")
            skew = ("" if row["load_imbalance"] < 1.0 else
                    f"  [!] load {load[left]:.1f} vs {load[right]:.1f}")
            print(f"      seed {seed}  {budget:>6.0f}s  "
                  f"{left:>3} buckets {solvers[left].iterations:>7,} it  vs  "
                  f"{right:>3} buckets {solvers[right].iterations:>7,} it   "
                  f"{outcome.mean:+7.3f} +/- {outcome.stderr:.3f} chips/hand  "
                  f"wins: {winner}{skew}", flush=True)
            if on_measurement:
                on_measurement(row)

    return measurements


def pool(rows):
    """
    One pair at one budget, pooled across seeds.

    The seeds are independent runs of the same configuration, so their spread is
    the error bar that describes the configuration. A single seed's stderr
    describes only that measurement, and quoting it for a multi-seed claim
    understates the uncertainty roughly threefold. This project has made that
    mistake and published from it.
    """
    means = [row["chips_per_hand_to_left"] for row in rows]
    mean = statistics.fmean(means)
    if len(means) > 1:
        stderr = statistics.stdev(means) / math.sqrt(len(means))
    else:
        stderr = float(rows[0]["stderr"])
    half = 1.96 * stderr
    return {
        "left_buckets": rows[0]["left_buckets"],
        "right_buckets": rows[0]["right_buckets"],
        "budget": rows[0]["budget"],
        "seeds": len(means),
        "chips_per_hand_to_left": mean,
        "stderr": stderr,
        "ci95": [mean - half, mean + half],
        "separated_from_zero": abs(mean) > half,
        "per_seed": means,
        # A rung whose arms trained under different load did not really get
        # equal budgets, whatever the label says. Carried through pooling so it
        # reaches the printed table instead of staying in the JSON.
        "contended_seeds": sum(1 for row in rows if row["load_imbalance"] >= 1.0),
        "worst_load_imbalance": max(row["load_imbalance"] for row in rows),
    }


def report(measurements, args):
    print(f"\n{'=' * 78}")
    print("pooled across seeds, chips/hand to the coarser arm\n")
    for left, right in itertools.combinations(args.buckets, 2):
        print(f"  {left} buckets vs {right} buckets")
        for budget in args.budgets:
            rows = [m for m in measurements
                    if m["left_buckets"] == left and m["right_buckets"] == right
                    and m["budget"] == budget]
            if not rows:
                continue
            entry = pool(rows)
            verdict = ("not separated" if not entry["separated_from_zero"]
                       else f"{left} buckets" if entry["chips_per_hand_to_left"] > 0
                       else f"{right} buckets")
            each = ", ".join(f"{v:+.3f}" for v in entry["per_seed"])
            mark = ("" if not entry["contended_seeds"] else
                    f"  [!] {entry['contended_seeds']}/{entry['seeds']} seeds "
                    f"contended, worst {entry['worst_load_imbalance']:.1f}")
            print(f"    {budget:>6.0f}s  {entry['chips_per_hand_to_left']:+7.3f} "
                  f"+/- {1.96 * entry['stderr']:.3f}   {verdict:<14} "
                  f"[{each}]{mark}")
        print()

    if args.seeds < 3:
        print("Fewer than three seeds: the pooled interval is not yet an "
              "estimate of anything.")
    contended = [m for m in measurements if m["load_imbalance"] >= 1.0]
    if contended:
        print(f"[!] {len(contended)} of {len(measurements)} matchups trained "
              f"under a load imbalance above 1.0. Those arms did not get equal\n"
              f"    budgets whatever the label says, and the bias runs against "
              f"whichever arm was starved.\n")
    print("Beating another bucket count does not show either arm is near "
          "equilibrium.\nThis answers which to use, not how good it is.")


def merge(args):
    """
    Pool seeds that were run as separate processes, refusing a mismatch.

    The four fields checked are the ones that change what is being measured. Two
    sweeps at different budgets or bucket counts are not seeds of one experiment,
    and averaging them would bury a change of instrument inside a spread that
    looks like ordinary seed variation.
    """
    loaded = []
    for path in args.merge:
        with open(path) as handle:
            loaded.append((path, json.load(handle)))

    first_path, first = loaded[0]
    for path, data in loaded[1:]:
        for field in ("buckets", "budgets", "hands", "raise_cap"):
            if data["args"][field] != first["args"][field]:
                raise SystemExit(
                    f"cannot merge: {field} differs\n"
                    f"  {first_path}: {first['args'][field]}\n"
                    f"  {path}: {data['args'][field]}")

    rows = [row for _, data in loaded for row in data["measurements"]]
    seeds = sorted({row["seed"] for row in rows})
    merged = argparse.Namespace(**{**first["args"], "seeds": len(seeds)})
    print(f"merged {len(rows)} matchups from seeds {seeds}\n")
    report(rows, merged)
    save(args.output, {"args": vars(merged), "measurements": rows,
                       "complete": True, "merged_from": list(args.merge)})
    print(f"\nwrote {args.output}")


def main():
    args = parse_args()
    if sorted(args.buckets) != args.buckets:
        raise SystemExit("--buckets must be increasing, so a positive result "
                         "always favours the coarser arm")

    if args.merge:
        return merge(args)

    seeds = args.seed_list if args.seed_list else list(range(args.seeds))
    args.seeds = len(seeds)

    print(f"bucket sweep: {args.buckets} buckets, raise cap {args.raise_cap}")
    print(f"budgets {args.budgets} seconds, cumulative, {args.seeds} seeds")
    print(f"{args.hands:,} hands per matchup, seats alternating")
    print(f"seeds {seeds}")
    total = max(args.budgets) * len(args.buckets) * len(seeds)
    print(f"training alone is {total:,.0f}s = {total / 3600:.1f}h by "
          f"construction; play is on top\n", flush=True)

    measurements = []

    def checkpoint(row):
        measurements.append(row) if row not in measurements else None
        save(args.output, {"args": vars(args), "measurements": measurements,
                           "complete": False})

    for seed in seeds:
        print(f"  seed {seed}", flush=True)
        run_seed(seed, args, on_measurement=lambda row: checkpoint(row))

    report(measurements, args)
    save(args.output, {"args": vars(args), "measurements": measurements,
                       "complete": True})
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
