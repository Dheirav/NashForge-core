"""
Given N seconds of training, which bucketing should you use?

This is the same question `crossover.py` asks, answered a different way. That
script measures each abstraction's exploitability and compares the two numbers.
It could not reach a conclusion: Local Best Response is confined to the same
abstraction as the strategy it measures, so it evaluates each on its own terms
and went slack at every budget above the shallowest.

Playing the two agents against each other sidesteps that entirely. Train one on
Monte-Carlo equity and one on made-hand strength, give both the same wall-clock
budget, and count chips. Whoever wins has answered the practical question.

**Each agent is asked the question its own bucketing poses.** The two live in
different abstractions, so handing one the other's information-set key would make
it look like a bad player rather than a mismatched lookup. `strategy_policy`
takes an abstraction for exactly this reason, and routes through
`NoLimitHoldem.information_set_with`. Nothing about the *game* is abstracted —
cards are dealt for real and showdowns settled on real hands — so both agents
play the same poker and only their view of it differs.

**The budget axis is wall-clock, not iterations.** Made-hand bucketing is far
cheaper per iteration, so an equal-seconds budget buys it far more iterations.
That is the trade-off being measured; counting iterations instead would compare
the maps while ignoring that one takes five times longer to read.

What this does and does not establish
-------------------------------------
Beating the other abstraction does **not** show either is near equilibrium. Two
weak strategies can be compared and one will win. This answers "which should I
use", not "how good is it" — the second needs exploitability, which is what
`crossover.py` is for and why fixing LBR remains open. See BACKLOG.md item 1.

Every figure carries a standard error and seats alternate, because a chip count
over a few thousand hands of poker is dominated by variance rather than skill.

Usage
-----
    python scripts/cfr/head_to_head.py
    python scripts/cfr/head_to_head.py --budgets 40 160 --seeds 2 --hands 4000
"""
import argparse
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

#: policies[0] is the equity agent, so a positive result favours equity.
SIGNALS = ("equity", "made_hand")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=float, nargs="+", default=[40, 160, 640, 2560],
                        help="cumulative training seconds at which to play")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--buckets", type=int, default=6)
    parser.add_argument("--raise-cap", type=int, default=1)
    parser.add_argument("--equity-samples", type=int, default=40)
    parser.add_argument("--hands", type=int, default=3000,
                        help="hands per matchup; seats alternate within each")
    parser.add_argument("--output", default="results/cfr/head_to_head.json")
    return parser.parse_args()


def save(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def load(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {}


def load_average():
    try:
        with open("/proc/loadavg") as handle:
            return float(handle.read().split()[0])
    except Exception:
        return float("nan")


def build(signal, seed, args):
    """A fitted abstraction, its game, and a solver ready to train."""
    rng = np.random.default_rng(seed)
    abstraction = CardAbstraction(
        preflop_buckets=args.buckets, postflop_buckets=args.buckets,
        samples=800, equity_samples=args.equity_samples, strength=signal).fit(rng)
    game = NoLimitHoldem(abstraction, raise_cap=args.raise_cap,
                         equity_samples=args.equity_samples)
    return abstraction, game, MCCFRSolver(game, rule=VANILLA, seed=seed)


def train_to(solver, seconds, sample_every=10.0):
    """
    Train until ``seconds`` more have elapsed.

    Load is sampled *during* the window rather than read once afterwards. The
    two signals are trained one after the other, so a spike landing in one
    agent's window and not the other's biases that rung directly — and a single
    reading taken after both have finished cannot show it happened.

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


def run_seed(seed, budgets, args, on_measurement=None):
    """Train both signals in lockstep and play them off at each budget."""
    built = {signal: build(signal, seed, args) for signal in SIGNALS}
    solvers = {signal: built[signal][2] for signal in SIGNALS}

    # Either game will do: dynamics are identical, and each agent is asked
    # through its own abstraction rather than the game's.
    arena = built["equity"][1]

    measurements = []
    spent = 0.0
    for budget in budgets:
        elapsed = {}
        gained = {}
        load = {}
        for signal in SIGNALS:
            elapsed[signal], gained[signal], load[signal] = train_to(
                solvers[signal], budget - spent)
        spent = budget

        policies = [strategy_policy(solvers[signal].average_strategy(),
                                    built[signal][0]) for signal in SIGNALS]
        outcome = play_hands(arena, policies, args.hands,
                             np.random.default_rng(9000 + seed))

        row = {
            "budget": budget,
            "chips_per_hand_to_equity": outcome.mean,
            "stderr": outcome.stderr,
            "ci95": list(outcome.ci95),
            "separated_from_zero": outcome.separated_from_zero,
            "hands": outcome.hands,
            # Kept as the mean of the two windows so old and new records stay
            # comparable; the per-signal figures are what a fairness check needs.
            "load_avg": statistics.fmean(load.values()),
            "load_imbalance": abs(load[SIGNALS[0]] - load[SIGNALS[1]]),
        }
        for signal in SIGNALS:
            row[f"{signal}_iterations"] = solvers[signal].iterations
            row[f"{signal}_ms_per_iteration"] = (elapsed[signal]
                                                 / max(1, gained[signal]) * 1000)
            row[f"{signal}_load_avg"] = load[signal]
        measurements.append(row)

        winner = ("equity" if outcome.separated_from_zero and outcome.mean > 0
                  else "made_hand" if outcome.separated_from_zero
                  else "not separated")
        # A rung where the two agents trained under noticeably different load is
        # one where the budget was not really equal. Say so at the time rather
        # than leaving it to be discovered in the JSON afterwards.
        skew = ("" if row["load_imbalance"] < 1.0 else
                f"  [!] load {load['equity']:.1f} vs {load['made_hand']:.1f}")
        print(f"      seed {seed}  {budget:>6.0f}s  "
              f"equity {solvers['equity'].iterations:>7,} it  vs  "
              f"made_hand {solvers['made_hand'].iterations:>7,} it   "
              f"{outcome.mean:+7.3f} +/- {outcome.stderr:.3f} chips/hand  "
              f"{winner}{skew}", flush=True)

        if on_measurement is not None:
            on_measurement(seed, measurements)

    return measurements


def aggregate(runs, index):
    """Pool one budget across seeds, as crossover.py does."""
    values = [run[index]["chips_per_hand_to_equity"] for run in runs]
    errors = [run[index]["stderr"] for run in runs]
    count = len(values)

    within = math.sqrt(sum(e * e for e in errors)) / count
    between = statistics.stdev(values) / math.sqrt(count) if count > 1 else 0.0
    stderr = max(within, between)
    mean = statistics.fmean(values)

    return {"mean": mean, "stderr": stderr,
            "separated": abs(mean) > 1.96 * stderr,
            "equity_iterations": statistics.fmean(
                run[index]["equity_iterations"] for run in runs),
            "made_hand_iterations": statistics.fmean(
                run[index]["made_hand_iterations"] for run in runs)}


def main():
    args = parse_args()
    budgets = sorted(args.budgets)

    print("Head to head: equity bucketing vs made-hand bucketing.")
    print(f"{args.seeds} seeds, {args.hands:,} hands per matchup, seats alternating.")
    print("Positive favours equity. Both agents get the same wall-clock budget.\n")

    stored = load(args.output)
    raw = dict(stored.get("per_seed", {}))

    def checkpoint(seed, measurements):
        raw[str(seed)] = measurements
        save(args.output, {"args": vars(args), "per_seed": raw, "summary": []})

    for seed in range(args.seeds):
        if str(seed) in raw and len(raw[str(seed)]) == len(budgets):
            print(f"      seed {seed}  already on disk, skipping", flush=True)
            continue
        run_seed(seed, budgets, args, on_measurement=checkpoint)
        print()

    runs = list(raw.values())
    print(f"\n{'budget':>8}{'equity iters':>16}{'made_hand iters':>18}"
          f"{'chips/hand':>16}{'verdict':>18}")
    print("-" * 76)

    summary = []
    for index, budget in enumerate(budgets):
        row = aggregate(runs, index)
        row["budget"] = budget
        verdict = ("equity ahead" if row["separated"] and row["mean"] > 0
                   else "made_hand ahead" if row["separated"]
                   else "not separated")
        row["verdict"] = verdict
        print(f"{budget:>7.0f}s{row['equity_iterations']:>16,.0f}"
              f"{row['made_hand_iterations']:>18,.0f}"
              f"{row['mean']:>10.3f} +/-{row['stderr']:<5.3f}{verdict:>18}")
        summary.append(row)

    print()
    decided = [r for r in summary if r["separated"]]
    if not decided:
        print(f"No budget separated the two. With {args.hands:,} hands the noise "
              f"floor is about {summary[0]['stderr'] * 1.96:.2f} chips/hand; a "
              f"real difference smaller than that cannot be seen from here, so "
              f"raise --hands rather than reading the signs of these means.")
    else:
        flips = [r for r in decided if (r["mean"] > 0) != (decided[0]["mean"] > 0)]
        first = decided[0]
        ahead = "equity" if first["mean"] > 0 else "made_hand"
        print(f"{ahead} is ahead from {first['budget']:.0f}s.", end=" ")
        if flips:
            print(f"The lead changes hands at {flips[0]['budget']:.0f}s — that "
                  f"crossover is the result worth reporting.")
        else:
            print("No crossover: the same signal leads at every budget where the "
                  "two separate.")

    save(args.output, {"args": vars(args), "per_seed": raw, "summary": summary})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
