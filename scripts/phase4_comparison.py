"""
Phase 4 — three agent families, one panel, one instrument.

The project's title promises a comparison and until today it could not be made:
PPO had trained but never been measured. It can be made now, and mostly by
assembly, because every family already recorded what it needs.

The budget axis is wall-clock, and it was already there
-------------------------------------------------------
The three families laddered along different axes — CFR along training seconds,
evolutionary search along generations, PPO along hands — and a comparison has to
declare one. Wall-clock is the only axis all three share, and none of them needs
re-running to supply it: the CFR budgets are seconds by construction,
`phase2_history.json` records `seconds` per generation, and each PPO history
records `seconds` per interval. So the axis is arithmetic over existing files.

Two hazards, both of which produce a plausible-looking wrong chart
------------------------------------------------------------------
**Units.** `results/cfr/strength_signals.json` is in chips/hand; the endpoint
tests are in BB/100. At big_blind 2 they differ by a factor of fifty, so plotting
them together unconverted shows the CFR agent as fifty times weaker than PPO
rather than stronger — an inversion of the project's central finding, in a chart
that looks fine. Everything is normalised to BB/100 at the boundary here, once.

**Evolution's CFR row exists twice.** `phase2_endpoint.log` reports +60.8 BB/100
against the solver beside a 74.3% lookup miss rate: the benchmark had quietly
become a second random opponent. `phase2_cfr_row.log` is the re-measurement after
that was fixed, at 0.0%, and reads −370.1. This script takes random and
always-call from the first file and the CFR row only from the second, and refuses
to run if the invalid row ever starts looking valid.

What is measured here rather than read
---------------------------------------
The CFR agent's own scores. Its numbers in `strength_signals.json` were taken at
3,000 hands over 2 seeds, against the endpoint tests' 40,000 over 3 — not the
same instrument, and the premise of this phase is one instrument. So it is
re-measured against random and always-call at 40,000 hands with the panel's own
seed. It has no score against itself, and that cell is left empty rather than
filled with a zero that would read as a measurement.

Usage
-----
    venv/bin/python scripts/phase4_comparison.py
    venv/bin/python scripts/phase4_comparison.py --hands 2000 --dry-run
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import numpy as np

from evaluation import always_call_agent, benchmark, cfr_agent, random_agent

#: The bar every endpoint in this project is taken at. ±14 BB/100.
HANDS = 40_000

#: The deals the PPO endpoint test used. Shared so the CFR row is played against
#: the same cards as the rows it is being put beside.
EVAL_SEED = 20260820

EVOLUTION_GENERATIONS = 50
#: Hands is the budget axis, not wall-clock.
#:
#: Wall-clock was the first choice because it is the only unit all three
#: families recorded, and it turned out to encode whatever else the machine was
#: doing: the same 8M-hand run read 4.81 h on a quiet box and 10.42 h when three
#: seeds shared it with another project's tuner. An axis that moves by a factor
#: of two on load is measuring the machine, not the algorithm.
#:
#: Hands played is exact, machine-independent, and the currency both learned
#: families actually spend. The solver is not on this axis at all -- it traverses
#: a tree rather than playing hands, and it is the opponent here rather than a
#: competitor.
EVOLUTION_POPULATION = 30
EVOLUTION_MATCHUPS = 4
EVOLUTION_HANDS_PER_MATCHUP = 6000
EVOLUTION_HANDS = (EVOLUTION_GENERATIONS * EVOLUTION_POPULATION
                   * EVOLUTION_MATCHUPS * EVOLUTION_HANDS_PER_MATCHUP)
PPO_SCRATCH = os.path.expanduser("~/pokerbot-scratch/phase3")

RESULTS = os.path.join(ROOT, "results")
CFR_STRATEGY = os.path.join(RESULTS, "cfr", "nolimit_strategy.pkl")


# ---------------------------------------------------------------------------
# The wall-clock axis
# ---------------------------------------------------------------------------

def evolution_seconds():
    """Cumulative training seconds over the fifty generations."""
    with open(os.path.join(RESULTS, "evolution", "phase2_history.json")) as handle:
        history = json.load(handle)
    if len(history) != EVOLUTION_GENERATIONS:
        raise SystemExit(f"expected {EVOLUTION_GENERATIONS} generations, "
                         f"found {len(history)}")
    return sum(row["seconds"] for row in history)


def ppo_seconds():
    """Mean cumulative seconds at each rung, across the three training seeds."""
    per_rung = {}
    for seed in (0, 1, 2):
        path = os.path.join(PPO_SCRATCH, f"seed{seed}", "history.json")
        with open(path) as handle:
            history = json.load(handle)
        cumulative = 0.0
        for row in history:
            cumulative += row["seconds"]
            if row["rung"]:
                per_rung.setdefault(row["hands"], []).append(cumulative)
    return {hands: float(np.mean(values)) for hands, values in per_rung.items()}


# ---------------------------------------------------------------------------
# Evolution, read from the logs its run produced
# ---------------------------------------------------------------------------

ROW = re.compile(
    r"^(random|always-call|cfr)\s+([-+][\d.]+)\s*\+/-\s*(\d+)\s+"
    r"([-+][\d.]+)\s*\+/-\s*(\d+)\s+([-+][\d.]+)\s*\+/-\s*(\d+)")


def evolution_endpoint(single_log=None):
    """One log if given, else the historical two-file split."""
    if single_log:
        with open(single_log) as handle:
            text = handle.read()
        rows = {}
        for line in text.splitlines():
            found = ROW.match(line.strip())
            if found:
                name, u, u_ci, t, t_ci, d, d_ci = found.groups()
                rows[name] = {"untrained": float(u), "untrained_ci95": float(u_ci),
                              "trained": float(t), "trained_ci95": float(t_ci),
                              "difference": float(d), "difference_ci95": float(d_ci),
                              "source": os.path.basename(single_log)}
        if set(rows) != {"random", "always-call", "cfr"}:
            raise SystemExit(f"expected three rows in {single_log}, "
                             f"parsed {sorted(rows)}")
        if "CFR lookup miss rate 0.0%" not in text:
            raise SystemExit(f"{single_log} does not report a 0.0% miss rate; "
                             "the CFR row cannot be trusted")
        return rows
    return _evolution_endpoint_split()


def _evolution_endpoint_split():
    """
    The fifty-generation endpoint, with the CFR row taken from the re-measurement.

    Parsed rather than transcribed: a number copied by hand into a comparison is
    a number nothing checks again.
    """
    endpoint = os.path.join(RESULTS, "evolution", "phase2_endpoint.log")
    with open(endpoint) as handle:
        text = handle.read()

    # The guard that keeps the withdrawn row withdrawn. If this file's CFR row
    # ever stops carrying its miss rate, the parse below is no longer safe to
    # trust and the failure should be loud.
    if "CFR lookup miss rate 74.3%" not in text:
        raise SystemExit(
            f"{endpoint} no longer carries the 74.3% miss rate that makes its "
            "CFR row invalid; re-check which rows in it can be used")

    rows = {}
    for line in text.splitlines():
        found = ROW.match(line.strip())
        if found:
            name, u, u_ci, t, t_ci, d, d_ci = found.groups()
            rows[name] = {"untrained": float(u), "untrained_ci95": float(u_ci),
                          "trained": float(t), "trained_ci95": float(t_ci),
                          "difference": float(d), "difference_ci95": float(d_ci),
                          "source": os.path.basename(endpoint)}
    if set(rows) != {"random", "always-call"}:
        raise SystemExit(f"expected random and always-call in {endpoint}, "
                         f"parsed {sorted(rows)}")

    corrected = os.path.join(RESULTS, "evolution", "phase2_cfr_row.log")
    with open(corrected) as handle:
        text = handle.read()
    numbers = re.findall(r"([-+][\d.]+)\s*\+/-\s*(\d+)", text)
    if len(numbers) != 3:
        raise SystemExit(f"expected three figures in {corrected}, "
                         f"parsed {len(numbers)}")
    (u, u_ci), (t, t_ci), (d, d_ci) = numbers
    rows["cfr"] = {"untrained": float(u), "untrained_ci95": float(u_ci),
                   "trained": float(t), "trained_ci95": float(t_ci),
                   "difference": float(d), "difference_ci95": float(d_ci),
                   "source": os.path.basename(corrected)}
    return rows


# ---------------------------------------------------------------------------
# CFR, measured here on the shared instrument
# ---------------------------------------------------------------------------

def measure_cfr(hands, strategy_path=CFR_STRATEGY):
    """
    The solver against the two baselines, at the endpoint tests' width.

    No row against itself: a strategy played against a copy of itself is zero by
    symmetry, and printing that zero would put a structural identity in a column
    of measurements.
    """
    import pickle
    with open(strategy_path, "rb") as handle:
        saved = pickle.load(handle)

    rows = {}
    for name, opponent in (("random", random_agent(np.random.default_rng(4))),
                           ("always-call", always_call_agent())):
        misses = [0, 0]
        agent = cfr_agent(saved["strategy"], saved["abstraction"],
                          np.random.default_rng(4), misses=misses, raise_cap=1)
        result = benchmark(agent, opponent, name, hands=hands, seed=EVAL_SEED)
        rate = misses[0] / misses[1] if misses[1] else 0.0
        rows[name] = {"bb_per_100": result.bb_per_100,
                      "ci95": 1.96 * result.stderr / 2 * 100,
                      "lookup_miss_rate": rate}
        print(f"  cfr vs {name:<12} {result.bb_per_100:+8.1f} BB/100 "
              f"(miss {rate:.1%})", flush=True)
    return rows


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", type=int, default=HANDS,
                        help="Hands for the CFR row. Below 40,000 is not a result.")
    parser.add_argument("--out", default=os.path.join(RESULTS, "comparison",
                                                      "phase4.json"))
    parser.add_argument("--dry-run", action="store_true")
    #: Which PPO endpoint to read. There are two now -- the pre-fix run and the
    #: refit under the corrected raise sizing -- and a comparison that does not
    #: say which it used cannot be checked against the other.
    parser.add_argument("--ppo-source",
                        default=os.path.join(RESULTS, "ppo", "phase3_endpoint.json"))
    #: Which solver plays the CFR column. It must be the same one PPO was
    #: measured against -- `train_ppo.build_panel` reads
    #: results/cfr/nolimit_strategy.pkl -- or the two halves of this table are
    #: about different opponents.
    parser.add_argument("--cfr-strategy", default=CFR_STRATEGY)
    #: A single endpoint log carrying all three rows. The default pair splits
    #: random/always-call from the CFR row because that row had to be redone at
    #: a 0.0% miss rate; a run measured cleanly in one pass does not need the
    #: split, and mixing panels across the two files would be worse than either.
    parser.add_argument("--evolution-log")
    args = parser.parse_args()

    if args.hands < HANDS and not args.dry_run:
        parser.error(f"--hands below {HANDS:,} is monitoring, not a result; "
                     "pass --dry-run if that is deliberate")

    evolution = evolution_endpoint(args.evolution_log)
    with open(args.ppo_source) as handle:
        ppo = json.load(handle)
    if ppo["hands_per_matchup"] != HANDS:
        raise SystemExit(f"PPO endpoint was taken at {ppo['hands_per_matchup']:,} "
                         f"hands, not {HANDS:,}; the rows are not comparable")

    print(f"measuring the CFR agent at {args.hands:,} hands, seed {EVAL_SEED}")
    cfr = measure_cfr(args.hands, args.cfr_strategy)

    seconds = {"evolution": evolution_seconds(), "ppo": ppo_seconds()}
    report(evolution, ppo, cfr, seconds)

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"hands_per_matchup": args.hands, "eval_seed": EVAL_SEED,
                       "units": "BB/100, big_blind 2",
                       "wall_clock_seconds": seconds,
                       "evolution": evolution, "cfr": cfr,
                       "ppo_source": os.path.basename(args.ppo_source),
                       "cfr_strategy": os.path.basename(args.cfr_strategy)},
                      handle, indent=1)
        print(f"\nwrote {args.out}")


def ppo_by_rung(ppo):
    """Mean trained score and mean gain per rung, per opponent, across seeds."""
    out = {}
    for rung in ppo["rungs"]:
        rows = [row for record in ppo["records"]
                if record["hands_trained"] == rung for row in record["rows"]]
        out[rung] = {}
        for name in ppo["panel"]:
            vals = [r for r in rows if r["opponent"] == name]
            trained = [v["trained"] for v in vals]
            gain = [v["difference"] for v in vals]
            out[rung][name] = {"trained": float(np.mean(trained)),
                               "gain": float(np.mean(gain)),
                               "spread": max(gain) - min(gain)}
    return out


def report(evolution, ppo, cfr, seconds):
    rungs = ppo_by_rung(ppo)
    print("\n" + "=" * 78)
    print("Phase 4 — every family against the same panel, at 40,000 hands, in BB/100")
    print("=" * 78)
    print(f"{'family':<22}{'hands':>13}{'vs random':>13}"
          f"{'vs always-call':>16}{'vs CFR':>12}")
    print("-" * 78)

    print(f"{'CFR (the solver)':<22}{'--':>13}"
          f"{cfr['random']['bb_per_100']:>+13.1f}"
          f"{cfr['always-call']['bb_per_100']:>+16.1f}{'--':>12}")

    print(f"{'evolution, 50 gens':<22}{EVOLUTION_HANDS:>13,}"
          f"{evolution['random']['trained']:>+13.1f}"
          f"{evolution['always-call']['trained']:>+16.1f}"
          f"{evolution['cfr']['trained']:>+12.1f}")

    for rung in ppo["rungs"]:
        print(f"{'PPO':<22}{rung:>13,}"
              f"{rungs[rung]['random']['trained']:>+13.1f}"
              f"{rungs[rung]['always-call']['trained']:>+16.1f}"
              f"{rungs[rung]['cfr']['trained']:>+12.1f}")

    print("-" * 78)
    print("'vs CFR' is the column to read: the two learned families are scored")
    print("against an opponent from outside their lineage, and its seed spreads")
    print("are three to ten times tighter than the baselines'. The solver has no")
    print("row against itself, which is a structural identity and not a result.\n")

    # The comparison the budget axis makes available, and the reason it was
    # worth putting the two families on one.
    smallest = min(ppo["rungs"])
    evo = evolution["cfr"]
    print(f"PPO scores {rungs[smallest]['cfr']['trained']:+.1f} against the "
          f"solver on {smallest:,} hands.")
    print(f"Evolutionary search spent {EVOLUTION_HANDS:,} — "
          f"{EVOLUTION_HANDS / smallest:.0f} times as many — and scores "
          f"{evo['trained']:+.1f},")
    # Computed, not asserted. This read "which is no change" for as long as
    # evolution was measured against a 4,000-iteration solver, where the gain
    # was +33.8 +/- 37. Against a converged one it is +50.0 +/- 20 and
    # separated, and a hardcoded verdict would have gone on denying it.
    separated = abs(evo["difference"]) > evo["difference_ci95"]
    print(f"a gain of {evo['difference']:+.1f} +/- {evo['difference_ci95']:.0f} "
          f"over its own untrained network, which is "
          f"{'a real improvement' if separated else 'no change'}.")
    print()
    print("Wall-clock is deliberately absent. It read 4.81 h for the 8M rung on")
    print("a quiet machine and 10.42 h for the same run sharing cores, so it")
    print("measures the load rather than the algorithm. Hands are exact.")


if __name__ == "__main__":
    main()
