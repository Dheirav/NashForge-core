"""
Why did betting outside the abstraction buy nothing?

Off-tree LBR measured -0.415 against an on-tree control of -0.402 on the
59,050-iteration strategy: no gain. Instrumenting the choice showed the sizes
*are* being used — 47.8% of decisions — but 97% of those are the smallest size
offered, 0.25x pot. That is below the abstraction's minimum of 0.5, so it is
perceived as a half-pot bet deterministically: LBR pays a quarter pot and is
defended against as though it bet half. Maximum fold equity per chip.

Two questions follow, and this answers both.

**Phase 1 — is that a choice or a floor effect?** If LBR simply picks whatever
the smallest offered size happens to be, the behaviour is an artifact of where
the grid stops rather than a considered exploit, and every conclusion drawn from
it is about my parameter choice. Offering progressively lower floors settles it.

**Phase 2 — is `raise_cap=1` the real constraint?** Every measurement in this
project has allowed at most one raise per street, which is the least sizing
structure a game can have. An exploiter cannot punish bet sizing in a game that
barely has any. This trains at cap 1 and cap 2 and measures both exploiters
against each.

Training is counted in ITERATIONS, never wall-clock, so a busy machine changes
how long this takes but not what it produces. Everything checkpoints to this
directory — which is not /tmp, because /tmp did not survive the last reboot and
took 49 minutes of training with it.

    python action_abstraction_study.py            # start or resume
    python action_abstraction_study.py --restart  # discard and begin again
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/dheirav/Code/PokerBot")

import numpy as np

from abstraction.betting import ACTION_NAMES
from abstraction.buckets import CardAbstraction
from cfr import MCCFRSolver, VANILLA
from cfr.lbr import DEFAULT_BET_SIZES, LocalBestResponse, lbr_value
from games.nolimit import NoLimitHoldem

ITERATIONS = 59_050          # matches the deep run, so cap 1 is comparable
CHUNK = 2_500
HANDS = 1000
ROLLOUTS = 20
CANDIDATES = 32
SEED = 0

#: Abstract raise sizes, so restricting LBR to these makes translation the
#: identity and turns it into an on-tree exploiter through the same code path.
ON_TREE = (0.5, 1.0, 2.0)

#: Phase 1 grids. Same shape, progressively lower floor.
FLOORS = {
    "floor 0.25 (default)": DEFAULT_BET_SIZES,
    "floor 0.10": (0.10,) + DEFAULT_BET_SIZES[1:],
    "floor 0.02": (0.02,) + DEFAULT_BET_SIZES[1:],
}

RESULTS = os.path.join(HERE, "action_abstraction_study.json")


class Counting(LocalBestResponse):
    """LocalBestResponse that records every move it settles on."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.picks = collections.Counter()

    def _choose(self, state, me, candidate_range, rng):
        move = super()._choose(state, me, candidate_range, rng)
        key = (f"abstract:{ACTION_NAMES[move.action]}" if move.fraction is None
               else f"off-tree:{move.fraction:g}")
        self.picks[key] += 1
        return move


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            return json.load(handle)
    return {}


def save_results(results):
    temporary = f"{RESULTS}.tmp"
    with open(temporary, "w") as handle:
        json.dump(results, handle, indent=2)
    os.replace(temporary, RESULTS)


def load_average():
    try:
        with open("/proc/loadavg") as handle:
            return float(handle.read().split()[0])
    except Exception:
        return float("nan")


def build(raise_cap):
    rng = np.random.default_rng(SEED)
    abstraction = CardAbstraction(preflop_buckets=6, postflop_buckets=6,
                                  samples=800, equity_samples=40,
                                  strength="equity").fit(rng)
    game = NoLimitHoldem(abstraction, raise_cap=raise_cap, equity_samples=40)
    return game, MCCFRSolver(game, rule=VANILLA, seed=SEED)


def trained_solver(raise_cap):
    """Train to ITERATIONS, resuming from a checkpoint if one exists."""
    path = os.path.join(HERE, f"solver_cap{raise_cap}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as handle:
            solver = pickle.load(handle)
        if solver.iterations >= ITERATIONS:
            print(f"  cap {raise_cap}: {solver.iterations:,} iterations already "
                  f"trained", flush=True)
            return solver
        print(f"  cap {raise_cap}: resuming at {solver.iterations:,}", flush=True)
    else:
        _, solver = build(raise_cap)
        print(f"  cap {raise_cap}: training to {ITERATIONS:,} iterations",
              flush=True)

    started = time.perf_counter()
    at_start = solver.iterations
    while solver.iterations < ITERATIONS:
        step = min(CHUNK, ITERATIONS - solver.iterations)
        chunk_started = time.perf_counter()
        solver.train(step)

        temporary = f"{path}.tmp"
        with open(temporary, "wb") as handle:
            pickle.dump(solver, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

        done = solver.iterations
        rate = (time.perf_counter() - chunk_started) / step * 1000
        overall = time.perf_counter() - started
        remaining = (ITERATIONS - done) / max(1, done - at_start) * overall
        print(f"    {done:>7,} / {ITERATIONS:,}  {rate:5.1f} ms/iter  "
              f"load {load_average():4.1f}  ~{remaining / 60:.0f} min left",
              flush=True)

    return solver


# ---------------------------------------------------------------------------
# Phase 1 — is the smallest size a choice or a floor effect?
# ---------------------------------------------------------------------------

def phase_one(results):
    print("\nPHASE 1 — does LBR choose 0.25x pot, or merely the smallest offered?")
    print("Lowering the floor. If the share simply follows it down, the "
          "behaviour is an artifact of where the grid stops.\n", flush=True)

    saved = pickle.load(open(
        "/home/dheirav/Code/PokerBot/results/cfr/nolimit_strategy.pkl", "rb"))
    game = NoLimitHoldem(saved["abstraction"],
                         starting_stack=saved["args"]["stack"],
                         big_blind=saved["args"]["big_blind"],
                         raise_cap=saved["args"]["raise_cap"],
                         equity_samples=saved["args"]["equity_samples"])

    rows = results.setdefault("phase_one", {})
    print(f"{'grid':<24}{'LBR':>18}{'at floor':>11}{'off-tree':>11}")
    print("-" * 64)

    for label, sizes in FLOORS.items():
        if label not in rows:
            lbr = Counting(game, saved["strategy"], rollout_samples=ROLLOUTS,
                           candidates=CANDIDATES, bet_sizes=sizes)
            outcome = lbr.play(400, np.random.default_rng(7000))
            total = sum(lbr.picks.values())
            floor_key = f"off-tree:{min(sizes):g}"
            rows[label] = {
                "sizes": list(sizes),
                "mean": outcome.mean, "stderr": outcome.stderr,
                "at_floor": lbr.picks[floor_key] / total,
                "off_tree": sum(c for k, c in lbr.picks.items()
                                if k.startswith("off-tree")) / total,
                "picks": dict(lbr.picks),
            }
            save_results(results)

        row = rows[label]
        print(f"{label:<24}{row['mean']:>+10.3f} +/-{row['stderr']:<5.3f}"
              f"{row['at_floor']:>10.1%}{row['off_tree']:>11.1%}", flush=True)

    shares = [rows[label]["at_floor"] for label in FLOORS]
    print()
    if min(shares) > 0.30:
        print("  The share stays high as the floor drops: LBR is betting the "
              "minimum whatever the minimum is. The 0.25x figure was an "
              "artifact of the grid, not a considered size.")
    else:
        print("  The share falls as the floor drops, so 0.25x was a genuine "
              "choice rather than a floor effect.")


# ---------------------------------------------------------------------------
# Phase 2 — is raise_cap=1 the real constraint?
# ---------------------------------------------------------------------------

def phase_two(results):
    print("\n\nPHASE 2 — does a second raise per street give the exploiter "
          "something to punish?")
    print(f"{ITERATIONS:,} iterations per cap, {HANDS:,} LBR hands per "
          f"measurement.\n", flush=True)

    rows = results.setdefault("phase_two", {})

    for raise_cap in (1, 2):
        solver = trained_solver(raise_cap)
        strategy = solver.average_strategy()

        for label, sizes in (("on-tree", ON_TREE), ("off-tree", DEFAULT_BET_SIZES)):
            key = f"cap{raise_cap}:{label}"
            if key in rows and rows[key].get("iterations") == solver.iterations:
                continue

            started = time.perf_counter()
            outcome = lbr_value(solver.game, strategy, hands=HANDS,
                                rng=np.random.default_rng(7000 + SEED),
                                rollout_samples=ROLLOUTS, candidates=CANDIDATES,
                                bet_sizes=sizes)
            rows[key] = {
                "raise_cap": raise_cap, "exploiter": label,
                "mean": outcome.mean, "stderr": outcome.stderr,
                "ci95": list(outcome.ci95),
                "proves_exploitable": outcome.proves_exploitable,
                "iterations": solver.iterations,
                "information_sets": len(solver.nodes),
                "seconds": time.perf_counter() - started,
            }
            save_results(results)

    print(f"\n{'':<8}{'exploiter':<12}{'LBR':>20}{'verdict':>24}")
    print("-" * 66)
    for raise_cap in (1, 2):
        for label in ("on-tree", "off-tree"):
            row = rows[f"cap{raise_cap}:{label}"]
            verdict = ("PROVES EXPLOITABLE" if row["proves_exploitable"]
                       else "slack")
            print(f"cap {raise_cap}  {label:<12}{row['mean']:>+11.3f} "
                  f"+/-{row['stderr']:<5.3f}{verdict:>24}")
        gain = (rows[f"cap{raise_cap}:off-tree"]["mean"]
                - rows[f"cap{raise_cap}:on-tree"]["mean"])
        print(f"{'':<8}off-tree is worth {gain:+.3f} chips/hand at cap "
              f"{raise_cap}  ({rows[f'cap{raise_cap}:on-tree']['information_sets']:,}"
              f" information sets)\n")

    cap1 = (rows["cap1:off-tree"]["mean"] - rows["cap1:on-tree"]["mean"])
    cap2 = (rows["cap2:off-tree"]["mean"] - rows["cap2:on-tree"]["mean"])
    if rows["cap2:off-tree"]["proves_exploitable"]:
        print("The bound clears zero at cap 2. raise_cap=1 was the constraint, "
              "and every exploitability measurement in this project has been "
              "made under it.")
    elif cap2 > cap1 + 0.5:
        print("Off-tree betting is worth more at cap 2 than cap 1, so sizing "
              "structure does matter — but not yet enough to clear the bound. "
              "The second barrel is the remaining lever.")
    else:
        print("A second raise per street changes nothing. The action "
              "abstraction is not the constraint at either cap, and the second "
              "barrel is what is left.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    if args.restart:
        for name in os.listdir(HERE):
            if name.startswith("solver_cap") or name == os.path.basename(RESULTS):
                os.remove(os.path.join(HERE, name))
        print("discarded previous state")

    started = time.perf_counter()
    results = load_results()

    phase_one(results)
    phase_two(results)

    save_results(results)
    print(f"\nWrote {RESULTS}")
    print(f"Total {(time.perf_counter() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
