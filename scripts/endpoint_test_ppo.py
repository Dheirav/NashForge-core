"""
Did Phase 3's self-play training learn anything?

The curve recorded during training cannot answer it. Its panel score is taken
over 10,000 hands, about ±50 BB/100, which is monitoring — it shows a policy
that has collapsed or a benchmark that has degraded, and nothing finer. Phase
2's per-generation curve was read as rising from +111 to +214 BB/100 while
being statistically indistinguishable from a constant, and that mistake is the
most repeated one in this project's history.

So this compares endpoints directly and precisely: each rung checkpoint against
an untrained policy *from the same initialisation*, both against the same
panel, both at 40,000 hands. That is about ±14 BB/100.

Why the untrained column is not optional
----------------------------------------
Each column on its own is a fact about the opponent as much as about the agent.
Only the difference between them is a fact about training. `PPOTrainer` seeds
torch before building the network precisely so this baseline can be
reconstructed — an untrained network from seed *n* here is bit-for-bit the one
seed *n*'s training started from.

Why the seed spread is the error bar that matters
-------------------------------------------------
Not the ±14. At 200,000 hands these same three seeds scored −68.6, +19.5 and
−3.3 against the CFR agent: 46 BB/100 apart on seed alone, over three times the
40,000-hand bar. A single seed's interval describes the measurement, not the
configuration, and quoting it for a multi-seed claim understates the
uncertainty roughly threefold. Run every seed; report the spread.

Usage
-----
    venv/bin/python scripts/endpoint_test_ppo.py --seed 0 1 2
    venv/bin/python scripts/endpoint_test_ppo.py --seed 0 --rungs 8000000
    venv/bin/python scripts/endpoint_test_ppo.py --seed 0 --hands 2000 --dry-run
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch

# One thread per process, as in training: these matrices are too small to pay
# for thread coordination, and it measured faster — 552 hands/s against 317.
torch.set_num_threads(1)

from rl import PPOConfig, PPOTrainer
from rl.ppo.agent import PPOAgent

# Reused rather than rebuilt: the panel and the adapter that puts a policy
# behind its (game, seat, mask, history) signature are what make this
# comparable to the training run's own monitoring and to Phase 2's endpoint.
from train_ppo import RUNGS, SCRATCH, build_panel, panel_scores

#: The bar Phase 2's endpoint test was taken at, and the one the training
#: instrument can express. 40,000 hands is about ±14 BB/100.
HANDS = 40_000

#: Fixes the deals the panel plays. Distinct from the training seed: it must
#: not vary between the trained and untrained columns, or the difference
#: between them absorbs a change of cards.
EVAL_SEED = 20260820

OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "ppo")


def untrained_agent(train_seed):
    """
    The policy seed `train_seed`'s run began from.

    Built through `PPOTrainer` rather than `PPOAgent` directly, because the
    trainer is what seeds torch before constructing the network. Reaching for
    the agent alone would produce a network from whatever global state happened
    to be current — which is the defect the trainer's seeding comment records.
    """
    config = PPOConfig.heads_up_default()
    config.seed = train_seed
    config.verbose = 0
    return PPOTrainer(config).agent


def rung_path(train_seed, rung, scratch=None):
    return os.path.join(scratch or SCRATCH, f"seed{train_seed}",
                        f"ppo_rung{rung}.pt")


def measure(agent, panel, hands, label):
    started = time.time()
    scores = panel_scores(agent, panel, hands, EVAL_SEED)
    print(f"    {label:<12} {time.time() - started:5.0f}s", flush=True)
    return scores


def format_row(name, untrained, trained):
    """
    One opponent, both columns, and the difference that is the actual result.

    The difference's interval is the two combined in quadrature, not either
    one: it is a comparison of two measurements, and reporting the narrower of
    them would claim a precision neither has.
    """
    u, t = untrained[name], trained[name]
    difference = t["bb_per_100"] - u["bb_per_100"]
    interval = (u["ci95_bb_per_100"] ** 2 + t["ci95_bb_per_100"] ** 2) ** 0.5
    verdict = "improved" if abs(difference) > interval else "no change"
    if difference < 0 and abs(difference) > interval:
        verdict = "worse"
    return {
        "opponent": name,
        "untrained": u["bb_per_100"],
        "untrained_ci95": u["ci95_bb_per_100"],
        "trained": t["bb_per_100"],
        "trained_ci95": t["ci95_bb_per_100"],
        "difference": difference,
        "difference_ci95": interval,
        "verdict": verdict,
        "lookup_miss_rate": t.get("lookup_miss_rate"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required for a measurement, but --merge takes its seeds from the files
    # it is given, so it is enforced below rather than by argparse.
    parser.add_argument("--seed", type=int, nargs="+",
                        help="Training seeds to measure. Report the spread.")
    parser.add_argument("--rungs", type=int, nargs="+", default=list(RUNGS),
                        help="Which rung checkpoints to measure.")
    parser.add_argument("--hands", type=int, default=HANDS,
                        help="Hands per matchup. Below 40,000 is not a result.")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "phase3_endpoint.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write the JSON. For checking the wiring.")
    #: Where the checkpoints are. Seeds trained into their own directory -- as
    #: --out-dir now allows -- are not found by the shared default.
    parser.add_argument("--scratch", default=SCRATCH)
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="Combine per-seed runs into one report. Use when "
                             "the seeds were run as parallel processes.")
    args = parser.parse_args()

    if args.merge:
        return merge(args)

    if not args.seed:
        parser.error("--seed is required: a measurement that does not record "
                     "which seed produced it cannot be reproduced or pooled")

    if args.hands < HANDS and not args.dry_run:
        parser.error(f"--hands below {HANDS:,} is monitoring, not a result; "
                     "pass --dry-run if that is deliberate")

    missing = [rung_path(s, r, args.scratch) for s in args.seed for r in args.rungs
               if not os.path.exists(rung_path(s, r, args.scratch))]
    if missing:
        parser.error("no checkpoint at:\n  " + "\n  ".join(missing))

    panel = build_panel()
    panel_names = [name for name, _ in panel]
    print(f"panel: {', '.join(panel_names)}")
    print(f"{args.hands:,} hands per matchup, duplicate play, seats alternating")
    print(f"eval seed {EVAL_SEED} — the same deals for every column\n", flush=True)

    records = []
    for train_seed in args.seed:
        print(f"seed {train_seed}", flush=True)
        # Measured once per seed and reused across that seed's rungs: it is the
        # same network for all of them, and re-measuring would only add noise
        # to the column every difference is taken against.
        baseline = measure(untrained_agent(train_seed), panel, args.hands,
                           "untrained")

        for rung in args.rungs:
            agent = PPOAgent.from_checkpoint(
                rung_path(train_seed, rung, args.scratch), device="cpu")
            trained = measure(agent, panel, args.hands, f"{rung:,}")
            records.append({
                "seed": train_seed,
                "hands_trained": rung,
                "rows": [format_row(name, baseline, trained)
                         for name in panel_names],
            })
        print(flush=True)

    report(records, panel_names, args)

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({
                "hands_per_matchup": args.hands,
                "eval_seed": EVAL_SEED,
                "seeds": args.seed,
                "rungs": args.rungs,
                "panel": panel_names,
                "records": records,
            }, handle, indent=1)
        print(f"wrote {args.out}")


def merge(args):
    """
    Recombine seeds that were measured as separate processes.

    Running the seeds in parallel is three times faster and loses the spread
    table, which is the part of the report worth reading. This puts it back.

    The widths are checked rather than assumed. Two seeds measured at different
    hand counts, or against different deals, are not columns of one table --
    their difference would carry a change of instrument as well as of seed, and
    nothing downstream would show it.
    """
    loaded = []
    for path in args.merge:
        with open(path) as handle:
            loaded.append((path, json.load(handle)))

    first_path, first = loaded[0]
    for path, data in loaded[1:]:
        for field in ("hands_per_matchup", "eval_seed", "panel", "rungs"):
            if data[field] != first[field]:
                raise SystemExit(
                    f"cannot merge: {field} differs\n"
                    f"  {first_path}: {first[field]}\n"
                    f"  {path}: {data[field]}")

    records = [record for _, data in loaded for record in data["records"]]
    records.sort(key=lambda r: (r["seed"], r["hands_trained"]))
    seeds = sorted({record["seed"] for record in records})
    if len(seeds) != len(records) / len(first["rungs"]):
        raise SystemExit("cannot merge: a seed is missing rungs the others have")

    args.seed, args.rungs, args.hands = seeds, first["rungs"], first["hands_per_matchup"]
    report(records, first["panel"], args)

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({**first, "seeds": seeds, "records": records},
                      handle, indent=1)
        print(f"wrote {args.out}")


def report(records, panel_names, args):
    """
    The tables, and then the spread, which is the part to read.

    Per-seed rows come first because they are what was measured, but a reader
    who stops there will quote one seed's interval for a three-seed result.
    The spread is printed last and against the same opponents so the two are
    hard to confuse.
    """
    print("=" * 74)
    print(f"{'seed':<6}{'trained':>10}  {'opponent':<13}"
          f"{'untrained':>12}{'trained':>12}{'difference':>15}")
    print("-" * 74)
    for record in records:
        for index, row in enumerate(record["rows"]):
            seed = f"{record['seed']}" if index == 0 else ""
            hands = f"{record['hands_trained']:,}" if index == 0 else ""
            print(f"{seed:<6}{hands:>10}  {row['opponent']:<13}"
                  f"{row['untrained']:>+9.1f}   {row['trained']:>+9.1f}   "
                  f"{row['difference']:>+7.1f} ±{row['difference_ci95']:<4.0f}"
                  f" {row['verdict']}")
        print()

    # Printed, not merely stored: this project once published +60.8 BB/100
    # against a CFR agent that was missing 74.3% of its lookups and therefore
    # playing close to at random under the solver's name.
    rates = [row["lookup_miss_rate"] for record in records
             for row in record["rows"]
             if row["opponent"] == "cfr" and row["lookup_miss_rate"] is not None]
    if rates:
        print(f"CFR lookup miss rate {max(rates):.1%} (worst of "
              f"{len(rates)} matchups) — a benchmark that has quietly become "
              "random shows here.\n")

    print("=" * 74)
    print("spread across seeds — the error bar that matters\n")
    for rung in args.rungs:
        print(f"  after {rung:,} hands")
        for name in panel_names:
            values = [row["difference"]
                      for record in records if record["hands_trained"] == rung
                      for row in record["rows"] if row["opponent"] == name]
            if not values:
                continue
            spread = max(values) - min(values)
            each = ", ".join(f"{v:+.1f}" for v in values)
            print(f"    {name:<13}{each:<28} spread {spread:5.1f} BB/100")
        print()

    if len(args.seed) < 3:
        print("Fewer than three seeds: the spread above is not yet an estimate "
              "of anything.")
    if args.hands < HANDS:
        print(f"{args.hands:,} hands is below the {HANDS:,} bar. "
              "This is a wiring check, not a result.")


if __name__ == "__main__":
    main()
