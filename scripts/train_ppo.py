"""
Phase 3 — PPO with self-play, heads-up, measured against an independent panel.

Two numbers are recorded and they are not the same thing, for exactly the
reason `scripts/train_evolution.py` gives.

**Self-play reward** is the policy against its own frozen past. It is what the
gradient acts on, and it is uninformative as a measure: the matchup is
symmetric and zero-sum, so it sits at zero whatever the policy has learned. A
rising self-play reward would be a defect rather than progress.

**Panel score** is the policy against opponents sharing no ancestry with it —
random, always-call, and the CFR agent, the last validated against Kuhn's
analytic value of -1/18 and exact Leduc exploitability. This is the curve worth
reading, and even then only with its error bar in hand.

The budget is a ladder
----------------------
Checkpoints at 500k, 2M and 8M hands, so Phase 4 gets a budget axis rather than
a single endpoint. They are rungs of one run, not three runs: the 2M policy is
the 500k policy trained further, which is what makes them comparable.

Read the endpoint tests, not the curve here
-------------------------------------------
The panel score recorded at each checkpoint is over 10,000 hands, which is
about **+/-50 BB/100** — measured, not assumed. That is monitoring: it will
show a policy that has collapsed or a benchmark that has degraded, and it will
show nothing else. It is **not** a result, and it is very close to the
resolution at which Phase 2's per-generation curve was read as rising from
+111 to +214 while being indistinguishable from a constant.
`scripts/endpoint_test.py` at 40,000 hands is where the answer comes from.

And the error bar that matters is not that one. Three training seeds of this
same configuration at 200,000 hands scored -68.6, +19.5 and -3.3 against the
CFR agent: a spread of 46 BB/100 between runs that differ only in seed. A
40,000-hand measurement has a +/-14 bar, so quoting that for a multi-seed
result understates the uncertainty threefold. Run the seeds. Report the spread.

Usage
-----
    python scripts/train_ppo.py --seed 0
    python scripts/train_ppo.py --seed 0 --resume        # after an interruption

Three seeds in parallel fit in about 2.1 GB and finish in the time one takes.
Torch is pinned to one thread per run, which is also faster here — 552 hands/s
against 317, the matrices being too small to pay for thread coordination.
"""
import argparse
import json
import zlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
import torch

torch.set_num_threads(1)

from evaluation import (always_call_agent, benchmark, cfr_agent,
                        checkpoint_every, random_agent)
from rl import PPOConfig, PPOTrainer

#: Rungs of the ladder, in hands.
RUNGS = (500_000, 2_000_000, 8_000_000)

#: Hands per panel matchup at each checkpoint. Monitoring, not measurement.
MONITOR_HANDS = 10_000

#: Never `/tmp`. WSL wipes it on restart, and doing so cost 49 minutes of
#: training on 14 August.
SCRATCH = os.path.expanduser("~/pokerbot-scratch/phase3")

CFR_STRATEGY = os.path.join(os.path.dirname(HERE), "results", "cfr",
                            "nolimit_strategy.pkl")


def ppo_as_benchmark_agent(agent):
    """Adapt a PPOAgent to the panel's (game, seat, mask, history) signature."""
    from engine import get_state_vector

    def act(game, player_id, mask, history):
        obs = np.asarray(get_state_vector(game, player_id), dtype=np.float32)
        return int(agent.act(obs, np.asarray(mask, dtype=np.float32)))
    return act


def build_panel():
    """
    Random, always-call, and the CFR agent if one has been trained.

    The solver is optional only so a run is still possible without one. A run
    without it measures against two baselines any competent agent beats, which
    is a floor check rather than a benchmark.
    """
    # Factories, not agents, and one generator each.
    #
    # These opponents carry state: `random_agent` and `cfr_agent` each close over
    # a Generator, so an agent reused across matchups draws from a stream the
    # previous matchup advanced. Built once and shared -- which is what a caller
    # naturally does -- that makes a column's result depend on every column
    # measured before it, and the panel used to share a single generator between
    # the random opponent and the solver on top of that. Advancing it by one draw
    # moved a 2,000-hand score by 24 BB/100.
    #
    # So each matchup constructs its own opponent from its own seed. The strategy
    # is read once and closed over, because it is the pickle load that is
    # expensive, not the agent.
    panel = [("random", lambda: random_agent(np.random.default_rng(4))),
             ("always-call", lambda: always_call_agent())]
    try:
        import pickle
        with open(CFR_STRATEGY, "rb") as handle:
            saved = pickle.load(handle)
        panel.append(("cfr", lambda: cfr_agent(saved["strategy"],
                                               saved["abstraction"],
                                               np.random.default_rng(5), None)))
    except Exception as error:                              # noqa: BLE001
        print(f"  CFR agent unavailable ({error}); panel is the two baselines",
              flush=True)
    return panel


def panel_scores(agent, panel, hands, seed):
    """
    The policy against each panel opponent, with the uncertainty attached.

    Torch is reseeded per matchup, and that is not cosmetic. The policy samples
    its actions from torch's global generator, which nothing else here reseeds,
    so a score depended on how many hands had already been played in this
    process: measured, the same matchup read -28.9, -29.5 and -3.9 purely on
    what ran before it. Every one of those is a valid sample and none of them is
    reproducible on its own, which is the property that matters when a number is
    going into a report. Seeded from the evaluation seed and the opponent's name
    rather than its position, so reordering the panel cannot move a result.
    """
    played = ppo_as_benchmark_agent(agent)
    out = {}
    for name, make_opponent in panel:
        # A fresh opponent and a fresh torch stream for every matchup, so a
        # number does not depend on what was measured before it.
        torch.manual_seed(seed + zlib.crc32(name.encode()))
        opponent = make_opponent()
        # A fresh counter per matchup, so the miss rate describes this
        # measurement rather than every measurement so far. It is reported
        # because a benchmark that has quietly become a second random opponent
        # is worse than no benchmark — this project published such a number
        # once, at +60.8 BB/100 alongside a 74.3% miss rate.
        misses = [0, 0] if name == "cfr" else None
        result = benchmark(played, opponent, name, hands=hands, seed=seed,
                           misses=misses)
        out[name] = {"bb_per_100": result.bb_per_100,
                     "ci95_bb_per_100": 1.96 * result.stderr / 2 * 100,
                     "separated_from_zero": bool(result.separated_from_zero),
                     "lookup_miss_rate": result.lookup_miss_rate}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True,
                        help="Training seed. Run several; report the spread.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the last checkpoint of this seed.")
    parser.add_argument("--rungs", type=int, nargs="+", default=list(RUNGS))
    parser.add_argument("--monitor-hands", type=int, default=MONITOR_HANDS)
    #: Where checkpoints go. It was hardcoded, and a retrain wrote over the
    #: checkpoints it was meant to be compared against -- the agents behind the
    #: published pre-fix numbers are gone because of it. A run that supersedes
    #: another should have to say so.
    parser.add_argument("--out-dir", default=SCRATCH,
                        help="checkpoint directory; defaults to the shared one, "
                             "so a comparison run should give its own")
    args = parser.parse_args()

    out = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    history_path = os.path.join(out, "history.json")
    state_path = os.path.join(out, "state_latest.pt")

    config = PPOConfig.heads_up_default()
    config.seed = args.seed
    config.total_hands = args.rungs[-1]
    config.eval_every = 10 ** 9        # the panel is the evaluation, below
    config.save_every = 10 ** 9        # checkpointing is driven here
    config.checkpoint_dir = out
    config.log_dir = out
    config.verbose = 1

    history = []
    if os.path.exists(history_path):
        with open(history_path) as handle:
            history = json.load(handle)

    trainer = PPOTrainer(config)
    if args.resume and os.path.exists(state_path):
        trainer.load_state(state_path)
        print(f"resumed at {trainer.total_hands:,} hands, update "
              f"{trainer.update_cycle}\n", flush=True)
    elif args.resume:
        print(f"nothing to resume at {state_path}; starting from zero\n",
              flush=True)

    panel = build_panel()

    # Ten checkpoints whatever the length, per the project's convention, plus
    # the rungs themselves. An interruption then costs a proportion of the run
    # rather than a count that means something different in every experiment.
    step = checkpoint_every(args.rungs[-1])
    points = sorted(set(list(range(step, args.rungs[-1] + 1, step))
                        + list(args.rungs)))

    print(f"seed {args.seed}, ladder {', '.join(f'{r:,}' for r in args.rungs)} "
          f"hands, ent_coef {config.ent_coef}", flush=True)
    print(f"self-play reward is symmetric and means nothing; the panel is the "
          f"measure\n", flush=True)

    started = time.perf_counter()
    for point in points:
        if trainer.total_hands >= point:
            continue                                # already past it, resuming

        trainer.cfg.total_hands = point
        clock = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - clock

        is_rung = point in args.rungs
        scores = panel_scores(trainer.agent, panel, args.monitor_hands,
                              seed=17 + args.seed)

        faced = trainer._faced_current + trainer._faced_snapshot
        row = {
            "hands": trainer.total_hands,
            "updates": trainer.update_cycle,
            "seconds": elapsed,
            "rung": is_rung,
            "pool_size": len(trainer.snapshots),
            "faced_current": trainer._faced_current / faced if faced else None,
            "panel": scores,
        }
        history.append(row)

        # The history file is small and *is* the run's result, so it is written
        # every checkpoint. Atomically, because a crash during the write of a
        # results file is the one way to lose a run twice.
        temporary = history_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(history, handle, indent=2)
        os.replace(temporary, history_path)

        trainer.save_state(state_path)
        if is_rung:
            trainer.agent.save(os.path.join(out, f"ppo_rung{point}.pt"))

        versus = "  ".join(
            f"{name} {value['bb_per_100']:+.0f}+/-{value['ci95_bb_per_100']:.0f}"
            for name, value in scores.items())
        print(f"  {trainer.total_hands:>9,} hands  {elapsed:>6.0f}s  "
              f"pool {row['pool_size']}  live {row['faced_current']:.0%}  "
              f"{versus}{'   [rung]' if is_rung else ''}", flush=True)

    total = time.perf_counter() - started
    print(f"\nseed {args.seed} complete in {total / 3600:.1f}h. Wrote "
          f"{history_path}")
    print("The curve above is monitoring. Run scripts/endpoint_test.py against "
          "the rung\ncheckpoints for the result, and report the spread across "
          "seeds rather than the\nerror bar of any single one.")


if __name__ == "__main__":
    main()
