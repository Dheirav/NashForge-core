#!/usr/bin/env python3
"""
PPO Training Launcher.

Usage
-----
# Heads-up with default config:
python run_ppo_training.py --mode hu

# 6-max with default config:
python run_ppo_training.py --mode mt

# Full custom config:
python run_ppo_training.py \\
    --mode hu \\
    --total-hands 1000000 \\
    --hidden-size 256 \\
    --num-layers 3 \\
    --lr 1e-4 \\
    --n-steps 1024 \\
    --snapshot-every 10 \\
    --snapshot-pool-size 8 \\
    --checkpoint-dir checkpoints/ppo_hu_v1 \\
    --device cpu \\
    --seed 42

# Floor-check an existing checkpoint against a random opponent:
python run_ppo_training.py --eval-only --checkpoint checkpoints/ppo/ppo_final.pt

# Dry-run (print config and exit):
python run_ppo_training.py --mode hu --dry-run
"""

import argparse
import json
import os
import sys

# Ensure repo root is on path. This pointed at `scripts/` itself, left over
# from when the launcher sat at the top level, so importing `rl` failed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl import PPOConfig, PPOTrainer, PPOAgent, run_tournament
from rl.agents import RandomOpponent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train / evaluate a PPO poker agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", choices=["hu", "mt"], default="hu",
                   help="hu = heads-up (2 players), mt = multi-table (6 players)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; run evaluation only.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without training.")

    # Network
    p.add_argument("--hidden-size", type=int,  default=None, help="Hidden layer width")
    p.add_argument("--num-layers",  type=int,  default=None, help="Number of trunk layers")

    # Environment
    # None so the mode default wins unless the flag is actually given. These
    # carried 1000/5/10, which silently overrode heads_up_default()'s 200/1/2
    # — the table `evaluation.benchmark` measures on — on every single run.
    p.add_argument("--starting-stack", type=int, default=None)
    p.add_argument("--small-blind",    type=int, default=None)
    p.add_argument("--big-blind",      type=int, default=None)
    p.add_argument("--use-aggression-shaper", action="store_true",
                   help="Add dense aggression reward signal")

    # Rollout
    p.add_argument("--n-steps",  type=int, default=None, help="Steps per rollout")
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)

    # PPO
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--lr-decay",    type=float, default=1.0)
    p.add_argument("--clip-range",  type=float, default=0.2)
    p.add_argument("--ent-coef",    type=float, default=None)
    p.add_argument("--gamma",       type=float, default=0.999)
    p.add_argument("--gae-lambda",  type=float, default=0.95)
    p.add_argument("--target-kl",   type=float, default=None)

    # Training schedule
    p.add_argument("--total-hands", type=int, default=None)
    p.add_argument("--eval-every",  type=int, default=20)
    p.add_argument("--save-every",  type=int, default=50)

    # Self-play opponents
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="Update cycles between policy snapshots into the pool")
    p.add_argument("--snapshot-pool-size", type=int, default=None,
                   help="How many past snapshots to keep")
    p.add_argument("--current-policy-prob", type=float, default=None,
                   help="Probability of facing the live policy, not a snapshot")

    # IO
    p.add_argument("--checkpoint-dir",  type=str,  default=None)
    p.add_argument("--checkpoint",      type=str,  default=None,
                   help="Path to .pt file (for --eval-only or resume)")
    p.add_argument("--log-dir",         type=str,  default=None)

    # Misc
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    return p


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> PPOConfig:
    """Build PPOConfig from parsed CLI args, falling back to mode defaults."""
    if args.mode == "hu":
        cfg = PPOConfig.heads_up_default()
    else:
        cfg = PPOConfig.multitable_default()

    # Override with explicit CLI args
    overrides = {
        "hidden_size":          args.hidden_size,
        "num_layers":           args.num_layers,
        "starting_stack":       args.starting_stack,
        "small_blind":          args.small_blind,
        "big_blind":            args.big_blind,
        "use_aggression_shaper": args.use_aggression_shaper or None,
        "n_steps":              args.n_steps,
        "n_epochs":             args.n_epochs,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        "lr_decay":             args.lr_decay,
        "clip_range":           args.clip_range,
        "ent_coef":             args.ent_coef,
        "gamma":                args.gamma,
        "gae_lambda":           args.gae_lambda,
        "target_kl":            args.target_kl,
        "total_hands":          args.total_hands,
        "eval_every":           args.eval_every,
        "save_every":           args.save_every,
        "snapshot_every":       args.snapshot_every,
        "snapshot_pool_size":   args.snapshot_pool_size,
        "current_policy_prob":  args.current_policy_prob,
        "device":               args.device,
        "seed":                 args.seed,
        "verbose":              args.verbose,
    }

    # Build name for checkpoint dir and log dir based on mode
    tag = f"ppo_{args.mode}"
    if args.seed is not None:
        tag += f"_s{args.seed}"

    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    else:
        overrides["checkpoint_dir"] = f"checkpoints/{tag}"

    if args.log_dir:
        overrides["log_dir"] = args.log_dir
    else:
        overrides["log_dir"] = f"logs/{tag}"

    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    return cfg


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    cfg = build_config(args)

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print("── PPO Config ────────────────────────────────────────")
        cfg_dict = {k: v for k, v in vars(cfg).items()}
        for k, v in cfg_dict.items():
            print(f"  {k:30s} = {v}")
        print("\n[dry-run] Exiting without training.")
        return

    # ── Eval-only ─────────────────────────────────────────────────────
    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint <path.pt>")

        print(f"[eval] Loading checkpoint: {args.checkpoint}")
        agent = PPOAgent.from_checkpoint(args.checkpoint, device=args.device)

        # A floor check only. The measurement is scripts/endpoint_test.py,
        # which plays the panel — random, always-call and the CFR agent — at a
        # hand count that can resolve the difference.
        opponents = [RandomOpponent()]

        from rl.eval.evaluator import evaluate_vs_pool
        result = evaluate_vs_pool(
            agent          = agent,
            opponents      = opponents,
            num_hands      = 2_000,
            cfg            = cfg,
        )
        print(f"\n── Evaluation Results ───────────────────────────────")
        print(f"  win%    : {result['win_pct']:.2f}%")
        print(f"  BB/100  : {result['bb_per_100']:+.2f}")
        print(f"  hands   : {result['total_hands']:,}")
        return

    # ── Training ──────────────────────────────────────────────────────
    print(f"[train] Mode={args.mode}  total_hands={cfg.total_hands:,}")
    print(f"        checkpoint_dir={cfg.checkpoint_dir}")
    print(f"        log_dir={cfg.log_dir}")
    print(f"        self-play: snapshot every {cfg.snapshot_every} updates, "
          f"keep {cfg.snapshot_pool_size}, live policy {cfg.current_policy_prob:.0%}")

    # Save config alongside checkpoint
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.checkpoint_dir, "config.json"))

    trainer = PPOTrainer(cfg)
    trained_agent = trainer.train()

    print("\n── Training complete ─────────────────────────────────────")
    print(f"  Checkpoint dir : {cfg.checkpoint_dir}")
    print(f"  Log            : {cfg.log_dir}/training_log.csv")
    print(
        "\nTo evaluate:\n"
        f"  python run_ppo_training.py --eval-only "
        f"--checkpoint {cfg.checkpoint_dir}/ppo_final.pt"
    )


if __name__ == "__main__":
    main()
