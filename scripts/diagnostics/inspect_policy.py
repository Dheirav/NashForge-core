#!/usr/bin/env python3
"""Inspect a saved PPO checkpoint: params, eval vs baselines, sample action traces.

Usage:
  python scripts/diagnostics/inspect_policy.py --checkpoint checkpoints/ppo_hu_v1/ppo_final.pt
"""
from __future__ import annotations
import argparse
import numpy as np

from rl.ppo.agent import PPOAgent
from rl.eval.evaluator import evaluate_vs_pool
from rl.agents import RandomOpponent, CallOpponent
from engine import PokerGame
from training.fitness import abstract_action_to_engine_action


def param_stats(agent: PPOAgent) -> None:
    print("\nModel parameter statistics:")
    total = 0
    for name, p in agent.net.named_parameters():
        arr = p.detach().cpu().numpy().ravel()
        total += arr.size
        print(f"  {name:40s}  mean={arr.mean():+.6f}  std={arr.std():.6f}  min={arr.min():+.6f}  max={arr.max():+.6f}  zeros={(arr==0).sum()}/{arr.size}")
    print(f"  Total params: {total}\n")


def sample_action_traces(agent: PPOAgent, n_hands: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    action_counts = np.zeros(agent.config.num_actions, dtype=int)

    for h in range(n_hands):
        stacks = [agent.config.starting_stack] * agent.config.num_players
        game = PokerGame(
            player_stacks=stacks,
            small_blind=agent.config.small_blind,
            big_blind=agent.config.big_blind,
            seed=int(rng.integers(0, 2**31)),
            enable_history=False,
        )

        print(f"\n--- Hand {h+1} ---")
        step = 0
        while game.state.betting_round != "showdown":
            pid = game.state.current_player
            if pid is None:
                break
            try:
                aidx = agent.get_action(game, pid)
            except Exception as e:
                print(f"  get_action error: {e}; defaulting to check/call")
                aidx = 1

            action_counts[aidx] += 1
            eng_act = abstract_action_to_engine_action(aidx, game, pid)
            print(f"  step={step:03d}  pid={pid}  abstract={aidx}  engine={eng_act}")
            game.apply_action(pid, eng_act)
            step += 1
            if step > 500:
                print("  truncated (too many steps)")
                break

        if game.state.betting_round == "showdown":
            from engine import resolve_showdown
            resolve_showdown(game.players, game.state.community_cards, game.state.pot, game.state.button)

        final = [p.stack for p in game.players]
        print(f"  final stacks: {final}")

    print("\nAction counts (abstract indices):")
    for i, c in enumerate(action_counts):
        print(f"  {i}: {c}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-hands", type=int, default=200)
    p.add_argument("--sample-hands", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = PPOAgent.from_checkpoint(args.checkpoint, device="cpu")

    param_stats(agent)

    print("\nEvaluating vs RandomOpponent...")
    res_rand = evaluate_vs_pool(agent, opponents=[RandomOpponent()], num_hands=args.eval_hands, seed=args.seed, cfg=agent.config)
    print(f"  Random -> win%={res_rand['win_pct']:.2f}  BB/100={res_rand['bb_per_100']:+.2f}")

    print("\nEvaluating vs CallOpponent (passive)...")
    res_call = evaluate_vs_pool(agent, opponents=[CallOpponent()], num_hands=args.eval_hands, seed=args.seed+1, cfg=agent.config)
    print(f"  Call -> win%={res_call['win_pct']:.2f}  BB/100={res_call['bb_per_100']:+.2f}")

    print("\nSampling action traces from a few hands:")
    sample_action_traces(agent, n_hands=args.sample_hands, seed=args.seed)


if __name__ == "__main__":
    main()
