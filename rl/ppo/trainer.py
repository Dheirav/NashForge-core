"""
PPO Trainer.

Implements the full Proximal Policy Optimisation training loop:

  1. Collect n_steps transitions using the current policy.
  2. Compute GAE advantages.
  3. Optimise the clipped surrogate objective for n_epochs mini-batches.
  4. Optionally evaluate against opponent pool and save checkpoint.
  5. Repeat until total_hands reached.

Key design choices
------------------
* One environment for the whole run, with the opponent swapped per hand.
  Rebuilding it re-seeded it, and a re-seeded env deals the same hand every
  time — see PokerEnv.set_opponents().
* Self-play through a snapshot pool: the policy is frozen into the pool every
  `snapshot_every` updates and faces its own past most hands, the live policy
  the rest. See rl/ppo/snapshots.py for why this replaced the hall of fame.
* CSV logging: one row per update cycle with key statistics.
* Evaluation is decoupled — the trainer calls rl.eval.evaluator.evaluate_vs_pool().
"""

from __future__ import annotations

import csv
import os
import time
from typing import List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.ppo.config import PPOConfig
from rl.ppo.agent import PPOAgent
from rl.ppo.buffer import RolloutBuffer
from rl.ppo.snapshots import SnapshotPool
from rl.poker_env import PokerEnv, AggressionShaper


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Full PPO training loop.

    Parameters
    ----------
    config:          PPOConfig controlling all hyperparameters.
    opponent_pool:   Fixed opponents to train against instead of self-play
                     (optional). Given, the agent faces only these — useful
                     for an ablation such as "train against random" and for
                     tests. Omitted, the agent plays itself: its own frozen
                     snapshots most hands and the live policy the rest.
    """

    def __init__(
        self,
        config:         PPOConfig,
        opponent_pool:  Optional[List[Any]] = None,
    ):
        self.cfg    = config
        self.device = torch.device(config.device)
        self._rng   = np.random.default_rng(config.seed)

        # `config.seed` used to seed the environment and nothing else. Three
        # other draws ran off unseeded global state: the network's orthogonal
        # initialisation and every action sample (torch), and the minibatch
        # shuffle (numpy's legacy global). A seeded run was not reproducible,
        # and two runs at the same seed converged to policies that folded 32%
        # and 64% of the time.
        #
        # The endpoint test needs more than reproducibility: it compares the
        # trained policy against an untrained one *from the same
        # initialisation*, which cannot be constructed at all unless the seed
        # reaches the initialiser. Seeded here, before the agent is built.
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # ── Agent ────────────────────────────────────────────────────
        self.agent = PPOAgent(config)

        # ── Opponents ─────────────────────────────────────────────────
        # Self-play unless the caller names opponents. The audit is specific
        # that self-play means current *and past* versions of the policy, so
        # an empty pool early in the run means facing the live policy, not
        # falling back to a random opponent — that fallback is what made the
        # previous "self-play" runs self-play in name only.
        self._fixed_opponents = list(opponent_pool) if opponent_pool else []
        self.snapshots = SnapshotPool(capacity=config.snapshot_pool_size,
                                      rng=self._rng)
        self._faced_current  = 0
        self._faced_snapshot = 0

        # ── Environment factory ───────────────────────────────────────
        shaper = AggressionShaper() if config.use_aggression_shaper else None
        self._env_kwargs = dict(
            num_players    = config.num_players,
            starting_stack = config.starting_stack,
            small_blind    = config.small_blind,
            big_blind      = config.big_blind,
            reward_shaper  = shaper,
            seed           = config.seed,
            reward_scale   = config.effective_reward_scale,
        )
        # One environment for the whole run; see _next_hand().
        self.env = PokerEnv(**self._env_kwargs)

        # ── Rollout buffer ────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            n_steps     = config.n_steps,
            obs_size    = config.obs_size,
            num_actions = config.num_actions,
            gamma       = config.gamma,
            gae_lambda  = config.gae_lambda,
            device      = str(self.device),
            rng         = np.random.default_rng(config.seed),
        )

        # ── Optimiser ─────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=config.lr)

        # ── State trackers ────────────────────────────────────────────
        self.total_hands     = 0
        self.update_cycle    = 0
        self._current_lr     = config.lr

        # ── Logging ───────────────────────────────────────────────────
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._log_path = os.path.join(config.log_dir, "training_log.csv")
        # `pool_size` and `faced_current` are logged because a self-play pool
        # that has quietly stopped filling, or one the agent never actually
        # meets, looks exactly like one that is working. The project has been
        # here before: the CFR benchmark's lookup-miss counter exists for the
        # same reason and caught a benchmark that had become a random opponent.
        self._log_fields = [
            "update", "total_hands", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_fraction",
            "mean_reward", "pool_size", "faced_current", "lr", "elapsed_s",
        ]
        self._start_time = time.time()
        self._init_log()

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> PPOAgent:
        """
        Run the full training loop until total_hands is reached.

        Returns the trained PPOAgent.
        """
        cfg = self.cfg
        # The banner and the closing line are verbose >= 2, because `train()`
        # is also called a segment at a time to walk a budget ladder, and
        # "Training complete" printed at every rung of a run that has ten of
        # them is a small untruth repeated ten times. Progress lines stay at 1.
        if cfg.verbose >= 2:
            print(f"[PPOTrainer] Starting PPO training")
            print(f"  total_hands={cfg.total_hands:,}  n_steps={cfg.n_steps}  n_epochs={cfg.n_epochs}")
            print(f"  network: {cfg.obs_size}→[{cfg.hidden_size}x{cfg.num_layers}]→{cfg.num_actions}")
            if self._fixed_opponents:
                print(f"  device: {self.device}  "
                      f"fixed opponents: {len(self._fixed_opponents)} (no self-play)")
            else:
                print(f"  device: {self.device}  self-play: snapshot every "
                      f"{cfg.snapshot_every} updates, keep {cfg.snapshot_pool_size}, "
                      f"face the live policy {cfg.current_policy_prob:.0%} of hands")

        while self.total_hands < cfg.total_hands:
            rewards = self._collect_rollout()
            stats   = self._ppo_update()
            self.update_cycle += 1

            # Snapshot into the opponent pool. After the update rather than
            # before, so the pool holds policies that existed rather than
            # ones that were about to be replaced.
            if not self._fixed_opponents and self.update_cycle % cfg.snapshot_every == 0:
                self.snapshots.add(self.agent, update_cycle=self.update_cycle)

            # Apply LR decay
            if cfg.lr_decay < 1.0:
                self._current_lr *= cfg.lr_decay
                for g in self.optimizer.param_groups:
                    g["lr"] = self._current_lr

            # Log
            self._write_log(stats, np.mean(rewards))

            if cfg.verbose >= 1 and (self.update_cycle % 10 == 0):
                elapsed = time.time() - self._start_time
                print(
                    f"  [{self.update_cycle:5d}] "
                    f"hands={self.total_hands:,}  "
                    f"πloss={stats['policy_loss']:.4f}  "
                    f"vloss={stats['value_loss']:.4f}  "
                    f"ent={stats['entropy']:.4f}  "
                    f"rew={np.mean(rewards):.3f}  "
                    f"elapsed={elapsed:.0f}s"
                )

            # Evaluate
            if self.update_cycle % cfg.eval_every == 0:
                self._run_eval()

            # Save checkpoint
            if self.update_cycle % cfg.save_every == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(label="final")
        if cfg.verbose >= 2:
            print(f"[PPOTrainer] Training complete. total_hands={self.total_hands:,}")

        return self.agent

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _sample_opponent(self) -> Optional[Any]:
        """
        Who the agent faces this hand.

        Fixed opponents if the caller named any. Otherwise self-play: a frozen
        snapshot with probability `1 - current_policy_prob`, and the live
        policy otherwise. Before the first snapshot exists the pool is empty
        and the live policy is the only past there is.

        Returns the opponent, or None meaning "the env's own default", which
        only happens if fixed opponents were requested and the list is empty.
        """
        if self._fixed_opponents:
            index = int(self._rng.integers(0, len(self._fixed_opponents)))
            return self._fixed_opponents[index]

        if self._rng.random() >= self.cfg.current_policy_prob:
            snapshot = self.snapshots.sample()
            if snapshot is not None:
                self._faced_snapshot += 1
                return snapshot

        self._faced_current += 1
        return self.agent

    def _next_hand(self):
        """
        Sample an opponent and deal the next hand into the *same* environment.

        The env is built once and kept. Building one per hand re-seeded it
        every time, so with `config.seed` set the whole run trained on two
        distinct deals — the audit's own finding, in a second place. Swapping
        the opponent leaves the env's RNG stream to advance as it should.
        """
        opponent = self._sample_opponent()
        self.env.set_opponents([opponent] if opponent is not None else None)
        return self.env.reset()

    def _collect_rollout(self) -> List[float]:
        """
        Collect n_steps transitions into self.buffer.

        Returns list of per-episode total rewards.
        """
        cfg = self.cfg
        self.buffer.reset()
        self.agent.net.eval()

        episode_rewards: List[float] = []
        episode_reward  = 0.0
        hands_in_rollout = 0

        env = self.env
        obs, _ = self._next_hand()

        while not self.buffer.is_full:
            action_mask = self._get_mask_from_obs(env, obs)

            action, log_prob, value = self.agent.act_with_value(obs, action_mask)
            next_obs, reward, terminated, _, info = env.step(action)

            self.buffer.add(
                obs         = obs,
                action      = action,
                reward      = float(reward),
                done        = terminated,
                value       = value,
                log_prob    = log_prob,
                action_mask = action_mask,
            )

            episode_reward += float(reward)
            obs = next_obs

            if terminated:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                hands_in_rollout += 1
                self.total_hands += 1

                if self.total_hands >= cfg.total_hands:
                    break

                obs, _ = self._next_hand()

        # Bootstrap last value
        last_value = self.agent.get_value(obs) if not terminated else 0.0
        self.buffer.compute_returns_and_advantages(last_value)

        return episode_rewards

    @staticmethod
    def _get_mask_from_obs(env: PokerEnv, obs: np.ndarray) -> np.ndarray:
        """Extract current action mask from env (after last step)."""
        from rl.poker_env import get_abstract_action_mask
        game   = env._game
        if game is None or env._is_done():
            return np.ones(6, dtype=np.float32)
        return get_abstract_action_mask(game, env._agent_id)

    # ------------------------------------------------------------------
    # PPO update step
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        """Run n_epochs of PPO on the stored rollout. Returns stats dict."""
        cfg = self.cfg
        self.agent.net.train()

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        total_clip_frac   = 0.0
        n_batches         = 0

        for _ in range(cfg.n_epochs):
            for batch in self.buffer.get_batches(cfg.batch_size):
                obs_b, act_b, old_lp_b, adv_b, ret_b, mask_b = batch

                # Forward pass
                log_probs, values, entropy = self.agent.net.evaluate_actions(
                    obs_b, act_b, mask_b
                )
                values = values.squeeze(-1)

                # Ratio for PPO clip
                ratio       = torch.exp(log_probs - old_lp_b)
                clip_ratio  = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()

                # Value loss (MSE with optional clipping)
                value_loss = F.mse_loss(values, ret_b)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.net.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate stats
                with torch.no_grad():
                    approx_kl  = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac  = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                total_kl          += approx_kl
                total_clip_frac   += clip_frac
                n_batches         += 1

                # Early stop on KL divergence
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

        n = max(n_batches, 1)
        return {
            "policy_loss":   total_policy_loss / n,
            "value_loss":    total_value_loss  / n,
            "entropy":       total_entropy     / n,
            "approx_kl":     total_kl          / n,
            "clip_fraction": total_clip_frac   / n,
        }

    # ------------------------------------------------------------------
    # Evaluation hook
    # ------------------------------------------------------------------

    def _run_eval(self) -> None:
        """
        A floor check during the run, and only that.

        Against a random opponent, so it is at least an opponent from outside
        the lineage. It used to run against the opponent pool, which under
        self-play means the agent's own snapshots — a number that says how the
        policy compares to its own past and nothing about whether it plays
        poker. That is precisely the self-referential measurement the audit
        found surviving 89 runs.

        **The measurement is the panel**, in `evaluation.benchmark`, run by the
        training script. 500 hands here is roughly +/-140 BB/100 and cannot
        resolve anything the training is likely to produce; read it as a
        tripwire for collapse, not as progress.
        """
        from rl.eval.evaluator import evaluate_vs_pool

        self.agent.net.eval()
        opponents = self._fixed_opponents or None
        result    = evaluate_vs_pool(
            agent          = self.agent,
            opponents      = opponents,
            num_hands      = 500,
            cfg            = self.cfg,
        )
        if self.cfg.verbose:
            print(
                f"  [eval @{self.update_cycle}] "
                f"win%={result['win_pct']:.1f}  "
                f"BB/100={result['bb_per_100']:.2f}"
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, label: str = "") -> None:
        tag  = label or f"step{self.update_cycle}"
        path = os.path.join(self.cfg.checkpoint_dir, f"ppo_{tag}.pt")
        self.agent.save(path)
        if self.cfg.verbose >= 2:
            print(f"  [ckpt] saved → {path}")

    # ------------------------------------------------------------------
    # Resuming an interrupted run
    # ------------------------------------------------------------------
    #
    # `_save_checkpoint` writes the policy, which is what the run is *for*.
    # This writes what the run *is*: optimiser moments, hand and update
    # counters, and the opponent pool. Restoring the policy alone restarts
    # Adam from zero moments and an empty pool, which is a different training
    # process wearing the same weights.
    #
    # This box has been terminated twice under memory pressure and a rung of
    # the ladder is four hours, so an interruption has to cost the time since
    # the last checkpoint rather than the run.
    #
    # A resumed run is deterministic from its resume point but is **not**
    # bit-identical to an uninterrupted one: the generators restart from the
    # seed rather than from their state at the moment of the crash. Stated
    # rather than quietly hoped, because `test_a_seeded_run_is_actually_
    # reproducible` promises exact reproducibility and this is the one path
    # where that promise does not hold.

    def save_state(self, path: str) -> None:
        """Write everything needed to continue this run."""
        torch.save({
            "policy":        self.agent.net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "total_hands":   self.total_hands,
            "update_cycle":  self.update_cycle,
            "current_lr":    self._current_lr,
            "faced_current": self._faced_current,
            "faced_snapshot": self._faced_snapshot,
            "snapshots":     [agent.net.state_dict() for agent in self.snapshots],
            "snapshot_taken_at": list(self.snapshots.taken_at),
            "snapshot_total":    self.snapshots.total_taken,
        }, path)

    def load_state(self, path: str) -> None:
        """Continue a run written by `save_state`."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.net.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_hands     = state["total_hands"]
        self.update_cycle    = state["update_cycle"]
        self._current_lr     = state["current_lr"]
        self._faced_current  = state.get("faced_current", 0)
        self._faced_snapshot = state.get("faced_snapshot", 0)

        for group in self.optimizer.param_groups:
            group["lr"] = self._current_lr

        self.snapshots = SnapshotPool(capacity=self.cfg.snapshot_pool_size,
                                      rng=self._rng)
        restored = PPOAgent(self.cfg, device=str(self.device))
        for weights, cycle in zip(state["snapshots"], state["snapshot_taken_at"]):
            restored.net.load_state_dict(weights)
            self.snapshots.add(restored, update_cycle=cycle)
        self.snapshots.total_taken = state["snapshot_total"]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        # Header only when there is nothing there. The constructor runs before
        # a resume does, so truncating here would erase the log of the run
        # being resumed — the history is small, is written continuously, and
        # is the only per-update record the run produces.
        if os.path.exists(self._log_path) and os.path.getsize(self._log_path):
            return
        with open(self._log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._log_fields).writeheader()

    def _write_log(self, stats: dict, mean_reward: float) -> None:
        faced = self._faced_current + self._faced_snapshot
        row = {
            "update":        self.update_cycle,
            "total_hands":   self.total_hands,
            "pool_size":     len(self.snapshots),
            "faced_current": f"{self._faced_current / faced:.3f}" if faced else "",
            "policy_loss":   f"{stats['policy_loss']:.6f}",
            "value_loss":    f"{stats['value_loss']:.6f}",
            "entropy":       f"{stats['entropy']:.6f}",
            "approx_kl":     f"{stats['approx_kl']:.6f}",
            "clip_fraction": f"{stats['clip_fraction']:.4f}",
            "mean_reward":   f"{mean_reward:.4f}",
            "lr":            f"{self._current_lr:.2e}",
            "elapsed_s":     f"{time.time() - self._start_time:.1f}",
        }
        with open(self._log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._log_fields).writerow(row)
