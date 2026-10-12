"""
Rollout buffer for PPO.

Stores a fixed-length trajectory of (obs, action, reward, done, value, log_prob,
action_mask) tuples and computes Generalised Advantage Estimates (GAE).

Design
------
* Pre-allocated NumPy arrays (no Python lists) for speed.
* Supports full-episode (poker hand) trajectories: because a hand is typically
  3-15 agent steps, episodes are short — the whole return fits cleanly in GAE.
* Converts to PyTorch tensors only at sample time (keep NumPy as long as possible).
"""

from __future__ import annotations

from typing import Generator, Optional, Tuple
import numpy as np
import torch


class RolloutBuffer:
    """
    Fixed-capacity circular rollout buffer.

    Parameters
    ----------
    n_steps:    Maximum number of steps to store before an update.
    obs_size:   Observation vector length.
    num_actions: Number of discrete actions (for action_mask storage).
    gamma:      Discount factor.
    gae_lambda: GAE λ.
    device:     Torch device for tensor output.
    """

    def __init__(
        self,
        n_steps:    int,
        obs_size:   int   = 19,
        num_actions: int  = 6,
        gamma:      float = 0.999,
        gae_lambda: float = 0.95,
        device:     str   = "cpu",
        rng:        Optional[np.random.Generator] = None,
    ):
        # Own generator rather than numpy's global one: the minibatch shuffle
        # was the third unseeded draw in a run advertised as seeded.
        self._rng        = rng if rng is not None else np.random.default_rng()
        self.n_steps     = n_steps
        self.obs_size    = obs_size
        self.num_actions = num_actions
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.device      = device

        self._pos     = 0
        self._full    = False

        # Pre-allocate storage
        self.obs          = np.zeros((n_steps, obs_size),    dtype=np.float32)
        self.actions      = np.zeros( n_steps,               dtype=np.int64)
        self.rewards      = np.zeros( n_steps,               dtype=np.float32)
        self.dones        = np.zeros( n_steps,               dtype=np.float32)
        self.values       = np.zeros( n_steps,               dtype=np.float32)
        self.log_probs    = np.zeros( n_steps,               dtype=np.float32)
        self.action_masks = np.ones( (n_steps, num_actions), dtype=np.float32)

        # Computed after rollout ends
        self.advantages   = np.zeros(n_steps, dtype=np.float32)
        self.returns      = np.zeros(n_steps, dtype=np.float32)

    # ------------------------------------------------------------------
    # Filling the buffer
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear buffer for a new rollout."""
        self._pos  = 0
        self._full = False

    def add(
        self,
        obs:         np.ndarray,
        action:      int,
        reward:      float,
        done:        bool,
        value:       float,
        log_prob:    float,
        action_mask: np.ndarray,
    ) -> None:
        """Store one (obs, action, reward, done, value, log_prob, mask) tuple."""
        if self._pos >= self.n_steps:
            raise RuntimeError("Buffer is full; call reset() first.")

        i = self._pos
        self.obs[i]          = obs
        self.actions[i]      = action
        self.rewards[i]      = reward
        self.dones[i]        = float(done)
        self.values[i]       = value
        self.log_probs[i]    = log_prob
        self.action_masks[i] = action_mask
        self._pos           += 1

    @property
    def is_full(self) -> bool:
        return self._pos >= self.n_steps

    @property
    def size(self) -> int:
        return self._pos

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Call this once after the rollout is complete, passing the critic
        estimate of the state after the last step (0 if last step was terminal).

        Args:
            last_value: V(s_{T+1}) — 0 if final step was terminal.
        """
        n = self._pos
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = self.values[t + 1]

            delta  = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Normalise advantages (zero mean, unit std)
        adv = self.advantages[:n]
        self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------------------------------------------------------------------
    # Mini-batch sampling
    # ------------------------------------------------------------------

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Yield shuffled mini-batches as Torch tensors.

        Yields (obs, actions, log_probs_old, advantages, returns, action_masks)
        """
        n       = self._pos
        indices = np.arange(n)
        self._rng.shuffle(indices)

        dev = self.device
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield (
                torch.from_numpy(self.obs[batch_idx]).to(dev),
                torch.from_numpy(self.actions[batch_idx]).to(dev),
                torch.from_numpy(self.log_probs[batch_idx]).to(dev),
                torch.from_numpy(self.advantages[batch_idx]).to(dev),
                torch.from_numpy(self.returns[batch_idx]).to(dev),
                torch.from_numpy(self.action_masks[batch_idx]).to(dev),
            )
