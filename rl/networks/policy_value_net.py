"""
Actor-Critic network for PPO (PyTorch).

Architecture
------------
Shared trunk:  Linear(obs_size, hidden) → LayerNorm → ReLU
               Linear(hidden, hidden)   → LayerNorm → ReLU

Actor head:    Linear(hidden, num_actions)   (logits, masked before softmax)
Critic head:   Linear(hidden, 1)            (state-value estimate)

Design notes
------------
* Separate actor / critic heads on a shared trunk gives good gradient flow
  while keeping parameter count small.
* LayerNorm (not BatchNorm) so single-sample inference works without issues.
* Orthogonal weight init (standard for PPO).
* The network is decoupled from the engine – it only knows obs_size and
  num_actions, so it can be used with any feature set.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _init_weights(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal init (recommended for actor-critic / PPO)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0)


class ActorCriticNet(nn.Module):
    """
    Shared-trunk actor-critic network.

    Parameters
    ----------
    obs_size:    Length of the observation vector.  Default: 19 (engine features).
    num_actions: Size of discrete action space.     Default: 6 (abstract actions).
    hidden_size: Width of each hidden layer.        Default: 128.
    num_layers:  Depth of shared trunk.             Default: 2.
    """

    def __init__(
        self,
        obs_size:    int = 19,
        num_actions: int = 6,
        hidden_size: int = 128,
        num_layers:  int = 2,
    ):
        super().__init__()

        self.obs_size    = obs_size
        self.num_actions = num_actions

        # ── Shared trunk ──────────────────────────────────────────────
        layers = []
        in_dim = obs_size
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
            ]
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)

        # ── Actor head ────────────────────────────────────────────────
        self.actor_head = nn.Linear(hidden_size, num_actions)

        # ── Critic head ───────────────────────────────────────────────
        self.critic_head = nn.Linear(hidden_size, 1)

        # ── Weight initialisation ─────────────────────────────────────
        self.trunk.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))
        _init_weights(self.actor_head,  gain=0.01)   # small gain for actor
        _init_weights(self.critic_head, gain=1.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits and state value.

        Args:
            obs:         Float tensor (..., obs_size).
            action_mask: Bool/float tensor (..., num_actions).
                         1 = legal, 0 = illegal.  None means all legal.

        Returns:
            logits: Raw logits (..., num_actions).  Illegal actions are
                    set to -1e8 so they get ≈0 probability.
            value:  State value estimate (..., 1).
        """
        features = self.trunk(obs)
        logits   = self.actor_head(features)
        value    = self.critic_head(features)

        if action_mask is not None:
            # Where mask == 0, set logits to large negative
            illegal = (action_mask == 0)
            logits  = logits.masked_fill(illegal, -1e8)

        return logits, value

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_distribution(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """Return a Categorical distribution over legal actions."""
        logits, _ = self.forward(obs, action_mask)
        return Categorical(logits=logits)

    def act(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or pick greedily) an action.

        Returns
        -------
        action:   int tensor (...)
        log_prob: float tensor (...)
        value:    float tensor (..., 1)
        """
        logits, value = self.forward(obs, action_mask)
        dist          = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored actions for PPO update.

        Returns
        -------
        log_probs: Log probability of each stored action.
        values:    Critic estimates.
        entropy:   Distribution entropy (for entropy bonus).
        """
        logits, values = self.forward(obs, action_mask)
        dist           = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network state dict to path (e.g. 'model.pt')."""
        torch.save(
            {
                "state_dict":  self.state_dict(),
                "obs_size":    self.obs_size,
                "num_actions": self.num_actions,
                "hidden_size": self.trunk[0].out_features,  # first Linear out_features
                "num_layers":  sum(1 for m in self.trunk if isinstance(m, nn.Linear)),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ActorCriticNet":
        """Load a saved ActorCriticNet from path."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net  = cls(
            obs_size=ckpt["obs_size"],
            num_actions=ckpt["num_actions"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.to(device)
        return net
