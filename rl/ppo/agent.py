"""
PPO Agent.

Wraps ActorCriticNet and exposes:
  * act()        – sample action (used during rollout collection)
  * get_action() – engine-compatible shim (used in evaluation / tournaments)
  * save() / load()

The agent inherits BaseRLAgent so it is drop-in compatible with any code
that uses the AgentPlayer interface from training/self_play.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from rl.base_agent import BaseRLAgent
from rl.networks.policy_value_net import ActorCriticNet
from rl.ppo.config import PPOConfig


class PPOAgent(BaseRLAgent):
    """
    PPO-trained poker agent.

    Parameters
    ----------
    config:   PPOConfig instance (controls network architecture).
    device:   Torch device string.
    net:      Pre-built ActorCriticNet (optional; created from config if None).
    """

    def __init__(
        self,
        config: PPOConfig,
        device: Optional[str] = None,
        net:    Optional[ActorCriticNet] = None,
    ):
        self.config = config
        self.device = torch.device(device or config.device)

        if net is not None:
            self.net = net.to(self.device)
        else:
            self.net = ActorCriticNet(
                obs_size    = config.obs_size,
                num_actions = config.num_actions,
                hidden_size = config.hidden_size,
                num_layers  = config.num_layers,
            ).to(self.device)

        self.net.eval()   # default to eval mode; PPOTrainer switches to train()

    # ------------------------------------------------------------------
    # BaseRLAgent contract
    # ------------------------------------------------------------------

    def act(
        self,
        obs:         np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Sample an action index given obs and legal-action mask."""
        obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _ = self.net.act(obs_t, mask_t, deterministic=deterministic)

        return int(action.item())

    # ------------------------------------------------------------------
    # Value inference (used by PPO rollout collection)
    # ------------------------------------------------------------------

    def act_with_value(
        self,
        obs:         np.ndarray,
        action_mask: np.ndarray,
    ) -> Tuple[int, float, float]:
        """
        Sample action and return (action, log_prob, value) for buffer storage.
        """
        obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.net.act(obs_t, mask_t)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def get_value(self, obs: np.ndarray) -> float:
        """Return scalar critic estimate V(obs)."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.net.forward(obs_t)
        return float(value.item())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights and config to path (*.pt)."""
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": {
                    "obs_size":    self.config.obs_size,
                    "num_actions": self.config.num_actions,
                    "hidden_size": self.config.hidden_size,
                    "num_layers":  self.config.num_layers,
                    "device":      str(self.device),
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load weights from a .pt file saved by save()."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["state_dict"])

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "PPOAgent":
        """Factory: build PPOAgent from a saved checkpoint."""
        ckpt   = torch.load(path, map_location=device, weights_only=True)
        c      = ckpt["config"]
        config = PPOConfig(
            obs_size    = c["obs_size"],
            num_actions = c["num_actions"],
            hidden_size = c["hidden_size"],
            num_layers  = c["num_layers"],
            device      = device,
        )
        agent = cls(config, device=device)
        agent.net.load_state_dict(ckpt["state_dict"])
        return agent
