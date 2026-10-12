"""
Abstract base class for RL agents.

All RL agents (PPO, DQN, etc.) implement this interface so they can be
used interchangeably in:
  - The poker_env rollout loop
  - The tournament evaluator
  - Forward-compatibility with any future algorithm
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseRLAgent(ABC):
    """
    Minimal contract every RL agent must satisfy.

    The interface is deliberately thin:
      * act()   – pick an abstract action index given an observation + mask
      * get_action() – engine-compatible shim used by the evaluator / tournament

    Concrete subclasses (PPOAgent, …) add their own training logic.
    """

    # Number of abstract actions (fold, check/call, raise×3, all-in)
    NUM_ACTIONS: int = 6
    # Feature vector dimensionality (must match engine.get_state_vector)
    OBS_SIZE: int = 19

    @abstractmethod
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select an abstract action index.

        Args:
            obs:          Feature vector of shape (OBS_SIZE,).
            action_mask:  Boolean/float array of shape (NUM_ACTIONS,).
                          1 = legal, 0 = illegal.
            deterministic: If True, pick argmax (greedy); else sample.

        Returns:
            action_idx in [0, NUM_ACTIONS).
        """

    # ------------------------------------------------------------------
    # Engine-compatible shim
    # ------------------------------------------------------------------

    def get_action(self, game, player_id: int) -> int:
        """
        Drop-in replacement for AgentPlayer.get_action().

        Compatible with:
            fitness.py evaluation harness
            tournament / evaluator code
            Any game loop that calls agent.get_action(game, player_id)

        Args:
            game:       live PokerGame instance
            player_id:  seat index of this agent

        Returns:
            Abstract action index (0-5).
        """
        from engine import get_state_vector  # lazy import
        from rl.poker_env import get_abstract_action_mask

        obs = np.array(get_state_vector(game, player_id), dtype=np.float32)
        mask = get_abstract_action_mask(game, player_id)
        return self.act(obs, mask, deterministic=False)

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (subclasses may override)
    # ------------------------------------------------------------------

    def on_episode_end(self, total_reward: float, info: dict) -> None:
        """Called at end of each episode; useful for stat tracking."""

    def save(self, path: str) -> None:
        """Persist agent weights to disk."""
        raise NotImplementedError(f"{self.__class__.__name__}.save() not implemented")

    def load(self, path: str) -> None:
        """Load agent weights from disk."""
        raise NotImplementedError(f"{self.__class__.__name__}.load() not implemented")
