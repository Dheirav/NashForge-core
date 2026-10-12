"""
RL module for PokerBot.

Plug-and-play reinforcement learning algorithms that sit *beside* the existing
evolutionary training (training/) without modifying it.

Public API
----------
    from rl import PPOAgent, PPOTrainer, PPOConfig
    from rl import PokerEnv
    from rl import evaluate_vs_pool, run_tournament
    from rl.agents import RandomOpponent, EvolutionOpponent

Supports any training paradigm — everything is hot-swappable through the
BaseRLAgent interface.
"""

from .base_agent import BaseRLAgent
from .poker_env  import PokerEnv, RewardShaper, AggressionShaper

from .ppo.config  import PPOConfig
from .ppo.agent   import PPOAgent
from .ppo.trainer import PPOTrainer
from .ppo.buffer  import RolloutBuffer
from .ppo.snapshots import SnapshotPool

from .eval.evaluator import evaluate_vs_pool, run_tournament

__all__ = [
    # Base interface
    "BaseRLAgent",
    # Environment
    "PokerEnv",
    "RewardShaper",
    "AggressionShaper",
    # PPO
    "PPOConfig",
    "PPOAgent",
    "PPOTrainer",
    "RolloutBuffer",
    "SnapshotPool",
    # Evaluation
    "evaluate_vs_pool",
    "run_tournament",
]
