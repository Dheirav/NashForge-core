"""
Lightweight opponent wrappers for use inside PokerEnv.

These implement the minimal .get_action(game, player_id) -> int interface
so they can be passed as opponents to PokerEnv without any training setup.
"""
from __future__ import annotations

import random
import numpy as np


class RandomOpponent:
    """Uniformly samples a legal abstract action."""

    def get_action(self, game, player_id: int) -> int:
        from engine import get_abstract_action_mask
        mask = get_abstract_action_mask(game, player_id)
        legal = [i for i, v in enumerate(mask) if v]
        return random.choice(legal) if legal else 1


class CallOpponent:
    """Always check/call (passive baseline)."""

    def get_action(self, game, player_id: int) -> int:
        return 1


class RaiseOpponent:
    """Always raise min (aggressive baseline)."""

    def get_action(self, game, player_id: int) -> int:
        from engine import get_abstract_action_mask
        mask = get_abstract_action_mask(game, player_id)
        # Prefer half-pot raise (2), then pot (3), then check/call
        for a in (2, 3, 1):
            if mask[a]:
                return a
        return 1


class EvolutionOpponent:
    """
    Wraps a trained evolution AgentPlayer so it can be used as an opponent
    inside PokerEnv.

    Parameters
    ----------
    agent_player: training.self_play.AgentPlayer instance
    """

    def __init__(self, agent_player):
        self._agent = agent_player

    def get_action(self, game, player_id: int) -> int:
        return self._agent.get_action(game, player_id)
