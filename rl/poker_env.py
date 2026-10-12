"""
Gym-style poker environment wrapping the PokerGame engine.

Design goals
------------
* Zero dependency on `gym` or `gymnasium` – this project only needs NumPy
  and PyTorch, so we ship a minimal Env interface here.
* Engine-agnostic reward shaping via a pluggable RewardShaper.
* Opponent pool that can hold any mix of:
    - EvolutionAgent (NumPy network from training/)
    - RandomAgent / HeuristicAgent (from agents/)
    - PPOAgent (from rl/)
    – i.e. anything with .get_action(game, player_id) -> int

Episode = one poker hand (reset_hand → apply actions until showdown).

Observation
-----------
19-dimensional float32 feature vector produced by engine.get_state_vector().

Action space
------------
Discrete(6):  0=fold  1=check/call  2=raise½pot  3=raise1pot  4=raise2x  5=all-in

Reward
------
Default: chip_delta / big_blind at end of hand (BB units).
Shaped reward (optional) adds immediate signals (see RewardShaper below).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from engine import PokerGame, get_state_vector, get_abstract_action_mask
from training.fitness import abstract_action_to_engine_action, finish_hand


# ---------------------------------------------------------------------------
# Space stubs (no gym dependency)
# ---------------------------------------------------------------------------

class DiscreteSpace:
    """Minimal stand-in for gym.spaces.Discrete."""
    def __init__(self, n: int):
        self.n = n
        self.shape = (n,)

    def sample(self) -> int:
        return random.randrange(self.n)


class BoxSpace:
    """Minimal stand-in for gym.spaces.Box."""
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low  = np.full(shape, low,  dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype


# ---------------------------------------------------------------------------
# Reward shaper (pluggable)
# ---------------------------------------------------------------------------

class RewardShaper:
    """
    Compute immediate (per-step) reward signals to augment the sparse
    terminal chip-delta reward.

    Override shape_step() to add custom signals without touching env code.
    Final reward = terminal_chips_reward + sum(shaping_bonuses_over_hand).
    """

    def shape_step(
        self,
        game: PokerGame,
        player_id: int,
        action_idx: int,
        prev_stack: int,
    ) -> float:
        """
        Called *after* the action is applied to the game.

        Default implementation: zero (pure terminal reward).
        """
        return 0.0


class AggressionShaper(RewardShaper):
    """
    Small bonus for +EV aggression: adds a tiny positive reward when the
    agent raises into a situation where the pot odds favour it, and a tiny
    penalty for calling with bad pot odds.

    This is intentionally small so it doesn't override the final chip signal.
    """

    RAISE_BONUS   =  0.05   # BB units
    BAD_CALL_PEN  = -0.03   # BB units

    def shape_step(
        self,
        game: PokerGame,
        player_id: int,
        action_idx: int,
        prev_stack: int,
    ) -> float:
        bb = game.state.big_blind
        player = game.players[player_id]
        to_call = game.current_bet - player.bet
        pot      = game.state.pot.total or 1

        pot_odds = to_call / (pot + to_call + 1e-8)

        if action_idx in (2, 3, 4, 5):  # raise / all-in
            return self.RAISE_BONUS / bb
        if action_idx == 1 and to_call > 0:  # call with bad pot odds
            if pot_odds > 0.5:
                return self.BAD_CALL_PEN / bb
        return 0.0


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class PokerEnv:
    """
    Single-agent poker environment (agent = seat 0).

    Opponents occupy the remaining seats and are sampled from
    `opponent_pool` at the start of each episode.

    Parameters
    ----------
    num_players:      Table size (2-6). Default: 2 (heads-up).
    starting_stack:   Chips each player starts with. Default: 1 000.
    small_blind:      SB amount. Default: 5.
    big_blind:        BB amount. Default: 10.
    opponent_pool:    List of agents with .get_action(game, pid). If None,
                      random agents are used.
    reward_shaper:    Optional RewardShaper for dense rewards.
    seed:             Random seed for reproducibility.
    """

    OBS_SIZE   = 19
    NUM_ACTIONS = 6

    def __init__(
        self,
        num_players:    int                     = 2,
        starting_stack: int                     = 1_000,
        small_blind:    int                     = 5,
        big_blind:      int                     = 10,
        opponent_pool:  Optional[List[Any]]     = None,
        reward_shaper:  Optional[RewardShaper]  = None,
        seed:           Optional[int]           = None,
        reward_scale:   float                   = 1.0,
    ):
        assert 2 <= num_players <= 6, "num_players must be 2-6"
        self.num_players    = num_players
        self.starting_stack = starting_stack
        self.small_blind    = small_blind
        self.big_blind      = big_blind
        self.reward_shaper  = reward_shaper or RewardShaper()

        # Reward is chip delta in big blinds, times this. 1.0 leaves it in BB.
        #
        # It exists because BB is the right unit to *read* and the wrong one to
        # *learn from* here. A 100 BB stack means a hand swings up to +/-100,
        # so returns arrive with a standard deviation near 60, the critic's MSE
        # is in the thousands, and on a shared trunk its gradient measured
        # 55.2 against the policy's 0.0078 — a factor of 7,000. `max_grad_norm`
        # then scales the sum down by 0.006 and the policy stops moving.
        #
        # The failure looks like a plateau, and `docs/training-plan.md` predicts
        # a plateau in advance. A predicted result arrived at by an artefact is
        # the worst kind this project can produce, so the scale is explicit and
        # `PPOConfig` derives it from the table rather than leaving it at 1.0.
        #
        # Nothing measured changes: `chip_delta` and `bb_per_100` in `info` stay
        # in chips, and BB/100 in the reports comes from `evaluation.benchmark`,
        # which never sees this.
        self.reward_scale   = float(reward_scale)

        # Opponent pool – any object with .get_action(game, pid) -> int
        if not opponent_pool:
            from rl.agents.random_opponent import RandomOpponent
            self.opponent_pool = [RandomOpponent() for _ in range(num_players - 1)]
        else:
            self.opponent_pool = list(opponent_pool)

        self._default_opponents = list(self.opponent_pool)
        self._rng  = np.random.default_rng(seed)
        self._game: Optional[PokerGame] = None
        # The agent's seat is drawn per hand by reset(); see the note there.
        self._agent_id: int = 0
        self._seat_to_opponent: Dict[int, Any] = {}

        # Spaces
        self.observation_space = BoxSpace(0.0, 1.0, (self.OBS_SIZE,))
        self.action_space      = DiscreteSpace(self.NUM_ACTIONS)

        # Per-episode stats
        self._episode_shaped_rewards: float = 0.0
        self._episode_actions: int = 0

        # Hands re-dealt because the agent never had a decision in them, and
        # the chips those hands were worth. See reset().
        self.hands_without_decision: int = 0
        self.chips_without_decision: int = 0

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def set_opponents(self, opponents: Optional[List[Any]]) -> None:
        """
        Swap the opponents this env deals against, keeping its RNG stream.

        Exists so a caller wanting a different opponent each hand changes the
        opponent rather than the environment. Rebuilding the env per hand
        re-seeds it, and a re-seeded env deals the same hand every time: the
        trainer did this and drew **two distinct deals across twelve hands**.
        That is the audit's original finding — "the deck re-dealt the same two
        hands every hand" — in a second file.

        Passing None restores the opponents the env was constructed with.
        """
        self.opponent_pool = list(opponents) if opponents else list(self._default_opponents)

    #: A hand the agent never acts in is re-dealt rather than returned. With an
    #: opponent that folds every hand the agent is dealt a decision only from
    #: the seat that acts first, so re-dealing can legitimately need several
    #: attempts — but never many. This bounds it so the loop cannot spin
    #: silently, which is the failure mode `reset_hand()` already has.
    MAX_REDEALS = 100

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Start a new hand in which the agent actually has a decision.

        A hand can end before the agent ever acts: heads-up, the agent is the
        big blind and the opponent folds preflop. Measured against a random
        opponent, that is **13% of hands**. This used to return the terminal
        zeros observation anyway; the caller — including `_collect_rollout` —
        would then step, an action would be applied to a dead hand, and both
        players would end up folded. The pot was destroyed, and the buffer
        received a transition with a zeros observation, an all-legal mask and
        a reward for a hand the agent never played.

        Such a hand carries no decision to learn from, so it is settled and
        re-dealt. Its chips are counted rather than discarded: they are won
        blinds, so dropping them silently would bias mean episode reward
        downward. `hands_without_decision` and `chips_without_decision` report
        the size of what was skipped.

        Returns
        -------
        obs:  Initial observation for the agent (float32, shape [OBS_SIZE]).
        info: Dict with metadata (game_seed, skipped-hand counters).
        """
        for _ in range(self.MAX_REDEALS):
            obs, seed = self._deal()
            if not self._is_done() and self._game.state.current_player is not None:
                return obs, {"game_seed": seed,
                             "hands_without_decision": self.hands_without_decision,
                             "chips_without_decision": self.chips_without_decision}

            finish_hand(self._game)
            self.hands_without_decision += 1
            self.chips_without_decision += (
                self._game.players[self._agent_id].stack - self.starting_stack)

        raise RuntimeError(
            f"{self.MAX_REDEALS} consecutive hands ended before the agent acted. "
            f"An opponent that never gives the agent a decision would otherwise "
            f"spin here forever, producing nothing and raising nothing.")

    def _deal(self) -> Tuple[np.ndarray, int]:
        """Deal one hand and run opponents up to the agent's first decision."""
        seed = int(self._rng.integers(0, 2**31))
        stacks = [self.starting_stack] * self.num_players

        # The agent moves seats, because the button cannot. `PokerGame` always
        # opens on button 0 and this env deals a *fresh* game per hand rather
        # than rotating one — deliberately, since reset_hand() does not restore
        # stacks. Left at seat 0 the agent would only ever be dealt the button,
        # learn one seat, and then be evaluated in both: benchmark() alternates
        # seats by construction. Drawing the seat here costs nothing and keeps
        # training and measurement over the same distribution of positions.
        self._agent_id = int(self._rng.integers(0, self.num_players))
        others = [s for s in range(self.num_players) if s != self._agent_id]
        self._seat_to_opponent = {
            seat: self.opponent_pool[i % len(self.opponent_pool)]
            for i, seat in enumerate(others)
        }

        self._game = PokerGame(
            player_stacks=stacks,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            seed=seed,
            enable_history=False,
        )

        self._episode_shaped_rewards = 0.0
        self._episode_actions = 0

        # Let opponents act until it is the agent's turn
        return self._advance_to_agent(), seed

    def step(
        self, action_idx: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply agent's action, let opponents respond, return next state.

        Returns
        -------
        obs:        Next observation (zeros if hand ended).
        reward:     Shaped step reward + terminal chip delta (in BB).
        terminated: True if hand is over.
        truncated:  Always False (no time-limit truncation).
        info:       Dict with chip_delta, bb_per_100, actions, etc.
        """
        game = self._game
        agent_id = self._agent_id
        prev_stack = game.players[agent_id].stack

        # A step on a finished hand is not recoverable, only maskable: the
        # engine accepts a fold from a player whose hand is already over, which
        # is how both players came to be folded and the pot to be destroyed.
        # reset() no longer returns such a state, so reaching here means the
        # caller stepped past a terminal one. Loud is the only safe option —
        # the quiet version corrupted 13% of every rollout.
        if self._is_done() or game.state.current_player != agent_id:
            raise RuntimeError(
                f"step() on a hand the agent cannot act in "
                f"(done={self._is_done()}, current_player="
                f"{game.state.current_player}, agent={agent_id}). "
                f"Call reset() after a terminated episode.")

        # Mask illegal actions
        mask = get_abstract_action_mask(game, agent_id)
        if not mask[action_idx]:
            action_idx = self._fallback_action(mask)

        # Apply agent action
        engine_action = abstract_action_to_engine_action(action_idx, game, agent_id)
        game.apply_action(agent_id, engine_action)
        self._episode_actions += 1

        # Shaped reward (immediate signal)
        shaped = self.reward_shaper.shape_step(
            game, agent_id, action_idx, prev_stack) * self.reward_scale
        self._episode_shaped_rewards += shaped

        # Check if hand is done
        terminated = self._is_done()

        if terminated:
            terminal_reward = self._terminal_reward(prev_stack)
            total_reward    = terminal_reward + self._episode_shaped_rewards
            obs             = np.zeros(self.OBS_SIZE, dtype=np.float32)
            chip_delta      = game.players[agent_id].stack - self.starting_stack
            info = {
                "chip_delta":  chip_delta,
                "bb_per_100":  (chip_delta / self.big_blind) * 100,
                "actions":     self._episode_actions,
                "shaped_bonus": self._episode_shaped_rewards,
            }
            return obs, total_reward, True, False, info

        # Let opponents act until agent's turn again (or hand ends)
        obs = self._advance_to_agent()

        if self._is_done():
            terminal_reward = self._terminal_reward(prev_stack)
            total_reward    = terminal_reward + self._episode_shaped_rewards
            chip_delta      = game.players[agent_id].stack - self.starting_stack
            info = {
                "chip_delta":  chip_delta,
                "bb_per_100":  (chip_delta / self.big_blind) * 100,
                "actions":     self._episode_actions,
                "shaped_bonus": self._episode_shaped_rewards,
            }
            return obs, total_reward, True, False, info

        return obs, shaped, False, False, {"actions": self._episode_actions}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_to_agent(self) -> np.ndarray:
        """
        Run opponent actions until it's the agent's turn or the hand ends.
        Returns the agent's observation.
        """
        game = self._game
        while True:
            if self._is_done():
                return np.zeros(self.OBS_SIZE, dtype=np.float32)

            current = game.state.current_player
            if current is None:
                return np.zeros(self.OBS_SIZE, dtype=np.float32)
            if current == self._agent_id:
                break

            # Opponent acts. The seat -> opponent map is fixed for the hand by
            # reset(); computing it from the seat index here assumed the agent
            # sat at 0, which it no longer does.
            opponent = self._seat_to_opponent[current]

            opp_action_idx = opponent.get_action(game, current)
            opp_mask       = get_abstract_action_mask(game, current)
            if not opp_mask[opp_action_idx]:
                opp_action_idx = self._fallback_action(opp_mask)

            eng_action = abstract_action_to_engine_action(opp_action_idx, game, current)
            game.apply_action(current, eng_action)

        obs = np.array(get_state_vector(game, self._agent_id), dtype=np.float32)
        return obs

    def _is_done(self) -> bool:
        """True if the hand has reached showdown or only one player remains."""
        game = self._game
        if game.state.betting_round == "showdown":
            return True
        active = [p for p in game.players if not p.has_folded]
        if len(active) <= 1:
            return True
        return False

    def _terminal_reward(self, prev_agent_stack: int) -> float:
        """
        Chip delta normalised to big-blind units.

        Using starting_stack as baseline so reward = 0 means break-even.

        The pot must be paid out first. `_is_done()` means the *betting* is
        finished, not that the chips have been awarded — at showdown they are
        still in the middle. Reading stacks without settling scored every hand
        as a loss of whatever the agent had contributed: measured over 300
        hands of random heads-up play, the reward was negative on 300 of them,
        mean -53.7 BB, in a symmetric matchup whose true expectation is zero.
        A policy gradient on that signal learns to fold and nothing else.

        The chip-conservation test did not catch it because it counts
        `sum(stacks) + pot.total`, which balances exactly while the pot is
        unpaid. `evaluation/benchmark.py` calls `finish_hand` for the same
        reason and documents the same trap.
        """
        finish_hand(self._game)
        final_stack = self._game.players[self._agent_id].stack
        chip_delta  = final_stack - self.starting_stack
        return chip_delta / self.big_blind * self.reward_scale

    @staticmethod
    def _fallback_action(mask: np.ndarray) -> int:
        """Return first legal action (check/call preferred, else fold)."""
        for preferred in (1, 0):            # try check/call first, then fold
            if mask[preferred]:
                return preferred
        legal = np.where(mask)[0]
        return int(legal[0]) if len(legal) else 1
