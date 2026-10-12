"""
PPO hyperparameter configuration.

Everything is in one dataclass so you can pass a single object around and
easily serialise/deserialise it (json.dumps(asdict(cfg))).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from typing import List, Optional


@dataclass
class PPOConfig:
    """
    Full configuration for a PPO training run.

    Network
    -------
    obs_size:        Feature vector length (must match engine.get_state_vector).
    num_actions:     Number of abstract actions (must match engine action space).
    hidden_size:     Width of each hidden layer in the actor-critic trunk.
    num_layers:      Number of shared trunk layers.

    Environment
    -----------
    num_players:     Table size.  2 = heads-up, 6 = 6-max.
    starting_stack:  Starting chip count per player.
    small_blind:     Small blind.
    big_blind:       Big blind.
    use_aggression_shaper: Add AggressionShaper dense reward.

    Rollout
    -------
    n_steps:         Agent *decisions* to collect per update, not hands. A
                     heads-up hand is about 1.8 decisions.
    n_envs:          Number of parallel environments (threads).

    PPO update
    ----------
    n_epochs:        Gradient epochs per rollout.
    batch_size:      Mini-batch size for gradient update.
    gamma:           Discount factor (set close to 1 for sparse terminal reward).
    gae_lambda:      GAE λ for advantage estimation.
    clip_range:      PPO clip ε.
    vf_coef:         Value function loss coefficient.
    ent_coef:        Entropy bonus coefficient.
    max_grad_norm:   Gradient clipping.
    lr:              Adam learning rate.
    lr_decay:        Multiply lr by this factor each update cycle (1.0 = no decay).
    target_kl:       Early-stop update if approx KL exceeds this (None = disabled).

    Training schedule
    -----------------
    total_hands:     Total environment steps (hands); controls run length.
    eval_every:      Evaluate vs opponent pool every N update cycles.
    save_every:      Save checkpoint every N update cycles.
    checkpoint_dir:  Directory to save checkpoints.
    log_dir:         Directory for training logs (CSV).

    Self-play opponents
    -------------------
    snapshot_every:      Update cycles between policy snapshots into the pool.
    snapshot_pool_size:  How many past snapshots to retain (the most recent N).
    current_policy_prob: Probability of facing the live policy rather than a
                         snapshot. Below 1.0 the policy spends most hands
                         against its own past, which is the point.

    The hall-of-fame options that used to live here are gone. The directory is
    empty, and every genome that would have filled it was bred on the fitness
    function the audit withdrew — see rl/ppo/snapshots.py.
    """

    # ── Network ───────────────────────────────────────────────────────
    obs_size:    int = 19
    num_actions: int = 6
    hidden_size: int = 128
    num_layers:  int = 2

    # ── Environment ───────────────────────────────────────────────────
    num_players:            int  = 2
    starting_stack:         int  = 1_000
    small_blind:            int  = 5
    big_blind:              int  = 10
    use_aggression_shaper:  bool = False
    # None derives it as big_blind / starting_stack, putting the reward in
    # stacks rather than big blinds so returns are O(1) and the critic does not
    # swamp the policy through the shared trunk. See PokerEnv.reward_scale for
    # the measurement. Set a float to override.
    reward_scale:           Optional[float] = None

    # ── Rollout ───────────────────────────────────────────────────────
    # n_steps counts *decisions*, not hands. Heads-up against a random
    # opponent that is about 1.8 decisions per hand, measured, so 512 steps is
    # roughly 280 hands. The docstring above used to call these hands, which is
    # the same class of mistake as commit 9d9d88e in the evolutionary loop:
    # a budget parameter whose name means something other than what it counts.
    n_steps: int = 512    # decisions per rollout
    n_envs:  int = 1      # parallel envs (kept as 1 for simplicity; extend later)

    # ── PPO update ────────────────────────────────────────────────────
    n_epochs:       int   = 4
    batch_size:     int   = 64
    gamma:          float = 0.999    # episodes are short; keep high
    gae_lambda:     float = 0.95
    clip_range:     float = 0.2
    vf_coef:        float = 0.5
    ent_coef:       float = 0.01
    max_grad_norm:  float = 0.5
    lr:             float = 3e-4
    lr_decay:       float = 1.0
    target_kl:      Optional[float] = None

    # ── Training schedule ─────────────────────────────────────────────
    total_hands:     int = 500_000
    eval_every:      int = 20        # update cycles between evals
    save_every:      int = 50        # update cycles between checkpoints
    checkpoint_dir:  str = "checkpoints/ppo"
    log_dir:         str = "logs/ppo"

    # ── Self-play opponents ───────────────────────────────────────────
    # Decided 15 August: snapshot every ~10 updates, keep the last 5-10, face
    # the pool most of the time. See docs/training-plan.md.
    snapshot_every:       int   = 10
    snapshot_pool_size:   int   = 8
    current_policy_prob:  float = 0.2

    # ── Misc ─────────────────────────────────────────────────────────
    seed:    Optional[int] = None
    device:  str           = "cpu"   # "cuda", "mps", "cpu"
    verbose: int           = 1       # 0=silent, 1=progress, 2=debug

    # ------------------------------------------------------------------

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PPOConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def __post_init__(self):
        # Sanity checks
        assert self.n_steps >= self.batch_size, \
            "n_steps must be >= batch_size"
        assert 0 < self.clip_range < 1, \
            "clip_range should be in (0, 1)"
        assert self.snapshot_every >= 1, \
            "snapshot_every counts update cycles and must be at least 1"
        assert self.snapshot_pool_size >= 1, \
            "snapshot_pool_size must retain at least one snapshot"
        assert 0.0 <= self.current_policy_prob <= 1.0, \
            "current_policy_prob is a probability"

    @property
    def effective_reward_scale(self) -> float:
        """`reward_scale`, or the table-derived default when it is None."""
        if self.reward_scale is not None:
            return float(self.reward_scale)
        return self.big_blind / self.starting_stack

    @classmethod
    def heads_up_default(cls) -> "PPOConfig":
        """
        Sensible defaults for heads-up training.

        The table is 200 chips at blinds 1/2 because that is the table
        `evaluation.benchmark` plays on and the one Phase 2's evolutionary run
        used. It is 100 big blinds either way, so this changes the unit rather
        than the game — but Phase 4 compares the two families directly, and a
        difference that has to be explained away is worse than one that does
        not exist.

        `ent_coef` is 0.02, and it is a choice on a proxy rather than a
        measured optimum. Calibrated 17 August at 200,000 hands over
        {0.01, 0.02, 0.05}, scored against the CFR agent at three training
        seeds each:

            0.01   -68.6  +19.5   -3.3   mean -17.5   spread 45.7
            0.02   +61.7  -28.3   -2.7   mean +10.2   spread 46.4
            0.05   -34.7  -55.4  +13.2   mean -25.6   spread 35.2

        The spread across seeds is as large as the gap between coefficients,
        so the panel cannot separate them at this budget and any ranking read
        off it is the seed. What can be separated is entropy: 0.01 fell to
        0.503 by 100,000 hands before recovering to 0.948, and it is the only
        candidate that came near collapse. A deterministic policy in an
        imperfect-information game is maximally exploitable and stops
        exploring, and 200,000 hands is 2.5% of the intended budget.

        So 0.02 is chosen as the lower of the two that showed no collapse
        risk, and 0.05 holds entropy at 80% of uniform, which is closer to
        regularising the policy into randomness than to training it.

        Recorded here rather than in a commit message because the next person
        to read a Phase 3 number needs to know it was picked this way.
        """
        return cls(
            num_players=2,
            starting_stack=200,
            small_blind=1,
            big_blind=2,
            n_steps=512,
            total_hands=500_000,
            hidden_size=128,
            ent_coef=0.02,
            current_policy_prob=0.2,
        )

    @classmethod
    def multitable_default(cls) -> "PPOConfig":
        """Sensible defaults for 6-max training."""
        return cls(
            num_players=6,
            starting_stack=200,
            small_blind=1,
            big_blind=2,
            n_steps=1024,
            total_hands=2_000_000,
            hidden_size=256,
            ent_coef=0.005,
            current_policy_prob=0.2,
        )
