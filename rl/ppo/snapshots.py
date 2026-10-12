"""
The opponent pool for self-play: frozen copies of the policy's own past.

Why this exists rather than a hall of fame
------------------------------------------
`PPOTrainer` used to train against `RandomOpponent`, or against a hall-of-fame
directory of evolution genomes. The directory is empty
(`hall_of_fame/FAULTY_PIPELINE_NOTICE.md` says why), and its contents would be
unusable if it were not: every genome in it was bred on the fitness function
the August audit withdrew. Loading them would have reintroduced the exact
failure the audit found — measuring an agent against opponents selected by a
broken metric.

Self-play means the current policy *and its own past versions*, so that is what
this holds.

Frozen means frozen
-------------------
A snapshot is a deep copy of the weights into a separate network, in eval mode
with gradients disabled. Storing a reference to the live agent instead would
give a pool whose every member silently tracks the current policy — the pool
would appear to be doing its job, `len(pool)` would grow, and every opponent
would be the same live network. That failure is invisible from outside, so
`test_a_snapshot_does_not_follow_the_policy_that_made_it` pins it.

Cost: the network is 19 inputs through two 128-wide layers, roughly 20k
parameters, so eight snapshots are a few hundred kilobytes. This box is
memory-limited and has been terminated twice under pressure; the pool is not
where that will come from.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np


class SnapshotPool:
    """
    The last `capacity` policy snapshots, sampled uniformly.

    Uniform over a sliding window, rather than over the whole history: the
    point of the pool is to stop the policy from chasing a single opponent it
    has already beaten, and a window of recent-but-not-current selves does
    that. Keeping every snapshot ever taken would spend most hands on
    opponents the policy outgrew thousands of updates ago.

    Parameters
    ----------
    capacity: How many snapshots to retain. The plan's decision of 15 August
              is 5-10; the oldest is dropped past that.
    rng:      Generator used for sampling, so a seeded run is reproducible.
    """

    def __init__(self, capacity: int = 8, rng: Optional[np.random.Generator] = None):
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self._rng = rng if rng is not None else np.random.default_rng()
        self._agents: List = []
        #: The update cycle each retained snapshot was taken at, same order as
        #: `_agents`. Reported in the training log so the pool's age spread is
        #: visible rather than assumed.
        self.taken_at: List[int] = []
        #: Snapshots ever added, including those since evicted.
        self.total_taken = 0

    # ------------------------------------------------------------------

    def add(self, agent, update_cycle: int = -1) -> None:
        """Freeze `agent` as it is now and put the copy in the pool."""
        from rl.ppo.agent import PPOAgent

        frozen = PPOAgent(agent.config, device=str(agent.device))
        frozen.net.load_state_dict(copy.deepcopy(agent.net.state_dict()))
        frozen.net.eval()
        for parameter in frozen.net.parameters():
            parameter.requires_grad_(False)

        self._agents.append(frozen)
        self.taken_at.append(update_cycle)
        self.total_taken += 1

        while len(self._agents) > self.capacity:
            self._agents.pop(0)
            self.taken_at.pop(0)

    def sample(self):
        """One snapshot, uniformly. None if nothing has been snapshotted yet."""
        if not self._agents:
            return None
        return self._agents[int(self._rng.integers(0, len(self._agents)))]

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self):
        return iter(self._agents)
