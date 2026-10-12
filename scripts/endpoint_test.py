"""
Did fifty generations of evolutionary search learn anything?

The per-generation panel scores could not answer this. They were taken over
3,000 hands, giving an error of about ±57 BB/100, and across all eleven
readings they scattered with a standard deviation of 56 — indistinguishable
from a constant. A curve measured that coarsely can only ever look like noise.

So this compares the two endpoints directly and precisely: the trained genome
against an untrained one drawn from the same initial distribution, both against
the same panel, both at 40,000 hands. That is ±15 BB/100, which resolves the
~35 BB/100 the training instrument can express.

The comparison that matters is the *difference* between the two columns. Each
individually is a fact about the opponent as much as the agent.
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from evaluation import always_call_agent, benchmark, cfr_agent, random_agent
from training.config import (EvolutionConfig, FitnessConfig, NetworkConfig,
                             TrainingConfig)
from training.evolution import EvolutionTrainer
from training.policy_network import PolicyNetwork

HANDS = 40_000
RUN = sorted(glob.glob("/home/dheirav/pokerbot-scratch/phase2/phase2/runs/run_*"))[-1]


def genome_agent(weights, network_config, rng):
    from engine.features import FeatureCache
    net = PolicyNetwork(network_config)
    net.set_weights_from_genome(weights)
    caches = {}

    def act(game, player_id, mask, history):
        key = (id(game), player_id)
        if key not in caches:
            caches.clear()
            caches[key] = FeatureCache(game, player_id)
        return int(net.select_action(caches[key].get_features(game),
                                     np.asarray(mask), rng))
    return act


config = TrainingConfig(
    network=NetworkConfig(hidden_sizes=[64, 32]),
    evolution=EvolutionConfig(population_size=30, mutation_sigma=0.1),
    fitness=FitnessConfig(num_players=2, starting_stack=200,
                          small_blind=1, big_blind=2, num_workers=1),
    num_generations=1, seed=0, experiment_name="endpoint",
    output_dir="/tmp/endpoint")

trained = np.load(os.path.join(RUN, "best_genome.npy"))

# The baseline is drawn from the same initial distribution and the same seed the
# run started from, so the only difference between the two columns is the fifty
# generations of selection applied to one of them.
baseline_trainer = EvolutionTrainer(config)
baseline_trainer.initialize()
untrained = baseline_trainer.population.genomes[0].weights.copy()

print(f"trained genome from {os.path.basename(RUN)}")
print(f"{HANDS:,} hands per matchup, duplicate play, seats alternating\n")

panel = []
rng = np.random.default_rng(4)
panel.append(("random", random_agent(rng)))
panel.append(("always-call", always_call_agent()))

try:
    import pickle
    with open("/home/dheirav/Code/PokerBot/results/cfr/nolimit_strategy.pkl", "rb") as fh:
        saved = pickle.load(fh)
    misses = [0, 0]
    panel.append(("cfr", cfr_agent(saved["strategy"], saved["abstraction"],
                                   rng, misses)))
    print("CFR agent loaded into the panel\n")
except Exception as error:                      # noqa: BLE001
    misses = None
    print(f"CFR agent unavailable ({error}); panel is the two baselines\n")

print(f"{'opponent':<14}{'untrained':>20}{'after 50 gens':>20}{'difference':>22}")
print("-" * 76)

for name, opponent in panel:
    row = []
    for weights in (untrained, trained):
        clock = time.perf_counter()
        result = benchmark(genome_agent(weights, config.network, rng), opponent,
                           name, hands=HANDS, seed=17,
                           misses=misses if name == "cfr" else None)
        row.append(result)
        print(f"    ({name} vs {'untrained' if len(row) == 1 else 'trained':<10}"
              f" {time.perf_counter() - clock:.0f}s)", flush=True)

    before, after = row
    delta = after.bb_per_100 - before.bb_per_100
    error = 1.96 * np.hypot(before.stderr, after.stderr) / 2 * 100
    verdict = "improved" if delta > error else \
              "worse" if delta < -error else "no change"
    print(f"{name:<14}{before.bb_per_100:>+13.1f} +/-{before.stderr / 2 * 100:<5.0f}"
          f"{after.bb_per_100:>+13.1f} +/-{after.stderr / 2 * 100:<5.0f}"
          f"{delta:>+12.1f} +/-{error:<4.0f} {verdict}\n", flush=True)

if misses:
    print(f"CFR lookup miss rate {misses[0] / max(1, misses[1]):.1%} — a benchmark "
          f"that has quietly become random would show here.")
