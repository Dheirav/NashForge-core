"""
Phase 2 — evolutionary search, heads-up, measured against an independent panel.

Two numbers are recorded per generation and they are not the same thing.

**Fitness** is self-play: a genome scored against sampled members of its own
population. It is what selection acts on, and it is exactly the measure that let
the previous pipeline improve for 89 runs while getting worse at poker. A rising
fitness curve on its own means nothing.

**Panel score** is the best genome played against opponents that share no
ancestry with it: random, always-call, and the CFR agent — the last validated
against Kuhn's analytic game value and exact Leduc exploitability. This is the
curve worth reading.

Budget is set from measurement rather than convention. Re-scoring the same
population under different hands moves a genome by roughly 136 BB/100 at 2,000
hands, 66 at 8,000 and 31 at 24,000, so 24,000 is used here and differences
smaller than about 35 BB/100 are below the resolution of the instrument.

The history file is written every generation, because it is small and it *is*
the result. The solver checkpoint follows the project convention of every 10% of
a run — ten saves whatever the length, so an interruption costs a proportion
rather than a count that means something different in every experiment. Both go
to a directory that survives a reboot; re-running resumes rather than restarts.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from engine import get_feature_names
from evaluation import (always_call_agent, benchmark, checkpoint_every,
                        random_agent)
from training.config import (EvolutionConfig, FitnessConfig, NetworkConfig,
                             TrainingConfig)
from training.evolution import EvolutionTrainer
from training.policy_network import PolicyNetwork

POPULATION = 30
GENERATIONS = 50
HANDS_PER_MATCHUP = 6000      # x4 matchups = 24,000 hands per genome
MATCHUPS = 4
BENCH_EVERY = 5
BENCH_HANDS = 3000

#: Overridable, for the reason train_ppo.py's --out-dir exists: a retrain that
#: writes where the run it supersedes wrote destroys the comparison.
OUT = os.environ.get("EVOLUTION_OUT", os.path.join(HERE, "phase2"))
RESULTS = os.path.join(OUT, "history.json")
os.makedirs(OUT, exist_ok=True)


def genome_agent(weights, network_config, rng):
    """Wrap a genome as a benchmark agent over the six abstract actions."""
    net = PolicyNetwork(network_config)
    net.set_weights_from_genome(weights)

    from engine.features import FeatureCache
    caches = {}

    def act(game, player_id, mask, history):
        key = (id(game), player_id)
        if key not in caches:
            caches.clear()
            caches[key] = FeatureCache(game, player_id)
        features = caches[key].get_features(game)
        return int(net.select_action(features, np.asarray(mask), rng))
    return act


def panel_scores(weights, network_config, rng):
    """The best genome against opponents unrelated to it."""
    agent = genome_agent(weights, network_config, rng)
    out = {}
    for name, opponent in (("random", random_agent(rng)),
                           ("always-call", always_call_agent())):
        result = benchmark(agent, opponent, name, hands=BENCH_HANDS, seed=99)
        out[name] = {"bb_per_100": result.bb_per_100,
                     "stderr_chips": result.stderr,
                     "separated": bool(result.separated_from_zero)}
    return out


def main():
    config = TrainingConfig(
        network=NetworkConfig(hidden_sizes=[64, 32]),
        evolution=EvolutionConfig(population_size=POPULATION, mutation_sigma=0.1),
        fitness=FitnessConfig(num_players=2, hands_per_matchup=HANDS_PER_MATCHUP,
                              matchups_per_agent=MATCHUPS, starting_stack=200,
                              small_blind=1, big_blind=2, num_workers=1),
        num_generations=GENERATIONS, seed=0, experiment_name="phase2",
        output_dir=OUT, checkpoint_interval=checkpoint_every(GENERATIONS),
    )

    history = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            history = json.load(handle)

    trainer = EvolutionTrainer(config)
    resume = os.path.join(OUT, "phase2", "checkpoint_latest.npz")
    started_at = len(history)
    if started_at and os.path.exists(resume):
        trainer.load_checkpoint(resume)
        print(f"resumed at generation {started_at}", flush=True)
    else:
        trainer.initialize()

    print(f"population {POPULATION}, {HANDS_PER_MATCHUP * MATCHUPS:,} hands per "
          f"genome, {len(get_feature_names())} features", flush=True)
    print(f"fitness is self-play; the panel score is the one that means "
          f"something\n", flush=True)

    rng = np.random.default_rng(7)
    for generation in range(started_at, GENERATIONS):
        clock = time.perf_counter()
        stats = trainer.train_generation()
        elapsed = time.perf_counter() - clock

        row = {"generation": generation,
               "mean_fitness": stats.get("mean_fitness", stats.get("avg_fitness")),
               "best_fitness": stats.get("best_fitness", stats.get("max_fitness")),
               "seconds": elapsed}

        if generation % BENCH_EVERY == 0 or generation == GENERATIONS - 1:
            best = trainer.population.genomes[0].weights
            row["panel"] = panel_scores(best, config.network, rng)
            versus = "  ".join(f"{k} {v['bb_per_100']:+.0f}"
                               for k, v in row["panel"].items())
        else:
            versus = ""

        history.append(row)
        # The history file is cheap and is the run's actual result, so it is
        # written every generation. The solver checkpoint, which is large,
        # follows the 10% convention.
        temporary = RESULTS + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(history, handle, indent=2)
        os.replace(temporary, RESULTS)
        if (generation + 1) % checkpoint_every(GENERATIONS) == 0 \
                or generation == GENERATIONS - 1:
            trainer.save_checkpoint()

        print(f"  gen {generation:>3}  fitness mean {row['mean_fitness']:+8.1f} "
              f"best {row['best_fitness']:+8.1f}  {elapsed:>5.0f}s   {versus}",
              flush=True)

    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
