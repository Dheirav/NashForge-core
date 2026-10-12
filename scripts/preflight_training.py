"""
Is the evolutionary training loop sound this time?

The previous 89 runs were invalid and none of them looked it. A population
improved against itself for months while getting worse at poker, and the
numbers never said so. Tests passing is not the same evidence: the tests
exercise components, and what failed before was the loop as a whole.

So this runs the real thing, briefly, and checks the properties that were false
last time — each one stated as what it would mean if it failed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from engine import get_feature_names
from training.config import (EvolutionConfig, FitnessConfig, NetworkConfig,
                             TrainingConfig)
from training.evolution import EvolutionTrainer
from training.fitness import ILLEGAL_ACTIONS

FAILURES = []


def check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}\n         {detail}")
    if not ok:
        FAILURES.append(name)


#: Hands per matchup. The default is small so the preflight is quick, but the
#: repeatability check below is a question about the budget the *run* will use,
#: and at 120 it cannot answer it: the project's own measurements put the
#: re-scoring swing at 136 BB/100 over 2,000 hands and 31 over 24,000, so a
#: population scored twice at 480 hands per genome disagrees with itself by
#: design. Pass --hands-per-matchup to check the budget you are about to spend.
HANDS_PER_MATCHUP = int(os.environ.get("PREFLIGHT_HANDS", 120))
for _i, _a in enumerate(sys.argv):
    if _a == "--hands-per-matchup" and _i + 1 < len(sys.argv):
        HANDS_PER_MATCHUP = int(sys.argv[_i + 1])

config = TrainingConfig(
    network=NetworkConfig(hidden_sizes=[32, 16]),
    evolution=EvolutionConfig(population_size=10, mutation_sigma=0.1),
    fitness=FitnessConfig(num_players=2, hands_per_matchup=HANDS_PER_MATCHUP,
                          matchups_per_agent=4, starting_stack=200,
                          small_blind=1, big_blind=2, num_workers=1),
    num_generations=6, seed=0, experiment_name="preflight",
    output_dir="/tmp/preflight_out",
)

print("Running six real generations, population 10, heads-up.\n")
trainer = EvolutionTrainer(config)
# The constructor allocates the population but does not fill it; initialize()
# does. Calling train_generation() straight after construction evaluates an
# empty population, which fails inside opponent selection rather than saying so.
trainer.initialize()

before = (ILLEGAL_ACTIONS.rejected, ILLEGAL_ACTIONS.truncated)
stats = [trainer.train_generation() for _ in range(6)]
after = (ILLEGAL_ACTIONS.rejected, ILLEGAL_ACTIONS.truncated)

print("\n" + "=" * 70)
print("THE CHECKS THE PREVIOUS RUNS WOULD HAVE FAILED")
print("=" * 70 + "\n")

# 1 -----------------------------------------------------------------------
# Tested as a mean across generations against its own spread, not as a
# threshold on a single generation. A few hundred hands per genome is noisy
# enough that any fixed cutoff either fires on noise or misses real bias; what
# distinguishes the two is whether the sign persists.
means = [s.get("mean_fitness", s.get("avg_fitness")) for s in stats]
means = [m for m in means if m is not None]
average = float(np.mean(means))
spread = float(np.std(means, ddof=1)) / np.sqrt(len(means))
sigma = average / spread if spread else float("inf")
check("Population mean fitness is consistent with zero",
      abs(sigma) < 3.0,
      f"per-generation means {['%+.1f' % m for m in means]} BB/100; "
      f"average {average:+.2f} +/- {spread:.2f}, {sigma:+.1f} sigma from zero. "
      f"Poker is zero-sum, so this must not drift persistently positive — it "
      f"was large and positive in every run the audit examined, the single "
      f"check that would have caught everything. A small positive bias is "
      f"expected by design here, since 10% of sampled opponents are random "
      f"noise agents the population genuinely beats.")

# 2 -----------------------------------------------------------------------
check("No illegal actions were silently converted to folds",
      after == before,
      f"rejected {after[0] - before[0]}, truncated {after[1] - before[1]} during "
      f"training. Previously these were caught and turned into folds with "
      f"nothing counted, so an agent emitting illegal actions every hand scored "
      f"as merely weak.")

# 3 -----------------------------------------------------------------------
# Not a threshold on the best genome — with a small evaluation the best is the
# luckiest, and a cutoff cannot tell luck from skill. This evaluates the *same*
# population twice under different hand seeds and correlates the two rankings.
# If a genome's fitness does not survive re-measurement, selection optimises
# noise, which is what "the metric was too noisy to learn from" means in
# practice.
first = trainer.evaluator.evaluate_population(
    list(trainer.population), hall_of_fame=trainer.population.hall_of_fame)
second = trainer.evaluator.evaluate_population(
    list(trainer.population), hall_of_fame=trainer.population.hall_of_fame)


def scores(result):
    """Fitness per genome, keyed consistently so the two runs line up."""
    items = result.values() if isinstance(result, dict) else result
    ordered = sorted(items, key=lambda r: r.genome_id)
    return np.array([r.fitness for r in ordered], dtype=float)


a, b = scores(first), scores(second)
repeatability = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
swing = float(np.mean(np.abs(a - b)))
check("Fitness survives re-measurement well enough to select on",
      repeatability > 0.5,
      f"the same population scored twice under different hands correlates at "
      f"r = {repeatability:+.2f}, mean absolute change {swing:.0f} BB/100 per "
      f"genome. Below about 0.5 the ranking is mostly luck and selection "
      f"optimises noise — the audit measured +/-258 BB/100 over 600 hands and "
      f"named this the reason the old sweeps ranked configurations separated by "
      f"a fraction of their own error.")

# 4 -----------------------------------------------------------------------
names = get_feature_names()
check("Networks are built on the rebuilt observation",
      config.network.input_size == len(names),
      f"config input_size {config.network.input_size}, engine defines "
      f"{len(names)} features, genome {trainer.population.genomes[0].weights.size} parameters. "
      f"A width disagreement here feeds the network permuted inputs and trains "
      f"happily on them.")

# 5 -----------------------------------------------------------------------
generations = [s.get("generation") for s in stats]
check("Generation statistics describe the generation they name",
      generations == list(range(generations[0], generations[0] + len(generations))),
      f"reported generations {generations}. Statistics once described the "
      f"population from before selection rather than the one evaluated.")

print("\n" + "=" * 70)
if FAILURES:
    print(f"NOT SAFE TO RUN — {len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("All checks passed. The loop is sound on the properties that were false before.")
print("This does not prove the run will be interesting — only that it will be honest.")
