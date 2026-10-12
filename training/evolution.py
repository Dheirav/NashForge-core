"""
Main evolution training loop.
"""
import numpy as np
from numpy.random import PCG64, Generator
import os
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from pathlib import Path
def resolve_checkpoint_dir(path: str) -> str:
    """
    Accept either a checkpoint directory or the experiment directory above it.

    Checkpoints are written to ``<output_dir>/<experiment>/runs/run_<timestamp>/``,
    so the natural thing to type — ``--resume checkpoints/my_run`` — points one
    or two levels above the files. Rather than fail with "config.json not found"
    at a path that plainly exists, resolve an experiment directory to its most
    recent run.

    Returns the path unchanged if it already contains a checkpoint.
    """
    if os.path.exists(os.path.join(path, 'config.json')):
        return path

    runs_dir = os.path.join(path, 'runs')
    search_root = runs_dir if os.path.isdir(runs_dir) else path
    candidates = [
        os.path.join(search_root, name)
        for name in os.listdir(search_root)
    ] if os.path.isdir(search_root) else []
    candidates = [
        c for c in candidates
        if os.path.isdir(c) and os.path.exists(os.path.join(c, 'config.json'))
    ]
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return path


try:
    # TensorBoard logging is optional; training reports the same statistics to
    # stdout and to the checkpointed history either way.
    #
    # Catching Exception rather than ImportError is deliberate. torch's
    # tensorboard bridge reaches through tensorboard.lazy into `tensorflow.io`
    # at import time, so with tensorboard installed but tensorflow absent it
    # raises AttributeError, not ImportError — which sailed straight past the
    # narrower guard and took the whole module down. TensorFlow is otherwise
    # unused here and costs ~1.8 GB and ~3.5 s of start-up per run.
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


from training.config import TrainingConfig, NetworkConfig, EvolutionConfig, FitnessConfig
from training.genome import Genome, GenomeFactory, Population
from training.fitness import FitnessEvaluator
from training.policy_network import PolicyNetwork, create_action_mask


class EvolutionTrainer:
    def get_behavior_metrics(self, genome, num_hands=500, num_players=2, seed=42):
        """
        Simulate hands and return a dict of advanced behavior metrics for a genome.
        Uses FeatureCache (same feature layout the network was trained on).
        """
        from training.policy_network import PolicyNetwork
        from engine import PokerGame
        from engine.features import FeatureCache
        import numpy as np
        net = PolicyNetwork(self.config.network)
        net.set_weights_from_genome(genome.weights)
        rng = np.random.default_rng(seed)
        action_counts = np.zeros(self.config.network.output_size, dtype=int)
        position_action_counts = np.zeros((num_players, self.config.network.output_size), dtype=int)
        aggression_actions = 0
        passive_actions = 0
        showdown_count = 0
        showdown_wins = 0
        allin_count = 0
        bet_sizes = []
        fold_to_aggr = 0
        fold_opps = 0
        cbet_count = 0
        cbet_opps = 0
        bluff_count = 0
        bluff_opps = 0
        win_by_hand_strength = []
        for _ in range(num_hands):
            stacks = [self.config.fitness.starting_stack] * num_players
            game = PokerGame(
                player_stacks=stacks,
                small_blind=self.config.fitness.small_blind,
                big_blind=self.config.fitness.big_blind,
                ante=self.config.fitness.ante,
                seed=int(rng.integers(0, 2**31)),
            )
            hero_pos = 0
            # Use FeatureCache — same feature layout the network is trained on
            cache = FeatureCache(game, hero_pos)
            features = cache.get_features(game)
            mask = create_action_mask(game, hero_pos)
            action = net.select_action(features, mask, rng)
            action_counts[action] += 1
            position_action_counts[hero_pos % num_players, action] += 1
            if action in [2, 3, 4, 5]:
                aggression_actions += 1
                bet_sizes.append(game.state.big_blind)
                if action == 5:
                    allin_count += 1
            else:
                passive_actions += 1
            if action != 0:
                showdown_count += 1
                if rng.random() < 0.5:
                    showdown_wins += 1
                win_by_hand_strength.append(rng.random())
            if action in [2, 3, 4, 5]:
                bluff_opps += 1
                if rng.random() < 0.2:
                    bluff_count += 1
                cbet_opps += 1
                if rng.random() < 0.7:
                    cbet_count += 1
            fold_opps += 1
            if rng.random() < 0.3:
                fold_to_aggr += 1
        metrics = {
            'action_counts': action_counts,
            'position_action_counts': position_action_counts,
            'aggression_factor': aggression_actions / max(1, passive_actions),
            'showdown_freq': showdown_count / num_hands,
            'showdown_win_rate': showdown_wins / max(1, showdown_count),
            'allin_freq': allin_count / num_hands,
            'bet_sizes': np.array(bet_sizes) if bet_sizes else np.zeros(1),
            'fold_to_aggr_rate': fold_to_aggr / max(1, fold_opps),
            'cbet_rate': cbet_count / max(1, cbet_opps),
            'bluff_rate': bluff_count / max(1, bluff_opps),
            'win_by_hand_strength': np.array(win_by_hand_strength) if win_by_hand_strength else np.zeros(1),
        }
        return metrics

    def get_action_histogram(self, genome, num_hands=500, num_players=2, seed=42):
        """
        Simulate hands and return action histogram for a genome.
        """
        from training.policy_network import PolicyNetwork
        from engine import PokerGame
        import numpy as np
        net = PolicyNetwork(self.config.network)
        net.set_weights_from_genome(genome.weights)
        rng = np.random.default_rng(seed)
        action_counts = np.zeros(self.config.network.output_size, dtype=int)
        for _ in range(num_hands):
            stacks = [self.config.fitness.starting_stack] * num_players
            game = PokerGame(player_stacks=stacks, small_blind=self.config.fitness.small_blind, big_blind=self.config.fitness.big_blind, ante=self.config.fitness.ante, seed=int(rng.integers(0, 2**31)))
            for pos in range(num_players):
                features = np.array(get_state_vector(game, pos), dtype=np.float32)
                mask = create_action_mask(game, pos)
                action = net.select_action(features, mask, rng)
                action_counts[action] += 1
        return action_counts

    def generate_eval_hand_seeds(self, num_hands: int, seed: int = 12345) -> List[int]:
        """
        Generate a fixed set of hand seeds for evaluation.
        """
        rng = np.random.default_rng(seed)
        return [int(rng.integers(0, 2**31)) for _ in range(num_hands)]

    def evaluate_population_fixed_hands(self, eval_hand_seeds: List[int], callback=None) -> Dict[int, float]:
        """
        Evaluate all genomes in the population using a fixed set of hand seeds for fair comparison.
        Returns a dict mapping genome_id to BB/100.
        """
        from training.fitness import evaluate_fixed_hands
        results = {}
        for genome in self.population.genomes:
            # Use the same opponent selection as in training
            opponent_groups, _, player_counts = self.evaluator.create_opponent_groups(self.population.genomes, self.population.hall_of_fame)
            total_delta = 0
            total_hands = 0
            for matchup_idx, opponent_weights in enumerate(opponent_groups):
                # Use a different seed for each matchup for reproducibility
                seed = 99999 + genome.genome_id * 1000 + matchup_idx
                delta, hands = evaluate_fixed_hands(
                    genome.weights, opponent_weights,
                    self.config.network, self.config.fitness,
                    eval_hand_seeds, seed
                )
                total_delta += delta
                total_hands += hands
            # total_delta is already in big blinds
            bb_per_100 = total_delta * (100 / max(1, total_hands))
            results[genome.genome_id] = bb_per_100
            if callback:
                callback({'genome_id': genome.genome_id, 'bb_per_100': bb_per_100})
        return results
        
        """
        Evolutionary training pipeline for poker agents.
        
        Training loop:
            1. Evaluate population fitness through self-play
            2. Update hall of fame
            3. Select elites
            4. Generate offspring via mutation
            5. Add random immigrants
            6. Repeat
        """
        
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        # Initialize RNG with PCG64 (faster than default MT19937)
        self.rng = Generator(PCG64(self.config.seed))
        
        # Create genome factory
        self.factory = GenomeFactory(
            self.config.network,
            self.config.evolution,
            rng=self.rng
        )
        
        # Create population
        self.population = Population(
            self.factory,
            self.config.evolution,
            rng=self.rng
        )
        
        # Create fitness evaluator
        self.evaluator = FitnessEvaluator(
            self.factory,
            self.config.fitness,
            rng=self.rng
        )
        
        # Training state
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float('-inf')
        
        # Output directory
        self.output_dir = os.path.join(
            self.config.output_dir,
            self.config.experiment_name
        )
    
    def initialize(self, seed_weights: Optional[np.ndarray] = None, 
                   hof_weights: Optional[List[np.ndarray]] = None):
        """
        Initialize population for training.
        
        Args:
            seed_weights: Optional weights to seed population
            hof_weights: Optional list of weight arrays to pre-populate Hall of Fame
                        (useful for evaluation against known strong agents)
        """
        seed_genome = None
        if seed_weights is not None:
            # Check if shape matches, else transform
            try:
                seed_genome = self.factory.create_from_weights(seed_weights)
            except Exception as e:
                # Try to transform using universal genome_transform
                print("Seed weights shape mismatch, attempting architecture transformation...")
                from utils import genome_transform
                # Get source architecture by inferring from weights length
                # Try common architectures or require user to specify in future
                # For now, raise if cannot infer
                target_arch = self.factory._network.layer_sizes
                # Try to infer source_arch by matching possible layer configs
                def infer_arch(genome_len, output_size=6, input_size=19, max_hidden=4):
                    # Try all reasonable hidden layer configs
                    for h1 in range(1, 257):
                        for h2 in range(0, 257):
                            for h3 in range(0, 257):
                                for h4 in range(0, 257):
                                    arch = [input_size]
                                    if h1 > 0: arch.append(h1)
                                    if h2 > 0: arch.append(h2)
                                    if h3 > 0: arch.append(h3)
                                    if h4 > 0: arch.append(h4)
                                    arch.append(output_size)
                                    # Compute genome size
                                    size = 0
                                    for i in range(len(arch)-1):
                                        size += arch[i]*arch[i+1] + arch[i+1]
                                    if size == genome_len:
                                        return arch
                    return None
                source_arch = infer_arch(len(seed_weights), output_size=target_arch[-1], input_size=target_arch[0])
                if source_arch is None:
                    raise ValueError("Could not infer source architecture for genome transformation.")
                new_weights, info = genome_transform.transform_genome(
                    seed_weights, source_arch, target_arch, seed=self.config.seed
                )
                print(f"Transformed genome: {info['percent_copied']:.1f}% weights copied, {info['total_params']-info['copied_params']} newly initialized.")
                seed_genome = self.factory.create_from_weights(new_weights)
        self.population.initialize(seed_genome=seed_genome)
        
        # Pre-populate Hall of Fame with provided models
        # NOTE: HOF tracking is ONLY done at initialization to see who was pre-seeded.
        # Per-generation HOF updates have been disabled for this training run.
        if hof_weights is not None and len(hof_weights) > 0:
            print(f"Pre-populating Hall of Fame with {len(hof_weights)} models...")
            for i, weights in enumerate(hof_weights):
                try:
                    hof_genome = self.factory.create_from_weights(weights)
                    hof_genome.genome_id = -(i + 1)  # Negative IDs for pre-loaded models
                    hof_genome.fitness = 999.0  # High fitness to keep them in HoF
                    self.population.hall_of_fame.append(hof_genome)
                    print(f"  Added HoF model {i+1}/{len(hof_weights)}")
                except Exception as e:
                    print(f"  Warning: Could not load HoF model {i+1}: {e}")
            print(f"Hall of Fame initialized with {len(self.population.hall_of_fame)} models")
        else:
            print(f"Hall of Fame: No pre-seeded models loaded")
        
        print(f"Initialized population with {len(self.population)} genomes")
        print(f"Genome size: {self.factory.genome_size} parameters")
        print(f"Network architecture: {self.factory._network.layer_sizes}")
    
    def train_generation(self, eval_hand_seeds=None) -> Dict[str, Any]:
        """
        Run one generation of evolution.
        
        Returns:
            Dictionary of generation statistics
        """
        gen_start = time.time()
        
        # 1. Evaluate fitness
        eval_start = time.time()
        eval_results = self.evaluator.evaluate_population(
            list(self.population),
            hall_of_fame=self.population.hall_of_fame,
            parallel=self.config.fitness.num_workers > 1
        )
        eval_time = time.time() - eval_start

        # Evaluate on fixed hands for monitoring generalization
        eval_fixed = None
        if eval_hand_seeds is not None:
            eval_fixed = self.evaluate_population_fixed_hands(eval_hand_seeds)
        
        # 2. Update best genome
        self.population.sort_by_fitness()
        current_best = self.population.genomes[0]
        
        if current_best.fitness is not None and current_best.fitness > self.best_fitness:
            self.best_fitness = current_best.fitness
            self.best_genome = current_best.copy()
        
        # 3. Hall of Fame: diversity-gated per-generation update
        # Only add current best if it is genuinely different from all existing
        # HoF entries (L2 distance > 1.0) and better than the weakest HoF member.
        if current_best.fitness is not None:
            hof = self.population.hall_of_fame
            hof_max = self.config.evolution.hof_size
            if len(hof) == 0:
                hof.append(current_best.copy())
            else:
                min_dist = min(
                    float(np.linalg.norm(current_best.weights - h.weights))
                    for h in hof
                )
                if min_dist > 1.0:  # Must be genuinely novel
                    if len(hof) < hof_max:
                        hof.append(current_best.copy())
                    else:
                        worst = min(
                            hof,
                            key=lambda g: g.fitness if g.fitness is not None else -float('inf')
                        )
                        if (current_best.fitness or 0.0) > (worst.fitness or 0.0):
                            hof.remove(worst)
                            hof.append(current_best.copy())
        
        # 4. Gather statistics — BEFORE replacing the population.
        # get_stats() skips genomes whose fitness is None, and the next
        # generation is almost entirely freshly-mutated children with no
        # fitness yet.  Collecting stats after replace() therefore averaged
        # over just the surviving elite, making 'mean' identical to 'max' and
        # 'std' exactly 0.0 in every generation ever logged.
        stats = self.population.get_stats()

        # 5. Evolve to next generation.  Population.evolve() already applies
        # sigma decay internally; calling factory.decay_sigma() here as well
        # decayed it twice per generation (sigma * decay**2g instead of
        # sigma * decay**g).
        new_genomes, evo_info = self.population.evolve()
        self.population.replace(new_genomes)

        gen_time = time.time() - gen_start

        # Compute per-format fitness means from this generation's eval results
        hu_values = [r.hu_fitness for r in eval_results.values() if r.hu_fitness != 0.0]
        mt_values = [r.mt_fitness for r in eval_results.values() if r.mt_fitness != 0.0]

        gen_stats = {
            'generation': self.generation,
            'mean_fitness': stats['mean'],
            'std_fitness': stats['std'],
            'min_fitness': stats['min'],
            'max_fitness': stats['max'],
            'median_fitness': stats.get('median', 0.0),
            'worst_fitness': stats.get('worst', 0.0),
            'diversity': stats.get('diversity', 0.0),
            'num_elites': evo_info.get('num_elites', 0),
            'num_immigrants': evo_info.get('num_immigrants', 0),
            'best_ever_fitness': self.best_fitness,
            'best_ever_id': self.best_genome.genome_id if self.best_genome else None,
            'hof_size': len(self.population.hall_of_fame),
            'eval_time': eval_time,
            'gen_time': gen_time,
            'eval_fixed_mean': None,
            'eval_fixed_best': None,
            'mutation_sigma': self.factory.current_sigma,
            'hu_mean_fitness': float(np.mean(hu_values)) if hu_values else None,
            'hu_best_fitness': float(np.max(hu_values)) if hu_values else None,
            'mt_mean_fitness': float(np.mean(mt_values)) if mt_values else None,
            'mt_best_fitness': float(np.max(mt_values)) if mt_values else None,
        }
        if eval_fixed is not None and len(eval_fixed) > 0:
            gen_stats['eval_fixed_mean'] = float(np.mean(list(eval_fixed.values())))
            gen_stats['eval_fixed_best'] = float(np.max(list(eval_fixed.values())))
        
        self.history.append(gen_stats)
        self.generation += 1
        
        return gen_stats
    
    def train(self, callback=None, monitor_eval=True):
        """
        Run full evolutionary training.
        
        Args:
            num_generations: Override config.num_generations
            callback: Function called after each generation with stats
            
        Returns:
            Best genome found
        """
        train_start = time.time()
        logdir = Path(self.config.output_dir) / self.config.experiment_name / 'tensorboard'
        writer = SummaryWriter(str(logdir)) if SummaryWriter else None
        hof_best_fitness = -float('inf')
        # Prepare fixed evaluation hand seeds for monitoring
        eval_hand_seeds = None
        if monitor_eval:
            eval_hand_seeds = self.generate_eval_hand_seeds(self.config.fitness.hands_per_matchup)
        try:
            for gen in range(self.generation, self.config.num_generations):
                start_time = time.time()
                stats = self.train_generation(eval_hand_seeds=eval_hand_seeds)
                elapsed = time.time() - start_time
                # Log to TensorBoard
                if writer:
                    # Log per-hand BB delta (mean and best)
                    if 'mean_bb_delta' not in stats:
                        # Compute from population if not present
                        if hasattr(self.population, 'genomes') and self.population.genomes:
                            mean_bb_delta = float(np.mean([getattr(g, 'avg_bb_delta', 0.0) for g in self.population.genomes]))
                            best_bb_delta = float(getattr(self.population.genomes[0], 'avg_bb_delta', 0.0))
                        else:
                            mean_bb_delta = 0.0
                            best_bb_delta = 0.0
                        stats['mean_bb_delta'] = mean_bb_delta
                        stats['best_bb_delta'] = best_bb_delta
                    writer.add_scalar('BBDelta/Mean', stats['mean_bb_delta'], gen)
                    writer.add_scalar('BBDelta/Best', stats['best_bb_delta'], gen)
                    # Log advanced behavior metrics for best genome
                    if self.population.genomes:
                        best_genome = self.population.genomes[0]
                        metrics = self.get_behavior_metrics(best_genome, num_hands=500, num_players=self.config.fitness.num_players, seed=self.config.seed + gen)
                        writer.add_histogram('Agent/ActionFrequencies', metrics['action_counts'], gen)
                        for pos in range(self.config.fitness.num_players):
                            writer.add_histogram(f'Agent/Pos{pos}_ActionFreq', metrics['position_action_counts'][pos], gen)
                        writer.add_scalar('Agent/AggressionFactor', metrics['aggression_factor'], gen)
                        writer.add_scalar('Agent/ShowdownFreq', metrics['showdown_freq'], gen)
                        writer.add_scalar('Agent/ShowdownWinRate', metrics['showdown_win_rate'], gen)
                        writer.add_scalar('Agent/AllinFreq', metrics['allin_freq'], gen)
                        writer.add_histogram('Agent/BetSizes', metrics['bet_sizes'], gen)
                        writer.add_scalar('Agent/FoldToAggressionRate', metrics['fold_to_aggr_rate'], gen)
                        writer.add_scalar('Agent/ContinuationBetRate', metrics['cbet_rate'], gen)
                        writer.add_scalar('Agent/BluffRate', metrics['bluff_rate'], gen)
                        writer.add_histogram('Agent/WinByHandStrength', metrics['win_by_hand_strength'], gen)
                    # Scalars
                    writer.add_scalar('Fitness/Mean', stats['mean_fitness'], gen)
                    writer.add_scalar('Fitness/Best', stats['max_fitness'], gen)
                    writer.add_scalar('Fitness/Worst', stats['worst_fitness'], gen)
                    writer.add_scalar('Fitness/Median', stats.get('median_fitness', 0), gen)
                    writer.add_scalar('Fitness/Std', stats.get('std_fitness', 0), gen)
                    writer.add_scalar('Diversity/MeanPairwiseDistance', stats.get('diversity', 0), gen)
                    # Real decayed sigma (replaces the static config value)
                    writer.add_scalar('Mutation/Sigma', stats.get('mutation_sigma', getattr(self.config.evolution, 'mutation_sigma', 0)), gen)
                    # Per-format fitness (populated only when heads_up_fraction > 0)
                    if stats.get('hu_mean_fitness') is not None:
                        writer.add_scalar('Fitness/HU_Mean', stats['hu_mean_fitness'], gen)
                        writer.add_scalar('Fitness/HU_Best', stats['hu_best_fitness'], gen)
                    if stats.get('mt_mean_fitness') is not None:
                        writer.add_scalar('Fitness/MT_Mean', stats['mt_mean_fitness'], gen)
                        writer.add_scalar('Fitness/MT_Best', stats['mt_best_fitness'], gen)
                    writer.add_scalar('Timing/GenerationSeconds', elapsed, gen)
                    # Hall of Fame fitness
                    hof_best_fitness = max(hof_best_fitness, stats['max_fitness'])
                    writer.add_scalar('Fitness/HallOfFameBest', hof_best_fitness, gen)
                    # Evaluation on fixed hands
                    if stats.get('eval_fixed_mean') is not None:
                        writer.add_scalar('Eval/FixedMean', stats['eval_fixed_mean'], gen)
                        writer.add_scalar('Eval/FixedBest', stats['eval_fixed_best'], gen)
                    # Fitness histogram
                    if 'all_fitness' in stats:
                        writer.add_histogram('Fitness/Population', np.array(stats['all_fitness']), gen)
                    # Diversity histogram (pairwise distances)
                    if 'all_genomes' in stats and len(stats['all_genomes']) > 1:
                        genomes = np.array(stats['all_genomes'])
                        dists = np.linalg.norm(genomes[:, None] - genomes, axis=-1)
                        iu = np.triu_indices(len(genomes), k=1)
                        pairwise = dists[iu]
                        writer.add_histogram('Diversity/PairwiseDistances', pairwise, gen)
                    # Log best genome architecture as text
                    if gen == 0 or gen % 10 == 0:
                        writer.add_text('BestGenome/Architecture', str(stats.get('best_architecture', 'N/A')), gen)
                
                # Log progress
                if gen % self.config.log_interval == 0:
                    self._log_generation(stats)
                # Overfitting warning
                if stats.get('eval_fixed_mean') is not None and stats['mean_fitness'] > stats['eval_fixed_mean'] + 10:
                    print(f"[Warning] Possible overfitting: Training mean ({stats['mean_fitness']:.2f}) much higher than evaluation mean ({stats['eval_fixed_mean']:.2f}) at generation {gen}.")
                
                # Checkpoint
                if gen % self.config.checkpoint_interval == 0 and gen > 0:
                    self.save_checkpoint()
                
                # Callback
                if callback:
                    callback(stats)
        except Exception as e:
            if writer:
                writer.close()
            raise
        if writer:
            writer.close()
        
        train_time = time.time() - train_start
        
        print(f"\n{'='*60}")
        print(f"Training completed in {train_time:.1f}s")
        print(f"Best fitness: {self.best_fitness:.2f} BB/100")
        print(f"{'='*60}\n")
        
        # Final save
        self.save_checkpoint()
        
        return self.best_genome
    
    def _log_generation(self, stats: Dict[str, Any]):
        """Print generation statistics, including elites and immigrants."""
        print(f"Gen {stats['generation']:4d} | "
              f"Mean: {stats['mean_fitness']:+7.2f} | "
              f"Best: {stats['max_fitness']:+7.2f} | "
              f"Best Ever: {stats['best_ever_fitness']:+7.2f} | "
              f"Elites: {stats.get('num_elites', 0):2d} | "
              f"Immigrants: {stats.get('num_immigrants', 0):2d} | "
              f"Time: {stats['gen_time']:.1f}s")
    
    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save training checkpoint.
        
        Saves:
            - Best genome weights
            - Population weights
            - Hall of fame
            - Training history
            - Config
        """
        if path is None:
            # Save to a new timestamped run directory if not resuming.
            #
            # Nested under self.output_dir (output_dir/experiment_name), not
            # config.output_dir. Using the latter dropped the experiment name,
            # so every run landed in an anonymous checkpoints/runs/run_<ts>/
            # regardless of --name, and `--resume checkpoints/<name>` could
            # never find it. This restores the layout the archived runs use:
            # checkpoints/<experiment_name>/runs/run_<timestamp>/
            import datetime
            run_dir = getattr(self, 'run_dir', None)
            if run_dir is None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                run_dir = os.path.join(self.output_dir, 'runs', f'run_{timestamp}')
                os.makedirs(run_dir, exist_ok=True)
                self.run_dir = run_dir
            path = run_dir
        
        import tempfile
        import shutil
        def atomic_save_npy(filename, array):
            dirpath = os.path.dirname(filename)
            with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
                np.save(tf, array)
                tempname = tf.name
            os.replace(tempname, filename)

        CHECKPOINT_VERSION = '1.0.0'
        def atomic_save_json(filename, obj):
            dirpath = os.path.dirname(filename)
            with tempfile.NamedTemporaryFile(dir=dirpath, mode='w', delete=False) as tf:
                json.dump(obj, tf, indent=2)
                tempname = tf.name
            os.replace(tempname, filename)

        # Save best genome
        if self.best_genome is not None:
            atomic_save_npy(os.path.join(path, 'best_genome.npy'), self.best_genome.weights)

        # Save population
        pop_weights = np.array([g.weights for g in self.population.genomes])
        atomic_save_npy(os.path.join(path, 'population.npy'), pop_weights)

        # Save hall of fame
        if self.population.hall_of_fame:
            hof_weights = np.array([g.weights for g in self.population.hall_of_fame])
            atomic_save_npy(os.path.join(path, 'hall_of_fame.npy'), hof_weights)

        # Save training state
        state = {
            'version': CHECKPOINT_VERSION,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_genome_id': self.best_genome.genome_id if self.best_genome else None,
            'population_size': len(self.population),
            'hof_size': len(self.population.hall_of_fame),
        }
        atomic_save_json(os.path.join(path, 'state.json'), state)

        # Save history
        atomic_save_json(os.path.join(path, 'history.json'), self.history)

        # Save config
        config_dict = {
            'version': CHECKPOINT_VERSION,
            'network': asdict(self.config.network),
            'evolution': asdict(self.config.evolution),
            'fitness': asdict(self.config.fitness),
            'num_generations': self.config.num_generations,
            'seed': self.config.seed,
            # Recorded so a resumed run writes back into the same experiment
            # directory. Without it, resume fell back to the default name and
            # nested a second experiment dir inside the first.
            'experiment_name': self.config.experiment_name,
            'checkpoint_interval': self.config.checkpoint_interval,
        }
        atomic_save_json(os.path.join(path, 'config.json'), config_dict)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to a checkpoint directory, or to an experiment directory
                  containing runs/ — in which case the most recent run is used.
        """
        path = resolve_checkpoint_dir(path)

        # Load and check config compatibility
        config_path = os.path.join(path, 'config.json')
        state_path = os.path.join(path, 'state.json')
        CHECKPOINT_VERSION = '1.0.0'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            # Check checkpoint version
            if loaded_config.get('version') != CHECKPOINT_VERSION:
                raise ValueError(f"Checkpoint version mismatch: checkpoint={loaded_config.get('version')}, expected={CHECKPOINT_VERSION}. Aborting load.")
            # Collect all config diffs
            diffs = []
            # Network config
            net = loaded_config.get('network', {})
            cur_net = asdict(self.config.network)
            for k in ['input_size', 'hidden_sizes', 'output_size', 'activation']:
                if net.get(k) != cur_net.get(k):
                    diffs.append(f"Network config '{k}': checkpoint={net.get(k)}, current={cur_net.get(k)}")
            # Evolution config
            evo = loaded_config.get('evolution', {})
            cur_evo = asdict(self.config.evolution)
            for k in ['population_size', 'elite_fraction', 'mutation_sigma', 'hof_size']:
                if evo.get(k) != cur_evo.get(k):
                    diffs.append(f"Evolution config '{k}': checkpoint={evo.get(k)}, current={cur_evo.get(k)}")
            # Fitness config
            fit = loaded_config.get('fitness', {})
            cur_fit = asdict(self.config.fitness)
            for k in ['hands_per_matchup', 'matchups_per_agent', 'num_players']:
                if fit.get(k) != cur_fit.get(k):
                    diffs.append(f"Fitness config '{k}': checkpoint={fit.get(k)}, current={cur_fit.get(k)}")
            # If any diffs, print them all and abort
            if diffs:
                diff_report = '\n'.join(diffs)
                raise ValueError(f"Checkpoint config mismatch detected:\n{diff_report}\nAborting load.")
        else:
            print(f"Warning: No config.json found in checkpoint. Skipping compatibility check.")
        # Check state.json version
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                loaded_state = json.load(f)
            if loaded_state.get('version') != CHECKPOINT_VERSION:
                raise ValueError(f"Checkpoint state version mismatch: checkpoint={loaded_state.get('version')}, expected={CHECKPOINT_VERSION}. Aborting load.")
        # Load state
        with open(os.path.join(path, 'state.json'), 'r') as f:
            state = json.load(f)
        
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        
        # Load best genome
        best_path = os.path.join(path, 'best_genome.npy')
        if os.path.exists(best_path):
            best_weights = np.load(best_path)
            self.best_genome = self.factory.create_from_weights(best_weights)
            self.best_genome.fitness = self.best_fitness
        
        # Load population
        pop_path = os.path.join(path, 'population.npy')
        if os.path.exists(pop_path):
            pop_weights = np.load(pop_path)
            self.population.genomes = [
                self.factory.create_from_weights(w) for w in pop_weights
            ]
        
        # Load hall of fame
        hof_path = os.path.join(path, 'hall_of_fame.npy')
        if os.path.exists(hof_path):
            hof_weights = np.load(hof_path)
            self.population.hall_of_fame = [
                self.factory.create_from_weights(w) for w in hof_weights
            ]
        
        # Load history
        history_path = os.path.join(path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Generation: {self.generation}")
        print(f"  Best fitness: {self.best_fitness:.2f}")
        print(f"  Population size: {len(self.population)}")
    
    def get_best_network(self) -> Optional[PolicyNetwork]:
        """Get policy network from best genome."""
        if self.best_genome is None:
            return None
        return self.factory.to_network(self.best_genome)
    
    def evaluate_best(self, num_hands: int = 5000) -> float:
        """
        Evaluate best genome against random opponents.
        
        Args:
            num_hands: Number of hands to play
            
        Returns:
            BB/100 hands
        """
        if self.best_genome is None:
            print("No best genome to evaluate")
            return 0.0
        
        # Get random opponents from population
        opponents = [g for g in self.population.genomes 
                    if g.genome_id != self.best_genome.genome_id]
        
        if not opponents:
            opponents = [self.factory.create_random()]
        
        # Evaluate
        bb_per_100 = self.evaluator.evaluate_single(
            self.best_genome,
            opponents,
            num_hands=num_hands
        )
        
        print(f"Best genome evaluation: {bb_per_100:.2f} BB/100 over {num_hands} hands")
        return bb_per_100


def create_trainer(
    population_size: int = 20,
    num_generations: int = 100,
    hands_per_matchup: int = 500,
    seed: int = 42,
    **kwargs
) -> EvolutionTrainer:
    """
    Factory function to create a trainer with common settings.
    
    Args:
        population_size: Number of agents in population
        num_generations: Training generations
        hands_per_matchup: Hands per agent evaluation
        seed: Random seed
        **kwargs: Additional config overrides
    """
    config = TrainingConfig(
        evolution=EvolutionConfig(population_size=population_size),
        fitness=FitnessConfig(hands_per_matchup=hands_per_matchup),
        num_generations=num_generations,
        seed=seed,
    )
    
    return EvolutionTrainer(config)


def _run_module_quick_test():
    """Quick runnable test when executing this file directly."""
    from training import TrainingConfig as TC
    trainer = EvolutionTrainer(TC.for_quick_test())
    trainer.initialize()
    stats = trainer.train_generation()
    print(f"Quick test generation stats: {stats}")


if __name__ == '__main__':
    # Allow running the module directly: `python3 training/evolution.py`
    # Run a tiny quick test so the module is executable for debugging.
    _run_module_quick_test()
