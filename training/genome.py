"""
Genome representation and population management for evolutionary training.

A genome is a flat numpy array representing neural network weights.
The population manages selection, mutation, and diversity preservation.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .config import NetworkConfig, EvolutionConfig
from .policy_network import PolicyNetwork

# Try to import Numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, fastmath=True)
def apply_mutation_jit(weights, noise):
    """
    JIT-compiled mutation application.
    Simply adds noise to weights element-wise.
    """
    return weights + noise


@jit(nopython=True, cache=True, fastmath=True)
def crossover_uniform_jit(parent1_weights, parent2_weights, mask):
    """
    JIT-compiled uniform crossover.
    Uses binary mask to select genes from each parent.
    
    Args:
        parent1_weights: First parent's weights
        parent2_weights: Second parent's weights
        mask: Binary mask (0 or 1) for selection
        
    Returns:
        Child weights
    """
    child = np.zeros_like(parent1_weights)
    for i in range(len(parent1_weights)):
        if mask[i]:
            child[i] = parent1_weights[i]
        else:
            child[i] = parent2_weights[i]
    return child


@jit(nopython=True, cache=True, fastmath=True)
def crossover_blend_jit(parent1_weights, parent2_weights, alpha):
    """
    JIT-compiled blend crossover (BLX-alpha).
    Creates child with weights in range around parents.
    
    Args:
        parent1_weights: First parent's weights
        parent2_weights: Second parent's weights
        alpha: Blending parameter (typically 0.5)
        
    Returns:
        Child weights
    """
    child = np.zeros_like(parent1_weights)
    for i in range(len(parent1_weights)):
        min_val = min(parent1_weights[i], parent2_weights[i])
        max_val = max(parent1_weights[i], parent2_weights[i])
        range_val = max_val - min_val
        
        # Blend with alpha expansion
        lower = min_val - alpha * range_val
        upper = max_val + alpha * range_val
        
        # Random value in expanded range
        child[i] = lower + np.random.random() * (upper - lower)
    
    return child


@dataclass
class Genome:
    """
    Represents a single agent's genotype.
    
    Attributes:
        genome_id: Unique identifier
        weights: Flat numpy array of network parameters
        fitness: Evaluated fitness score (None if not yet evaluated)
        generation: Generation this genome was created
        parent_id: ID of parent genome (None if initial)
        hof_opponents_used: List of HOF genome IDs used as opponents (optional tracking)
    """
    genome_id: int
    weights: np.ndarray
    fitness: Optional[float] = None
    generation: int = 0
    parent_id: Optional[int] = None
    hof_opponents_used: Optional[List[int]] = field(default_factory=list)
    
    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        return Genome(
            genome_id=self.genome_id,
            weights=self.weights.copy(),
            fitness=self.fitness,
            generation=self.generation,
            parent_id=self.parent_id,
            hof_opponents_used=self.hof_opponents_used.copy() if self.hof_opponents_used else []
        )
    
    def __repr__(self) -> str:
        fit_str = f"{self.fitness:.2f}" if self.fitness is not None else "N/A"
        return f"Genome(id={self.genome_id}, fitness={fit_str}, gen={self.generation})"


class GenomeFactory:
    """
    Creates and mutates genomes.
    
    Handles:
        - Random genome initialization
        - Mutation with Gaussian noise
        - Crossover (optional)
        - Conversion to/from neural networks
    """
    
    def __init__(self, network_config: NetworkConfig,
                 evolution_config: EvolutionConfig,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize factory.
        
        Args:
            network_config: Neural network architecture config
            evolution_config: Evolution parameters
            rng: Random number generator for reproducibility
        """
        self.network_config = network_config
        self.evolution_config = evolution_config
        self.rng = rng or np.random.default_rng()
        
        # Create template network to get genome size
        self._network = PolicyNetwork(network_config)
        self.genome_size = self._network.genome_size
        
        # ID counter
        self._next_id = 0
        
        # Current mutation sigma (may decay)
        self.current_sigma = evolution_config.mutation_sigma
    
    def _get_next_id(self) -> int:
        """Get next unique genome ID."""
        gid = self._next_id
        self._next_id += 1
        return gid
    
    def create_random(self, generation: int = 0) -> Genome:
        """
        Create a randomly initialized genome.
        
        Uses Xavier/He-like initialization scaled by mutation sigma.
        
        Args:
            generation: Generation number for metadata
            
        Returns:
            New randomly initialized genome
        """
        # Use small random weights (Xavier-like)
        weights = self.rng.standard_normal(self.genome_size) * 0.1
        
        return Genome(
            genome_id=self._get_next_id(),
            weights=weights.astype(np.float32),
            fitness=None,
            generation=generation,
            parent_id=None,
        )
    
    def create_from_weights(self, weights: np.ndarray,
                           generation: int = 0) -> Genome:
        """
        Create genome from existing weights.
        
        Args:
            weights: Weight array
            generation: Generation number
            
        Returns:
            New genome with given weights
        """
        return Genome(
            genome_id=self._get_next_id(),
            weights=weights.astype(np.float32),
            fitness=None,
            generation=generation,
            parent_id=None,
        )
    
    def mutate(self, parent: Genome, generation: int) -> Genome:
        """
        Create mutated offspring from parent.
        
        Applies Gaussian noise to all weights.
        
        Args:
            parent: Parent genome to mutate
            generation: Current generation number
            
        Returns:
            New mutated genome
        """
        # Generate noise
        noise = self.rng.standard_normal(self.genome_size).astype(np.float32) * self.current_sigma
        
        if HAS_NUMBA:
            # Use JIT-compiled mutation (1.5-2× faster)
            new_weights = apply_mutation_jit(parent.weights, noise)
        else:
            # Fallback: numpy implementation
            new_weights = parent.weights + noise
        
        return Genome(
            genome_id=self._get_next_id(),
            weights=new_weights.astype(np.float32),
            fitness=None,
            generation=generation,
            parent_id=parent.genome_id,
        )
    
    def crossover(self, parent1: Genome, parent2: Genome,
                  generation: int) -> Genome:
        """
        Create offspring via uniform crossover.
        
        Each gene randomly taken from one parent.
        
        Args:
            parent1: First parent
            parent2: Second parent
            generation: Current generation
            
        Returns:
            New crossover offspring
        """
        mask = self.rng.random(self.genome_size) < 0.5
        new_weights = np.where(mask, parent1.weights, parent2.weights)
        
        return Genome(
            genome_id=self._get_next_id(),
            weights=new_weights.astype(np.float32),
            fitness=None,
            generation=generation,
            parent_id=parent1.genome_id,
        )
    
    def decay_sigma(self):
        """Decay mutation sigma by configured rate."""
        self.current_sigma *= self.evolution_config.mutation_decay
    
    def to_network(self, genome: Genome) -> PolicyNetwork:
        """
        Convert genome to neural network.
        
        Args:
            genome: Genome to convert
            
        Returns:
            PolicyNetwork with genome's weights
        """
        network = PolicyNetwork(self.network_config)
        network.set_weights_from_genome(genome.weights)
        return network


class Population:
    """
    Manages a population of genomes.
    
    Handles:
        - Initialization (random or seeded)
        - Fitness-based selection
        - Elitism
        - Tournament selection
        - Random immigrants for diversity
        - Hall of fame for opponent diversity
    """
    
    def __init__(self, factory: GenomeFactory,
                 config: EvolutionConfig,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize population manager.
        
        Args:
            factory: GenomeFactory for creating/mutating genomes
            config: Evolution configuration
            rng: Random number generator
        """
        self.factory = factory
        self.config = config
        self.rng = rng or np.random.default_rng()
        
        self.genomes: List[Genome] = []
        self.hall_of_fame: List[Genome] = []
        self.generation = 0
    
    def initialize(self, size: Optional[int] = None,
                   seed_genome: Optional[Genome] = None):
        """
        Initialize population with random genomes.
        
        Args:
            size: Population size (uses config default if None)
            seed_genome: Optional genome to seed population with
        """
        size = size or self.config.population_size
        self.genomes = []
        
        if seed_genome is not None:
            # Add seed and mutations of it
            self.genomes.append(seed_genome.copy())
            
            for _ in range(size - 1):
                mutated = self.factory.mutate(seed_genome, 0)
                self.genomes.append(mutated)
        else:
            # All random
            for _ in range(size):
                self.genomes.append(self.factory.create_random(0))
        
        self.generation = 0
    
    def sort_by_fitness(self):
        """Sort genomes by fitness (highest first)."""
        self.genomes.sort(
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )
    
    def get_elites(self) -> List[Genome]:
        """
        Get elite genomes (top performers).
        
        Returns:
            List of elite genomes (copied)
        """
        self.sort_by_fitness()
        num_elites = self.config.num_elites
        return [g.copy() for g in self.genomes[:num_elites]]
    
    def tournament_select(self) -> Genome:
        """
        Select parent via tournament selection.
        
        Picks k random candidates and returns the fittest.
        
        Returns:
            Selected genome
        """
        k = min(self.config.tournament_size, len(self.genomes))
        candidates = self.rng.choice(self.genomes, size=k, replace=False)
        
        best = max(
            candidates,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf')
        )
        return best
    
    def update_hall_of_fame(self):
        """
        Update hall of fame with best current genomes.
        
        Keeps the top genomes seen across all generations.
        
        NOTE: This method is currently DISABLED and not called during training.
        HOF tracking is now only done at initialization to observe pre-seeded members.
        """
        # Add top genomes from current population
        self.sort_by_fitness()
        
        for genome in self.genomes[:3]:  # Add top 3
            if genome.fitness is not None:
                # Check if significantly different from existing HoF
                if self._is_novel_for_hof(genome):
                    self.hall_of_fame.append(genome.copy())
        
        # Sort HoF by fitness and trim
        self.hall_of_fame.sort(
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )
        self.hall_of_fame = self.hall_of_fame[:self.config.hof_size]
    
    def _is_novel_for_hof(self, genome: Genome,
                         novelty_threshold: float = 0.1) -> bool:
        """
        Check if genome is sufficiently different from HoF members.
        
        Args:
            genome: Candidate genome
            novelty_threshold: Minimum distance threshold
            
        Returns:
            True if novel enough to add
        """
        if not self.hall_of_fame:
            return True
        
        for hof_genome in self.hall_of_fame:
            # Normalized L2 distance
            dist = np.linalg.norm(genome.weights - hof_genome.weights)
            dist /= np.sqrt(len(genome.weights))
            
            if dist < novelty_threshold:
                return False
        
        return True
    
    def evolve(self) -> (List['Genome'], dict):
        """
        Create next generation through selection and mutation.
        
        Returns:
            Tuple: (new list of genomes, dict with counts of elites and immigrants)
        """
        self.generation += 1
        new_genomes: List[Genome] = []
        
        # 1. Elitism - keep best unchanged
        elites = self.get_elites()
        num_elites = len(elites)
        new_genomes.extend(elites)
        
        # 2. Random immigrants for diversity
        num_immigrants = self.config.num_immigrants
        for _ in range(num_immigrants):
            immigrant = self.factory.create_random(self.generation)
            new_genomes.append(immigrant)
        
        # 3. Fill rest via tournament selection + mutation
        while len(new_genomes) < self.config.population_size:
            parent = self.tournament_select()
            child = self.factory.mutate(parent, self.generation)
            new_genomes.append(child)
        
        # 4. Decay mutation rate
        self.factory.decay_sigma()
        
        return new_genomes[:self.config.population_size], {'num_elites': num_elites, 'num_immigrants': num_immigrants}
    
    def replace(self, new_genomes: List[Genome]):
        """Replace current population with new genomes."""
        self.genomes = new_genomes
    
    def get_stats(self) -> dict:
        """
        Get population statistics, including diversity.
        
        Returns:
            Dictionary with mean, std, min, max, median, worst fitness, all fitnesses, all genomes, and diversity.
        """
        fitnesses = [
            g.fitness for g in self.genomes
            if g.fitness is not None
        ]
        genomes = [g.weights for g in self.genomes]
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'worst': 0.0,
            'diversity': 0.0,
            'all_fitness': [],
            'all_genomes': [],
        }
        if not fitnesses:
            return stats
        arr = np.array(fitnesses)
        stats['mean'] = float(np.mean(arr))
        stats['std'] = float(np.std(arr))
        stats['min'] = float(np.min(arr))
        stats['max'] = float(np.max(arr))
        stats['median'] = float(np.median(arr))
        stats['worst'] = float(np.min(arr))
        stats['all_fitness'] = arr.tolist()
        if genomes:
            arr_genomes = np.stack(genomes)
            stats['all_genomes'] = arr_genomes.tolist()
            if len(arr_genomes) > 1:
                dists = np.linalg.norm(arr_genomes[:, None] - arr_genomes, axis=-1)
                # Only upper triangle, excluding diagonal
                iu = np.triu_indices(len(arr_genomes), k=1)
                pairwise = dists[iu]
                stats['diversity'] = float(np.mean(pairwise))
        return stats
    
    def get_random_opponent_from_hof(self) -> Optional[Genome]:
        """Get random genome from hall of fame."""
        if not self.hall_of_fame:
            return None
        return self.rng.choice(self.hall_of_fame)
    
    def __len__(self) -> int:
        return len(self.genomes)
    
    def __iter__(self):
        return iter(self.genomes)
    
    def __getitem__(self, idx: int) -> Genome:
        return self.genomes[idx]
