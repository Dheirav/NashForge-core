"""
Configuration dataclasses for evolutionary training.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NetworkConfig:
    """
    Policy network architecture configuration.
    
    The network maps game state features to action logits.
    
    Attributes:
        input_size: Number of input features (from engine.get_state_vector)
        hidden_sizes: List of hidden layer sizes
        output_size: Number of abstract actions (fold, check/call, raise_half, raise_pot, raise_2x, all-in)
        activation: Activation function ('relu', 'tanh', 'sigmoid')
    """
    input_size: int = 19          # From engine features — see get_feature_names()
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 6          # Abstract actions
    activation: str = 'relu'
    
    def __post_init__(self):
        # Ensure hidden_sizes is a list
        if isinstance(self.hidden_sizes, tuple):
            self.hidden_sizes = list(self.hidden_sizes)


@dataclass
class EvolutionConfig:
    """
    Evolution algorithm parameters.
    
    Attributes:
        population_size: Number of agents in population
        elite_fraction: Fraction of top performers to preserve unchanged
        tournament_size: Number of candidates for tournament selection
        mutation_rate: Probability of mutating each gene
        mutation_sigma: Standard deviation of Gaussian mutation noise
        mutation_decay: Factor to decay sigma each generation (1.0 = no decay)
        immigrant_fraction: Fraction of population to replace with random agents
        hof_size: Maximum hall-of-fame size for diversity
        hof_opponent_prob: Probability of facing hall-of-fame opponent
    """
    population_size: int = 20
    elite_fraction: float = 0.1    # Top 10% survive unchanged
    tournament_size: int = 3
    mutation_rate: float = 1.0     # Mutate all genes (ES-style)
    mutation_sigma: float = 0.1    # Mutation strength
    mutation_decay: float = 0.995  # Sigma decay per generation
    immigrant_fraction: float = 0.05  # 5% random immigrants
    hof_size: int = 10             # Keep best 10 ever
    hof_opponent_prob: float = 0.2 # 20% chance to face HoF opponent
    
    @property
    def num_elites(self) -> int:
        """Number of elite genomes to preserve."""
        return max(1, int(self.population_size * self.elite_fraction))
    
    @property
    def num_immigrants(self) -> int:
        """Number of random immigrants per generation."""
        return max(0, int(self.population_size * self.immigrant_fraction))


@dataclass
class FitnessConfig:
    """
    Fitness evaluation configuration.
    
    Fitness = average chip gain in big blinds per 100 hands (BB/100).
    
    Attributes:
        hands_per_matchup: Hands to play per matchup
        matchups_per_agent: Number of different opponents to face
        num_players: Players per table (2-6)
        starting_stack: Starting chips
        small_blind: Small blind amount
        big_blind: Big blind amount
        ante: Ante per player
        num_workers: Parallel workers for evaluation
        temperature: Action sampling temperature (lower = more deterministic)
    """
    hands_per_matchup: int = 500   # Hands per opponent set
    matchups_per_agent: int = 4    # Different opponent groups
    num_players: int = 6           # 6-max
    starting_stack: int = 1000     # 100BB
    small_blind: int = 5
    big_blind: int = 10
    ante: int = 0
    num_workers: int = 1          # CPU workers (set to 4 to match checkpoint)
    temperature: float = 1.0       # Action sampling temperature

    # ── Mixed-format training ────────────────────────────────────────────────
    # When heads_up_fraction > 0, some matchups are played 2-player (HU) and
    # the remainder at num_players (MultiTable).  The fitness score is a
    # weighted average: HU BB/100 * heads_up_fraction + MT BB/100 * (1 - frac).
    # Recommended value for a format-agnostic agent: 0.33 (1-in-3 matchups HU)
    heads_up_fraction: float = 0.0  # 0.0 = all MT, 1.0 = all HU
    # None means 'same as hands_per_matchup'. It used to default to 500, which
    # made hands_per_matchup silently inert in any heads-up configuration:
    # with num_players=2 every matchup is a 2-player one, so this branch always
    # won and the obvious parameter did nothing. Sweeping it compared
    # configurations that were identical.
    hu_hands_per_matchup: Optional[int] = None  # Hands for HU matchups; None = same as MT

    # ── Per-hand condition randomisation (opt-in) ────────────────────────────
    # These were previously read out of the config with getattr() defaults even
    # though the field did not exist, so EVERY evaluation silently randomised
    # stacks to 0.5-1.5x starting_stack, blinds to 1-2x, and injected a 0-1
    # ante — none of it recorded in the saved config.  It is now explicit and
    # off by default: each hand uses exactly the values declared above.
    #
    # Set randomise_conditions=True to opt in.  A None bound falls back to the
    # historical formula, so the old behaviour is still reproducible.
    randomise_conditions: bool = False
    stack_min: Optional[int] = None   # default: starting_stack // 2
    stack_max: Optional[int] = None   # default: starting_stack * 3 // 2
    sb_min:    Optional[int] = None   # default: small_blind
    sb_max:    Optional[int] = None   # default: small_blind * 2
    bb_min:    Optional[int] = None   # default: big_blind
    bb_max:    Optional[int] = None   # default: big_blind * 2
    ante_min:  Optional[int] = None   # default: ante
    ante_max:  Optional[int] = None   # default: max(1, ante * 2)

    @property
    def total_hands_per_agent(self) -> int:
        """Total hands played per fitness evaluation."""
        return self.hands_per_matchup * self.matchups_per_agent

    @property
    def num_hu_matchups(self) -> int:
        """Number of HeadsUp matchups per agent per generation."""
        return round(self.matchups_per_agent * self.heads_up_fraction)

    @property
    def num_mt_matchups(self) -> int:
        """Number of MultiTable matchups per agent per generation."""
        return self.matchups_per_agent - self.num_hu_matchups


@dataclass
class TrainingConfig:
        
    """
    Top-level training configuration.
    
    Attributes:
        network: Network architecture config
        evolution: Evolution algorithm config
        fitness: Fitness evaluation config
        num_generations: Total training generations
        seed: Master random seed
        output_dir: Directory for checkpoints
        experiment_name: Name for this training run
        log_interval: Generations between progress logs
        checkpoint_interval: Generations between checkpoints
    """
    network: NetworkConfig = field(default_factory=NetworkConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    num_generations: int = 100
    seed: int = 42
    output_dir: str = 'checkpoints'
    experiment_name: str = 'evolution_run'
    log_interval: int = 1
    checkpoint_interval: int = 10
    
    @classmethod
    def for_quick_test(cls) -> 'TrainingConfig':
        """Create config for quick testing."""
        return cls(
            evolution=EvolutionConfig(population_size=8),
            fitness=FitnessConfig(hands_per_matchup=50, matchups_per_agent=2),
            num_generations=5,
            seed=42,
        )
    
    @classmethod
    def for_production(cls, seed: int = 42) -> 'TrainingConfig':
        """Create config for production training."""
        return cls(
            network=NetworkConfig(hidden_sizes=[128, 64, 32]),
            evolution=EvolutionConfig(
                population_size=50,
                elite_fraction=0.1,
                mutation_sigma=0.05,
                hof_size=20,
            ),
            fitness=FitnessConfig(
                hands_per_matchup=1000,
                matchups_per_agent=8,
                num_workers=4,
            ),
            num_generations=500,
            seed=seed,
            checkpoint_interval=25,
        )

    @classmethod
    def for_phase_1(cls, seed: int = 42) -> 'TrainingConfig':
            """Config for Phase 1: Stabilization and Debug Phase."""
            return cls(
                network=NetworkConfig(hidden_sizes=[64, 32]),
                evolution=EvolutionConfig(
                    population_size=20,
                    elite_fraction=0.10,
                    mutation_sigma=0.01,
                    hof_size=6,
                ),
                fitness=FitnessConfig(
                    hands_per_matchup=3000,
                    matchups_per_agent=12,
                    num_workers=6,
                ),
                num_generations=100,
                seed=seed,
                checkpoint_interval=10,
            )

    @classmethod
    def for_balanced_formats(cls, seed: int = 42) -> 'TrainingConfig':
        """Config for format-agnostic training (HeadsUp + MultiTable mixed).

        Uses m=8 matchups per agent with 1-in-3 being HeadsUp and 2-in-3
        MultiTable, σ=0.08, h=500.  Empirically this produces agents that
        are competitive in both formats without catastrophic format inversion.
        """
        return cls(
            network=NetworkConfig(hidden_sizes=[64, 32]),
            evolution=EvolutionConfig(
                population_size=12,
                elite_fraction=0.1,
                mutation_sigma=0.08,
                hof_size=10,
                hof_opponent_prob=0.2,
            ),
            fitness=FitnessConfig(
                hands_per_matchup=500,
                matchups_per_agent=9,   # 3 HU + 6 MT = 9 total
                num_players=6,
                heads_up_fraction=0.333,  # 3 of 9 matchups are HU
                hu_hands_per_matchup=500,
                num_workers=1,
            ),
            num_generations=200,
            seed=seed,
            checkpoint_interval=10,
        )

    @classmethod
    def for_phase_2(cls, seed: int = 42) -> 'TrainingConfig':
            """Config for Phase 2: Strategy Refinement Phase."""
            return cls(
                network=NetworkConfig(hidden_sizes=[128, 64]),
                evolution=EvolutionConfig(
                    population_size=32,
                    elite_fraction=0.15,
                    mutation_sigma=0.005,
                    hof_size=10,
                ),
                fitness=FitnessConfig(
                    hands_per_matchup=4000,
                    matchups_per_agent=16,
                    num_workers=8,
                ),
                num_generations=200,
                seed=seed,
                checkpoint_interval=10,
            )