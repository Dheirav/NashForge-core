"""
Evolutionary training system for poker AI.

This module provides:
- PolicyNetwork: Small neural network for action selection
- Genome: Weight representation for evolution
- Population: Collection of genomes with selection/mutation
- FitnessEvaluator: Self-play evaluation harness
- EvolutionTrainer: Main training loop

Example usage:
    from training import EvolutionTrainer, TrainingConfig
    
    trainer = EvolutionTrainer()
    trainer.initialize()
    best_genome = trainer.train(num_generations=100)
"""

from .config import (
    NetworkConfig,
    EvolutionConfig,
    FitnessConfig,
    TrainingConfig,
)

from .policy_network import PolicyNetwork, create_action_mask, ABSTRACT_ACTIONS

from .genome import (
    Genome,
    GenomeFactory,
    Population,
)

from .fitness import (
    FitnessEvaluator,
    abstract_action_to_engine_action,
)

from .evolution import (
    EvolutionTrainer,
    create_trainer,
)

from .self_play import (
    AgentPlayer,
    SessionStats,
    HandResult,
    load_agent,
    play_match,
    compare_agents,
    analyze_agent,
)

__all__ = [
    # Config
    'NetworkConfig',
    'EvolutionConfig',
    'FitnessConfig',
    'TrainingConfig',
    # Network
    'PolicyNetwork',
    'create_action_mask',
    'ABSTRACT_ACTIONS',
    # Genome
    'Genome',
    'GenomeFactory',
    'Population',
    # Fitness
    'FitnessEvaluator',
    'abstract_action_to_engine_action',
    # Training
    'EvolutionTrainer',
    'create_trainer',
    # Self-play
    'AgentPlayer',
    'SessionStats',
    'HandResult',
    'load_agent',
    'play_match',
    'compare_agents',
    'analyze_agent',
]
