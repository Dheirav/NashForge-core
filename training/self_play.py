"""
Self-play utilities for poker agents.
Provides functions for playing agents against each other and analysis.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .policy_network import PolicyNetwork, create_action_mask
from .config import FitnessConfig, NetworkConfig


@dataclass
class HandResult:
    """Result of a single hand."""
    winners: List[int]
    pot_size: int
    chip_changes: Dict[int, int]
    actions_count: int


@dataclass  
class SessionStats:
    """Statistics from a playing session."""
    hands_played: int
    total_chip_change: int
    bb_per_100: float
    win_rate: float  # Fraction of hands won
    vpip: float      # Voluntarily put money in pot
    pfr: float       # Preflop raise rate
    aggression: float  # Bets+Raises / Calls


class AgentPlayer:
    """
    Wrapper for playing a trained agent.
    """
    
    def __init__(self, network: PolicyNetwork, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None):
        self.network = network
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()
        
        # Stats tracking
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.bets_raises = 0
        self.calls = 0
    
    def get_action(self, game, player_id: int) -> int:
        """
        Get action for current game state.
        
        Returns abstract action index (0-5).
        """
        from engine import get_state_vector
        
        features = np.array(get_state_vector(game, player_id), dtype=np.float32)
        mask = create_action_mask(game, player_id)
        
        action_idx = self.network.select_action(
            features, mask, self.rng, self.temperature
        )
        
        return action_idx
    
    def reset_stats(self):
        """Reset tracking statistics."""
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.bets_raises = 0
        self.calls = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        hands = max(1, self.hands_played)
        actions = max(1, self.bets_raises + self.calls)
        
        return {
            'hands': self.hands_played,
            'vpip': self.vpip_count / hands,
            'pfr': self.pfr_count / hands,
            'aggression': self.bets_raises / actions if actions > 0 else 0,
        }


def load_agent(weights_path: str, config: Optional[NetworkConfig] = None,
               temperature: float = 1.0) -> AgentPlayer:
    """
    Load a trained agent from weights file.
    
    Args:
        weights_path: Path to .npy weights file
        config: Network config (default if None)
        temperature: Action sampling temperature
        
    Returns:
        AgentPlayer ready to play
    """
    weights = np.load(weights_path)
    
    network = PolicyNetwork(config or NetworkConfig())
    network.set_weights_from_genome(weights)
    
    return AgentPlayer(network, temperature)


def play_match(agents: List[AgentPlayer], num_hands: int = 100,
               config: Optional[FitnessConfig] = None,
               seed: Optional[int] = None,
               verbose: bool = False) -> Dict[int, SessionStats]:
    """
    Play a match between multiple agents.
    
    Args:
        agents: List of AgentPlayer instances
        num_hands: Number of hands to play
        config: Game config (default if None)
        seed: Random seed
        verbose: Print hand summaries
        
    Returns:
        Dict mapping agent index to SessionStats
    """
    from engine import PokerGame, Action, get_state_vector
    from .fitness import (abstract_action_to_engine_action, hand_start_stacks,
                          chip_deltas, apply_action_or_fold)
    
    config = config or FitnessConfig()
    rng = np.random.default_rng(seed)
    
    num_players = len(agents)
    starting_stacks = [config.starting_stack] * num_players
    
    # Track stats per agent
    stats = {i: {
        'chip_delta': 0,
        'hands': 0,
        'wins': 0,
        'vpip': 0,
        'pfr': 0,
        'bets_raises': 0,
        'calls': 0,
    } for i in range(num_players)}
    
    # Create game
    game = PokerGame(
        player_stacks=starting_stacks.copy(),
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        ante=config.ante,
        seed=rng.integers(0, 2**31)
    )
    
    for hand_num in range(num_hands):
        # Reset agent hand stats
        hand_vpip = {i: False for i in range(num_players)}
        hand_pfr = {i: False for i in range(num_players)}
        
        stacks_before = hand_start_stacks(game)

        action_count = 0
        max_actions = 200
        
        while not game.is_hand_over() and action_count < max_actions:
            current = game.state.current_player
            if current is None:
                break
            
            player = game.players[current]
            if player.has_folded or player.is_all_in:
                break
            
            # Get action from agent
            action_idx = agents[current].get_action(game, current)
            action = abstract_action_to_engine_action(action_idx, game, current)
            
            # Track stats
            street = game.state.betting_round
            if street == 'preflop':
                if action.action_type in ['call', 'raise', 'all-in']:
                    hand_vpip[current] = True
                if action.action_type in ['raise', 'all-in']:
                    hand_pfr[current] = True
            
            if action.action_type in ['raise', 'all-in']:
                stats[current]['bets_raises'] += 1
            elif action.action_type == 'call':
                stats[current]['calls'] += 1
            
            # Apply action, folding for the player if the engine refuses it.
            # Shares one handler with the fitness loop so rejected actions are
            # counted in ILLEGAL_ACTIONS rather than vanishing.
            if not apply_action_or_fold(game, current, action):
                break

            action_count += 1
        
        # Award the pot however the hand ended (fold-outs included)
        pot_before = game.state.pot.total
        winners = []
        if pot_before > 0:
            winnings = game.resolve_showdown()
            winners = [pid for pid, amt in winnings.items() if amt > 0]

        # Calculate deltas and update stats
        deltas = chip_deltas(game, stacks_before)
        for i, player in enumerate(game.players):
            delta = deltas[i]
            stats[i]['chip_delta'] += delta
            stats[i]['hands'] += 1
            
            if i in winners:
                stats[i]['wins'] += 1
            
            if hand_vpip[i]:
                stats[i]['vpip'] += 1
            if hand_pfr[i]:
                stats[i]['pfr'] += 1
        
        if verbose and hand_num % 10 == 0:
            print(f"Hand {hand_num}: pot={game.state.pot.total}, winners={winners}")
        
        # Check for busted players, reset stacks if needed
        active = [p for p in game.players if p.stack > 0]
        if len(active) < 2:
            for i, p in enumerate(game.players):
                game.players[i].stack = config.starting_stack
        
        # Reset for next hand
        try:
            game.reset_hand()
        except Exception:
            game = PokerGame(
                player_stacks=starting_stacks.copy(),
                small_blind=config.small_blind,
                big_blind=config.big_blind,
                ante=config.ante,
                seed=rng.integers(0, 2**31)
            )
    
    # Convert to SessionStats
    bb = config.big_blind
    results = {}
    
    for i in range(num_players):
        s = stats[i]
        hands = max(1, s['hands'])
        actions = max(1, s['bets_raises'] + s['calls'])
        
        results[i] = SessionStats(
            hands_played=s['hands'],
            total_chip_change=s['chip_delta'],
            bb_per_100=(s['chip_delta'] / bb) * (100 / hands),
            win_rate=s['wins'] / hands,
            vpip=s['vpip'] / hands,
            pfr=s['pfr'] / hands,
            aggression=s['bets_raises'] / actions,
        )
    
    return results


def compare_agents(agent1_path: str, agent2_path: str,
                   num_hands: int = 5000,
                   seed: int = 42) -> Tuple[float, float]:
    """
    Compare two trained agents head-to-head.
    
    Args:
        agent1_path: Path to first agent weights
        agent2_path: Path to second agent weights
        num_hands: Hands to play
        seed: Random seed
        
    Returns:
        Tuple of (agent1_bb_per_100, agent2_bb_per_100)
    """
    agent1 = load_agent(agent1_path)
    agent2 = load_agent(agent2_path)
    
    # Play 6-max with both agents
    agents = [agent1, agent2, agent1, agent2, agent1, agent2]
    
    results = play_match(agents, num_hands, seed=seed)
    
    # Average results for each agent type
    a1_bb = (results[0].bb_per_100 + results[2].bb_per_100 + results[4].bb_per_100) / 3
    a2_bb = (results[1].bb_per_100 + results[3].bb_per_100 + results[5].bb_per_100) / 3
    
    return a1_bb, a2_bb


def analyze_agent(weights_path: Optional[str] = None, num_hands: int = 2000,
                  verbose: bool = True, network: Optional[PolicyNetwork] = None) -> SessionStats:
    """
    Analyze a trained agent's playing style.
    
    Args:
        weights_path: Path to agent weights
        num_hands: Hands to analyze
        verbose: Print detailed stats
        
    Returns:
        SessionStats with playing style metrics
    """
    # Load agent
    if network is not None:
        agent = AgentPlayer(network)
    else:
        agent = load_agent(weights_path)
    
    # Create random opponents
    random_agents = []
    for _ in range(5):
        net = PolicyNetwork()
        net.set_weights_from_genome(np.random.randn(net.genome_size) * 0.1)
        random_agents.append(AgentPlayer(net))
    
    agents = [agent] + random_agents
    
    # Play and get stats
    results = play_match(agents, num_hands, verbose=False)
    hero_stats = results[0]
    
    if verbose:
        print(f"\nAgent Analysis ({num_hands} hands)")
        print("=" * 40)
        print(f"BB/100:      {hero_stats.bb_per_100:+.2f}")
        print(f"Win Rate:    {hero_stats.win_rate*100:.1f}%")
        print(f"VPIP:        {hero_stats.vpip*100:.1f}%")
        print(f"PFR:         {hero_stats.pfr*100:.1f}%")
        print(f"Aggression:  {hero_stats.aggression:.2f}")
        print("=" * 40)
        
        # Style interpretation
        if hero_stats.vpip > 0.4:
            print("Style: Loose")
        elif hero_stats.vpip < 0.2:
            print("Style: Tight")
        else:
            print("Style: Normal")
        
        if hero_stats.aggression > 0.6:
            print("Tendency: Aggressive")
        elif hero_stats.aggression < 0.4:
            print("Tendency: Passive")
        else:
            print("Tendency: Balanced")
    
    return hero_stats
