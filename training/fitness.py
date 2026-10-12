"""
Fitness evaluation through self-play.

Evaluates agent fitness by playing poker hands against other agents.
Fitness = average big blinds won per 100 hands (BB/100).
"""
import warnings
import numpy as np
from numpy.random import PCG64, Generator
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

from .config import FitnessConfig, NetworkConfig
from .genome import Genome, GenomeFactory
from .policy_network import PolicyNetwork, create_action_mask

# Action object cache to avoid repeated object creation
_ACTION_CACHE = {}

# Game object pool for memory reuse
class GamePool:
    """
    Pool of reusable PokerGame objects to reduce allocation overhead.
    Provides ~1.2-1.4× speedup by reusing game objects.
    """
    def __init__(self, pool_size: int = 100):
        self.available = []
        self.pool_size = pool_size
    
    def acquire(self, player_stacks, small_blind, big_blind, ante, seed):
        """Get a game from pool or create new one."""
        if self.available:
            game = self.available.pop()
            # Reset game state
            game.__init__(
                player_stacks=player_stacks,
                small_blind=small_blind,
                big_blind=big_blind,
                ante=ante,
                seed=seed,
                enable_history=False
            )
            return game
        # Create new if pool empty
        from engine import PokerGame
        return PokerGame(
            player_stacks=player_stacks,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            seed=seed,
            enable_history=False
        )
    
    def release(self, game):
        """Return game to pool."""
        if len(self.available) < self.pool_size:
            self.available.append(game)

# Global game pool (thread-local would be better for multiprocessing)
_GAME_POOL = GamePool(100)


@dataclass
class EvalResult:
    """Result of evaluating a single genome."""
    genome_id: int
    fitness: float  # BB/100 (combined)
    total_hands: int
    total_bb_delta: float  # hero's own net result, in big blinds
    matchups_played: int
    hu_fitness: float = 0.0   # BB/100 in heads-up matchups (0 when none played)
    mt_fitness: float = 0.0   # BB/100 in multi-table matchups (0 when none played)


def abstract_action_to_engine_action(action_idx: int, game, player_id: int):
    """
    Convert abstract action index to engine Action.
    Optimized with cached action objects for frequently used actions.
    
    Abstract actions:
        0: fold
        1: check/call
        2: raise 0.5x pot
        3: raise 1.0x pot
        4: raise 2.0x pot
        5: all-in
    
    Args:
        action_idx: Abstract action index (0-5)
        game: PokerGame instance
        player_id: Acting player
        
    Returns:
        Action object for the engine
    """
    from engine import Action
    
    # Use cached actions for fold/check/call/allin
    if action_idx == 0:
        if 'fold' not in _ACTION_CACHE:
            _ACTION_CACHE['fold'] = Action('fold')
        return _ACTION_CACHE['fold']
    
    player = game.players[player_id]
    to_call = game.current_bet - player.bet
    
    if action_idx == 1:
        # Check or Call
        if to_call == 0:
            if 'check' not in _ACTION_CACHE:
                _ACTION_CACHE['check'] = Action('check')
            return _ACTION_CACHE['check']
        else:
            if 'call' not in _ACTION_CACHE:
                _ACTION_CACHE['call'] = Action('call')
            return _ACTION_CACHE['call']
    
    # The pot a raise is sized against includes the call that precedes it.
    #
    # This read `game.state.pot.total` -- the pot *before* calling -- which made
    # every pot-fraction raise about 20% smaller than the same abstract action
    # in `games/nolimit.py`, where the solver is trained. A pot-sized raise
    # conventionally means calling first and then betting the pot including your
    # call, which is what `NoLimitHoldem._raise_cost` does:
    #
    #     pot = sum(state.contributions) + to_call
    #
    # The divergence is measured in tests/test_betting_equivalence.py: the two
    # implementations agreed to the chip on every line without a pot-fraction
    # raise and differed on every line with one.
    pot = game.state.pot.total + to_call
    
    if action_idx == 2:
        # Raise 0.5x pot
        raise_amount = max(game.state.big_blind, int(pot * 0.5))
        if player.stack <= to_call + raise_amount:
            if 'allin' not in _ACTION_CACHE:
                _ACTION_CACHE['allin'] = Action('all-in')
            return _ACTION_CACHE['allin']
        return Action('raise', amount=raise_amount)
    
    elif action_idx == 3:
        # Raise 1x pot
        raise_amount = max(game.state.big_blind, pot)
        if player.stack <= to_call + raise_amount:
            if 'allin' not in _ACTION_CACHE:
                _ACTION_CACHE['allin'] = Action('all-in')
            return _ACTION_CACHE['allin']
        return Action('raise', amount=raise_amount)
    
    elif action_idx == 4:
        # Raise 2x pot
        raise_amount = max(game.state.big_blind, pot * 2)
        if player.stack <= to_call + raise_amount:
            if 'allin' not in _ACTION_CACHE:
                _ACTION_CACHE['allin'] = Action('all-in')
            return _ACTION_CACHE['allin']
        return Action('raise', amount=raise_amount)
    
    elif action_idx == 5:
        # All-in
        if 'allin' not in _ACTION_CACHE:
            _ACTION_CACHE['allin'] = Action('all-in')
        return _ACTION_CACHE['allin']
    
    else:
        # Default to fold for invalid actions
        return Action('fold')


class IllegalActionTally:
    """
    Counts actions the engine rejected, so they stop being invisible.

    Both hand loops respond to a rejected action by folding for the player and
    carrying on.  That is a reasonable way to survive a bad action, but it used
    to be completely silent: an agent emitting illegal actions on every hand
    scored as merely *weak* rather than broken, which is exactly how defects
    survive dozens of training runs unnoticed.

    A warning is raised the first time so a run cannot fail quietly, and the
    counts stay available for callers that want to report them.
    """

    def __init__(self):
        self.rejected = 0        # engine refused the chosen action
        self.fold_failed = 0     # even the fallback fold was refused
        self.truncated = 0       # hand hit the max_actions safety cap

    def reset(self) -> None:
        self.rejected = 0
        self.fold_failed = 0
        self.truncated = 0

    @property
    def total(self) -> int:
        return self.rejected + self.fold_failed + self.truncated

    def __repr__(self) -> str:
        return (f"IllegalActionTally(rejected={self.rejected}, "
                f"fold_failed={self.fold_failed}, truncated={self.truncated})")


# Process-wide tally. Read it after a run; call .reset() to zero it.
ILLEGAL_ACTIONS = IllegalActionTally()
_WARNED_ILLEGAL = False


def apply_action_or_fold(game, seat: int, action, tally: IllegalActionTally = None) -> bool:
    """
    Apply `action` for `seat`, falling back to a fold if the engine refuses it.

    Returns True if the hand can continue, False if even folding failed and the
    hand must be abandoned.
    """
    global _WARNED_ILLEGAL
    tally = tally if tally is not None else ILLEGAL_ACTIONS

    try:
        game.apply_action(seat, action)
        return True
    except Exception as exc:
        tally.rejected += 1
        if not _WARNED_ILLEGAL:
            _WARNED_ILLEGAL = True
            warnings.warn(
                f"engine rejected an agent action ({action!r}: {exc}); folding "
                f"instead. Further occurrences are counted in "
                f"training.fitness.ILLEGAL_ACTIONS rather than warned about.",
                RuntimeWarning, stacklevel=2,
            )
        try:
            from engine import Action
            game.apply_action(seat, Action('fold'))
            return True
        except Exception:
            tally.fold_failed += 1
            return False


def hand_start_stacks(game) -> List[int]:
    """
    Stacks as they were at the START of the hand, before blinds and antes left
    them.

    Callers receive a game whose blinds are already posted, so reading
    `player.stack` alone understates each player's starting chips by whatever
    they have already put in.  Measuring chip deltas against that understated
    baseline credits a winner with their own blind back as profit — a
    systematic, position-dependent bias, since the button does not rotate
    between hands here.  `total_contributed` at this point is exactly the
    blind/ante already posted, so adding it back recovers the true baseline.
    """
    return [p.stack + p.total_contributed for p in game.players]


def finish_hand(game) -> None:
    """
    Award the pot, whichever way the hand ended.

    `resolve_showdown()` handles a lone surviving player as well as a real
    showdown, so it must be called on fold-outs too.  Guarding it with
    `betting_round == 'showdown'` silently destroyed the pot on every hand won
    by folding — the majority of hands — and scored all players as having lost
    their contributions.
    """
    if game.state.pot.total > 0:
        game.resolve_showdown()


def chip_deltas(game, start_stacks: List[int]) -> Dict[int, int]:
    """
    Per-seat chip change for a completed hand, keyed by SEAT index.

    Poker is zero-sum: once the pot is awarded the deltas must sum to zero.
    Violating that means chips were destroyed (pot never awarded) or created
    (double payout), so it is asserted rather than trusted.
    """
    changes = {i: p.stack - start_stacks[i] for i, p in enumerate(game.players)}
    total = sum(changes.values())
    assert total == 0, (
        f"chips not conserved: net {total:+d} over the hand "
        f"(pot={game.state.pot.total}, round={game.state.betting_round})"
    )
    return changes


def play_hands_batched(networks_list: List[List[PolicyNetwork]],
                       games: List,
                       rng: np.random.Generator,
                       temperature: float = 1.0) -> List[Dict[int, int]]:
    """
    Play multiple hands in parallel with batched neural network inference.
    Provides 1.3-1.5× speedup by processing multiple decisions simultaneously.
    
    Args:
        networks_list: List of network lists (one list per game)
        games: List of PokerGame instances
        rng: Random number generator
        temperature: Action sampling temperature
        
    Returns:
        List of chip change dicts (one per game)
    """
    from engine.features import FeatureCache
    
    batch_size = len(games)
    stacks_before = [hand_start_stacks(game) for game in games]
    max_actions = 200
    
    # Track state for each game
    game_states = []
    for idx, game in enumerate(games):
        feature_caches = [FeatureCache(game, i) for i in range(len(game.players))]
        game_states.append({
            'game': game,
            'networks': networks_list[idx],
            'feature_caches': feature_caches,
            'action_count': 0,
            'finished': False
        })
    
    # Play all games step by step, batching decisions across games
    while any(not gs['finished'] for gs in game_states):
        # Collect current decisions from all active games
        batch_features = []
        batch_masks = []
        batch_info = []  # (game_state_idx, current_player, network)
        
        for gs_idx, gs in enumerate(game_states):
            if gs['finished']:
                continue
                
            game = gs['game']
            if game.is_hand_over():
                gs['finished'] = True
                continue
            if gs['action_count'] >= max_actions:
                # A hand needing 200 actions means the betting loop is not
                # terminating. Truncating silently turned that into a slightly
                # odd chip delta instead of a visible failure.
                ILLEGAL_ACTIONS.truncated += 1
                gs['finished'] = True
                continue
            
            current = game.state.current_player
            if current is None:
                gs['finished'] = True
                continue
            
            player = game.players[current]
            if player.has_folded or player.is_all_in:
                gs['finished'] = True
                continue
            
            # Collect features and mask for this decision
            features = gs['feature_caches'][current].get_features(game)
            mask = create_action_mask(game, current)
            
            batch_features.append(features)
            batch_masks.append(mask)
            batch_info.append((gs_idx, current, gs['networks'][current]))
        
        # If no active decisions, we're done
        if len(batch_features) == 0:
            break
        
        # Batch inference: process all decisions at once
        if len(batch_features) == 1:
            # Single decision - use regular select_action
            action_idx = batch_info[0][2].select_action(
                batch_features[0], batch_masks[0], rng, temperature
            )
            action_indices = [action_idx]
        else:
            # Multiple decisions - use batched processing
            features_array = np.array(batch_features, dtype=np.float32)
            masks_array = np.array(batch_masks, dtype=np.float32)
            
            # Use first network's select_action_batch (all networks same architecture)
            action_indices = batch_info[0][2].select_action_batch(
                features_array, masks_array, rng, temperature
            )
        
        # Apply actions to respective games
        for i, (gs_idx, current, network) in enumerate(batch_info):
            gs = game_states[gs_idx]
            game = gs['game']
            action_idx = action_indices[i]
            
            # Convert to engine action and apply it
            action = abstract_action_to_engine_action(action_idx, game, current)
            if not apply_action_or_fold(game, current, action):
                gs['finished'] = True

            gs['action_count'] += 1
    
    # Award pots and calculate results
    results = []
    for idx, gs in enumerate(game_states):
        game = gs['game']
        finish_hand(game)
        results.append(chip_deltas(game, stacks_before[idx]))

    return results


def play_hand(networks: List[PolicyNetwork], game,
              rng: np.random.Generator,
              temperature: float = 1.0) -> Dict[int, int]:
    """
    Play a single hand and return chip changes.
    
    Args:
        networks: List of policy networks (one per player)
        game: PokerGame instance (already initialized)
        rng: Random number generator
        temperature: Action sampling temperature
        
    Returns:
        Dict mapping player_id to chip change

    Note:
        This is a batch of one.  It used to carry its own copy of the hand loop,
        which then drifted from the batched version — the two disagreed about
        which feature layout to use, and that is how agents ended up trained on
        one observation and evaluated on another.  There is now a single loop.
    """
    return play_hands_batched([networks], [game], rng, temperature)[0]


def evaluate_matchup(genome_weights: np.ndarray,
                    opponent_weights: List[np.ndarray],
                    network_config: NetworkConfig,
                    fitness_config: FitnessConfig,
                    seed: int,
                    hand_seeds: Optional[List[int]] = None,
                    num_players_override: Optional[int] = None) -> Tuple[int, int]:
    """
    Evaluate one matchup (hero vs opponents over many hands).

    Args:
        genome_weights: Hero genome weights
        opponent_weights: List of opponent weights
        network_config: Network config
        fitness_config: Evaluation config
        seed: Random seed for reproducibility
        hand_seeds: Fixed hand seeds (None = random, used for training)
        num_players_override: If set, overrides fitness_config.num_players for
            this matchup.  Used by mixed-format training to interleave 2-player
            HeadsUp matchups and 6-player MultiTable matchups in a single
            fitness evaluation, producing format-agnostic agents.

    Returns:
        Tuple of (hero_delta_in_big_blinds, hands_played).  The delta is the
        HERO's own net result — not seat 0's — accumulated in big blinds so a
        varying blind level cannot corrupt the normalisation.
    """
    from engine import PokerGame

    # Use PCG64 for faster random number generation
    rng = Generator(PCG64(seed))
    # Create networks
    hero_net = PolicyNetwork(network_config)
    hero_net.set_weights_from_genome(genome_weights)
    opponent_nets = []
    for w in opponent_weights:
        net = PolicyNetwork(network_config)
        net.set_weights_from_genome(w)
        opponent_nets.append(net)
    # Respect per-matchup player count (for mixed-format training)
    num_players = num_players_override if num_players_override is not None else fitness_config.num_players
    # Create network list and randomize seat positions
    networks = [hero_net] + opponent_nets[:num_players - 1]
    while len(networks) < num_players:
        networks.append(opponent_nets[rng.integers(len(opponent_nets))])

    # For each hand, shuffle seat positions
    # Prepare hand seeds
    num_hands = fitness_config.hands_per_matchup
    if hand_seeds is None:
        # Use random hands for training
        hand_seeds = [int(rng.integers(0, 2**31)) for _ in range(num_hands)]
    # Per-hand conditions.  Off by default: every hand uses the declared
    # starting_stack / blinds / ante.  Randomisation is opt-in via the config.
    cfg = fitness_config
    randomise = cfg.randomise_conditions
    if randomise:
        def _bound(value, fallback):
            return int(fallback if value is None else value)
        stack_min = _bound(cfg.stack_min, cfg.starting_stack // 2)
        stack_max = _bound(cfg.stack_max, cfg.starting_stack * 3 // 2)
        sb_min    = _bound(cfg.sb_min, cfg.small_blind)
        sb_max    = _bound(cfg.sb_max, cfg.small_blind * 2)
        bb_min    = _bound(cfg.bb_min, cfg.big_blind)
        bb_max    = _bound(cfg.bb_max, cfg.big_blind * 2)
        ante_min  = _bound(cfg.ante_min, cfg.ante)
        ante_max  = _bound(cfg.ante_max, max(1, cfg.ante * 2))

    def new_game(hand_seed):
        if randomise:
            stacks = [int(rng.integers(stack_min, stack_max + 1)) for _ in range(num_players)]
            sb   = int(rng.integers(sb_min, sb_max + 1))
            bb   = int(rng.integers(bb_min, bb_max + 1))
            ante = int(rng.integers(ante_min, ante_max + 1)) if ante_max > 0 else 0
        else:
            stacks = [cfg.starting_stack] * num_players
            sb, bb, ante = cfg.small_blind, cfg.big_blind, cfg.ante
        # Use game pool for memory reuse
        return _GAME_POOL.acquire(
            player_stacks=stacks,
            small_blind=sb,
            big_blind=bb,
            ante=ante,
            seed=hand_seed
        ), bb

    # Accumulated in BIG BLINDS, not chips: with randomisation enabled the blind
    # varies per hand, so dividing a chip total by a single nominal blind at the
    # end (as this used to) normalises most hands by the wrong denominator.
    total_delta_bb = 0.0
    hands_played = 0
    
    # Pair every deal with its mirror before play begins: index 2k and 2k+1
    # share a hand seed and hold reversed seat orders. Half as many distinct
    # deals for the same hand budget, each seen from both sides.
    schedule = []
    for pair_start in range(0, num_hands, 2):
        order = list(range(num_players))
        rng.shuffle(order)
        schedule.append((hand_seeds[pair_start], order))
        if pair_start + 1 < num_hands:
            schedule.append((hand_seeds[pair_start], order[::-1]))

    # Process hands in batches for better performance
    batch_size = 8  # Process 8 hands simultaneously
    
    for batch_start in range(0, num_hands, batch_size):
        batch_end = min(batch_start + batch_size, num_hands)
        batch_hands = batch_end - batch_start
        
        # Prepare batch of games
        games_batch = []
        networks_batch = []
        hero_seats_batch = []
        bb_batch = []

        for hand_idx in range(batch_start, batch_end):
            # Duplicate play: consecutive hands share a deal and reverse the
            # seating, so the hero holds both sides of the same cards. Rotating
            # seats alone cancels position and leaves card luck untouched, which
            # is the larger of the two — a genome's score moved by 136 BB/100
            # between two measurements of the same weights before this.
            seed, seat_order = schedule[hand_idx]

            # Shuffle networks to match seat order: seat j plays networks[seat_order[j]].
            shuffled_networks = [networks[i] for i in seat_order]
            networks_batch.append(shuffled_networks)

            # The hero is networks[0], so it occupies the seat holding index 0.
            # Chip deltas come back keyed by SEAT, so this is the only seat whose
            # result belongs to the hero.  (Reading seat 0 unconditionally, as
            # this used to, scored an opponent on ~(n-1)/n of all hands.)
            hero_seats_batch.append(seat_order.index(0))

            # Create game
            game, bb = new_game(seed)
            games_batch.append(game)
            bb_batch.append(bb)

        # Play batch of hands with batched inference
        if batch_hands == 1:
            # Single hand - use regular play_hand
            changes_batch = [play_hand(networks_batch[0], games_batch[0], rng, fitness_config.temperature)]
        else:
            # Multiple hands - use batched processing
            changes_batch = play_hands_batched(networks_batch, games_batch, rng, fitness_config.temperature)

        # Accumulate the hero's own result, in big blinds
        for changes, hero_seat, bb in zip(changes_batch, hero_seats_batch, bb_batch):
            total_delta_bb += changes[hero_seat] / bb
            hands_played += 1

        # Return games to pool
        for game in games_batch:
            _GAME_POOL.release(game)

    return total_delta_bb, hands_played


def _worker_evaluate_genome_with_hof(args: Tuple) -> Dict:
    """
    Worker function for parallel evaluation with HOF tracking.
    
    Args:
        args: Tuple of (genome_id, genome_weights, opponent_weights_list, 
                       network_config, fitness_config, base_seed, hof_info)
                       
    Returns:
        Dict with evaluation results including HOF usage
    """
    if len(args) == 7:
        # New format with HOF tracking
        genome_id, genome_weights, opponent_weights_list, network_config, fitness_config, base_seed, hof_info = args
    else:
        # Old format for backward compatibility  
        genome_id, genome_weights, opponent_weights_list, network_config, fitness_config, base_seed = args
        hof_info = {}
    
    # Call the original worker function
    original_args = (genome_id, genome_weights, opponent_weights_list, network_config, fitness_config, base_seed, hof_info)
    result = _worker_evaluate_genome(original_args)

    # Convert EvalResult to dict and add HOF tracking
    result_dict = {
        'genome_id': genome_id,
        'fitness': result.fitness,
        'num_hands': result.total_hands,       # EvalResult field is total_hands
        'win_rate': result.fitness,             # BB/100 is the closest proxy
        'num_matchups': result.matchups_played, # EvalResult field is matchups_played
        'hu_fitness': result.hu_fitness,
        'mt_fitness': result.mt_fitness,
    }
    
    # Add HOF tracking information
    if hof_info and 'hof_ids_used' in hof_info:
        result_dict['hof_ids_used'] = hof_info['hof_ids_used']
        result_dict['hof_count_per_matchup'] = hof_info['hof_count_per_matchup']
        result_dict['total_hof_opponents'] = len(hof_info['hof_ids_used'])
    else:
        result_dict['hof_ids_used'] = []
        result_dict['hof_count_per_matchup'] = []
        result_dict['total_hof_opponents'] = 0
    
    return result_dict


def _worker_evaluate_genome(args: Tuple) -> EvalResult:
    """
    Worker function for parallel evaluation.

    Args:
        args: Tuple of (genome_id, genome_weights, opponent_weights_list,
                       network_config, fitness_config, base_seed, hof_info
                       [, player_counts])
               ``player_counts`` is an optional list[int] — one entry per
               matchup — specifying the table size (2 = HeadsUp, 6 = 6-max).
               Omitting it falls back to fitness_config.num_players for all
               matchups (backward-compatible).

    Returns:
        EvalResult for this genome
    """
    if len(args) == 8:
        (genome_id, genome_weights, opponent_weights_list,
         network_config, fitness_config, base_seed, hof_info, player_counts) = args
    else:
        (genome_id, genome_weights, opponent_weights_list,
         network_config, fitness_config, base_seed, hof_info) = args
        player_counts = None

    total_delta = 0
    total_hands = 0
    mt_delta = 0
    mt_hands = 0
    hu_delta = 0
    hu_hands = 0

    for matchup_idx, opponents in enumerate(opponent_weights_list):
        seed = base_seed + genome_id * 1000 + matchup_idx

        # Determine per-matchup player count and hands
        if player_counts is not None and matchup_idx < len(player_counts):
            n_players = player_counts[matchup_idx]
        else:
            n_players = fitness_config.num_players

        is_hu = (n_players == 2)
        hands = (fitness_config.hu_hands_per_matchup
                 if is_hu and fitness_config.hu_hands_per_matchup is not None
                 else fitness_config.hands_per_matchup)

        # Use pre-generated hand seeds of the desired length to avoid mutating the
        # shared config object (important for sequential / num_workers=1 mode).
        import dataclasses
        matchup_cfg = dataclasses.replace(fitness_config, hands_per_matchup=hands)

        delta, h_played = evaluate_matchup(
            genome_weights, opponents,
            network_config, matchup_cfg, seed,
            hand_seeds=None,
            num_players_override=n_players,
        )

        total_delta += delta
        total_hands += h_played
        if is_hu:
            hu_delta += delta
            hu_hands += h_played
        else:
            mt_delta += delta
            mt_hands += h_played

    # Final fitness = weighted BB/100 across all matchups
    # (HU and MT are naturally weighted by their share of total hands)
    # evaluate_matchup already returns big blinds, so no division by blind here.
    bb_per_100 = total_delta * (100 / max(1, total_hands))
    hu_bb100 = hu_delta * (100 / max(1, hu_hands)) if hu_hands > 0 else 0.0
    mt_bb100 = mt_delta * (100 / max(1, mt_hands)) if mt_hands > 0 else 0.0

    return EvalResult(
        genome_id=genome_id,
        fitness=bb_per_100,
        total_hands=total_hands,
        total_bb_delta=total_delta,
        matchups_played=len(opponent_weights_list),
        hu_fitness=hu_bb100,
        mt_fitness=mt_bb100,
    )

def evaluate_fixed_hands(genome_weights: np.ndarray,
                        opponent_weights: List[np.ndarray],
                        network_config: NetworkConfig,
                        fitness_config: FitnessConfig,
                        eval_hand_seeds: List[int],
                        seed: int) -> Tuple[float, int]:
    """
    Evaluate a genome on a fixed set of hands for fair comparison.

    Returns:
        (total_delta_in_big_blinds, hands_played)
    """
    return evaluate_matchup(
        genome_weights, opponent_weights,
        network_config, fitness_config, seed,
        hand_seeds=eval_hand_seeds
    )


class FitnessEvaluator:
    """
    Evaluates genome fitness through self-play.
    
    Each genome plays as "hero" against various opponent configurations.
    Opponents are sampled from:
        1. Other genomes in current population
        2. Hall of fame (historical good agents)
        3. Random agents (for diversity)
    
    Fitness = average BB/100 across all matchups.
    """
    
    def __init__(self, factory: GenomeFactory,
                 config: FitnessConfig,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize evaluator.
        
        Args:
            factory: GenomeFactory for creating networks
            config: Fitness evaluation config
            rng: Random number generator
        """
        self.factory = factory
        self.config = config
        self.rng = rng or np.random.default_rng()
    
    def create_opponent_groups(self, genomes: List[Genome],
                               hall_of_fame: Optional[List[Genome]] = None,
                               hof_max_size: int = 20) -> Tuple[List[List[np.ndarray]], List[List[int]], List[int]]:
        """
        Create opponent weight groups for evaluation.

        Supports mixed-format training: when ``heads_up_fraction > 0`` in the
        FitnessConfig, some matchups are designated as 2-player HeadsUp and the
        rest as num_players-player MultiTable.  The caller must honour the
        returned ``player_counts`` list when invoking evaluate_matchup so the
        correct table size is used.

        Args:
            genomes: Current population
            hall_of_fame: Optional historical best agents
            hof_max_size: Max number of diverse HoF agents to retain

        Returns:
            Tuple of:
              - ``groups``: opponent weight lists, one per matchup
              - ``hof_tracking``: list of HoF genome IDs used per matchup
              - ``player_counts``: list of int (2 for HU, num_players for MT)
        """
        num_matchups = self.config.matchups_per_agent

        # Determine how many matchups are HeadsUp vs MultiTable
        num_hu = self.config.num_hu_matchups  # uses heads_up_fraction
        num_mt = self.config.num_mt_matchups
        mt_num_players = self.config.num_players  # typically 6

        # Build the ordered matchup format list (HU first, then MT — shuffled later)
        matchup_formats = [2] * num_hu + [mt_num_players] * num_mt
        self.rng.shuffle(matchup_formats)  # randomise order across generations

        # Collect all potential opponent weights
        all_weights = [g.weights for g in genomes]

        hof_weights = []
        hof_genome_ids = []
        if hall_of_fame:
            # Keep only the most diverse and highest-performing agents
            hof_sorted = sorted(hall_of_fame, key=lambda g: g.fitness if g.fitness is not None else -1, reverse=True)
            hof_selected = []
            for g in hof_sorted:
                if not hof_selected:
                    hof_selected.append(g)
                else:
                    dists = [np.linalg.norm(g.weights - h.weights) for h in hof_selected]
                    if min(dists) > 0.1:  # Diversity threshold
                        hof_selected.append(g)
                if len(hof_selected) >= hof_max_size:
                    break
            hof_weights = [g.weights for g in hof_selected]
            hof_genome_ids = [g.genome_id for g in hof_selected]

        groups = []
        hof_tracking = []
        player_counts = []

        for fmt_players in matchup_formats:
            # For HU matchups, only 1 opponent needed; for MT, num_players-1
            num_opponents = fmt_players - 1
            group = []
            group_hof_ids = []

            for _ in range(num_opponents):
                r = self.rng.random()
                if r < 0.2 and hof_weights:
                    # 20%: from Hall of Fame
                    idx = self.rng.integers(len(hof_weights))
                    group.append(hof_weights[idx])
                    group_hof_ids.append(hof_genome_ids[idx])
                elif r < 0.3:
                    # 10%: random noise agent (maintains diversity)
                    random_weights = self.rng.standard_normal(
                        self.factory.genome_size
                    ).astype(np.float32) * 0.1
                    group.append(random_weights)
                else:
                    # 70%: from current population
                    idx = self.rng.integers(len(all_weights))
                    group.append(all_weights[idx])

            groups.append(group)
            hof_tracking.append(group_hof_ids)
            player_counts.append(fmt_players)

        return groups, hof_tracking, player_counts
    
    def evaluate_single(self, genome: Genome,
                       opponents: List[Genome],
                       num_hands: Optional[int] = None) -> float:
        """
        Evaluate a single genome against opponents.
        
        Args:
            genome: Genome to evaluate
            opponents: Opponent genomes
            num_hands: Override hands per evaluation
            
        Returns:
            BB/100 fitness score
        """
        if num_hands is not None:
            old_hands = self.config.hands_per_matchup
            self.config.hands_per_matchup = num_hands // self.config.matchups_per_agent
        
        opponent_groups, _, pc = self.create_opponent_groups(opponents)

        total_delta = 0
        total_hands = 0

        for matchup_idx, opponent_weights in enumerate(opponent_groups):
            seed = self.rng.integers(0, 2**31)
            n_players = pc[matchup_idx] if matchup_idx < len(pc) else self.config.num_players
            is_hu = (n_players == 2)
            hands_override = (self.config.hu_hands_per_matchup
                              if is_hu and self.config.hu_hands_per_matchup is not None
                              else self.config.hands_per_matchup)
            orig = self.config.hands_per_matchup
            self.config.hands_per_matchup = hands_override
            delta, hands = evaluate_matchup(
                genome.weights, opponent_weights,
                self.factory.network_config,
                self.config, seed,
                num_players_override=n_players,
            )
            self.config.hands_per_matchup = orig
            total_delta += delta
            total_hands += hands
        
        if num_hands is not None:
            self.config.hands_per_matchup = old_hands

        # total_delta is already in big blinds
        return total_delta * (100 / max(1, total_hands))
    
    def evaluate_population(self, genomes: List[Genome],
                           hall_of_fame: Optional[List[Genome]] = None,
                           parallel: bool = False,
                           track_hof_usage: bool = True) -> Dict[int, EvalResult]:
        """
        Evaluate fitness for all genomes in population.
        
        Args:
            genomes: List of genomes to evaluate
            hall_of_fame: Optional HoF for opponent diversity
            parallel: Use multiprocessing
            track_hof_usage: Whether to track which HOF members were used as opponents
            
        Returns:
            Dict mapping genome_id to EvalResult
        """
        # Create opponent groups with optional HOF tracking
        # create_opponent_groups now returns (groups, hof_tracking, player_counts)
        if track_hof_usage:
            opponent_groups, hof_tracking, player_counts = self.create_opponent_groups(genomes, hall_of_fame)
        else:
            opponent_groups, hof_tracking, player_counts = self.create_opponent_groups(genomes, hall_of_fame)
        
        # Prepare evaluation arguments
        base_seed = self.rng.integers(0, 2**31)
        args_list = []
        for i, g in enumerate(genomes):
            # Include HOF tracking info for each genome
            hof_info = {
                'hof_ids_used': [hof_id for group_hof in hof_tracking for hof_id in group_hof],
                'hof_count_per_matchup': [len(group_hof) for group_hof in hof_tracking]
            } if track_hof_usage else {}
            
            args_list.append((
                g.genome_id, g.weights, opponent_groups,
                self.factory.network_config, self.config, base_seed, hof_info,
                player_counts,  # per-matchup table sizes for mixed-format training
            ))
        
        if parallel and self.config.num_workers > 1:
            # Parallel evaluation
            with mp.Pool(self.config.num_workers) as pool:
                results = pool.map(_worker_evaluate_genome, args_list)
        else:
            # Sequential evaluation
            results = [_worker_evaluate_genome(args) for args in args_list]
        
        # Update genome fitness and build result dict
        result_dict = {}
        for result in results:
            result_dict[result.genome_id] = result
            
            # Update genome fitness
            for g in genomes:
                if g.genome_id == result.genome_id:
                    g.fitness = result.fitness
                    break
        
        return result_dict
    
    def evaluate_against_baseline(self, genome: Genome,
                                  num_hands: int = 5000) -> float:
        """
        Evaluate genome against random baseline agents.
        
        Args:
            genome: Genome to evaluate
            num_hands: Total hands to play
            
        Returns:
            BB/100 against random opponents
        """
        # Create random opponent genomes
        random_opponents = []
        for _ in range(self.config.num_players - 1):
            random_weights = self.rng.standard_normal(
                self.factory.genome_size
            ).astype(np.float32) * 0.1
            random_opponents.append(random_weights)
        
        # Single matchup with all hands
        seed = self.rng.integers(0, 2**31)
        
        # Temporarily increase hands
        old_hands = self.config.hands_per_matchup
        self.config.hands_per_matchup = num_hands
        
        delta, hands = evaluate_matchup(
            genome.weights, random_opponents,
            self.factory.network_config,
            self.config, seed
        )
        
        self.config.hands_per_matchup = old_hands

        # delta is already in big blinds
        return delta * (100 / max(1, hands))
