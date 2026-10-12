"""
Feature extraction for AI training.
Provides normalized state vectors and hand strength calculations.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .cards import Card, RANKS, SUITS
from .hand_eval import evaluate_hand, RANK_ORDER
from .hand_eval_fast import evaluate_hand_fast

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
def compute_pot_odds_jit(to_call: float, pot_size: float) -> float:
    """JIT-compiled pot odds calculation."""
    if pot_size + to_call <= 0:
        return 0.0
    return to_call / (pot_size + to_call)


@jit(nopython=True, cache=True, fastmath=True)
def compute_stack_to_pot_jit(stack: float, pot_size: float) -> float:
    """JIT-compiled stack-to-pot ratio calculation."""
    if pot_size <= 0:
        return 10.0
    return min(stack / pot_size, 20.0) / 20.0  # Normalize to 0-1


# NOTE: a second, JIT-assembled 17-dim layout used to live here
# (`build_feature_vector_jit`, ordered pot_odds/spr/position-one-hot/...).  It
# was a *different* ordering from FeatureCache, which meant agents were trained
# on one layout and evaluated on the other.  It has been removed so that
# FeatureCache remains the single definition of the observation.


# Precomputed lookup tables for common calculations
# Pot odds lookup: POT_ODDS_TABLE[to_call][pot] = to_call/(pot+to_call)
# Using 5-chip granularity for indices, covering 0-5000 chips
#
# Built vectorised rather than with a Python double loop over 1,002,001 cells.
# This runs at import, and every multiprocessing worker re-imports the module,
# so the old loop cost ~0.3 s per process — paid again for each worker in the
# evaluation pool. The vectorised form produces a bit-identical table in ~8 ms.
_TO_CALL = (np.arange(1001, dtype=np.float32) * 5)[:, None]
_POT = (np.arange(1001, dtype=np.float32) * 5)[None, :]
_DENOM = _TO_CALL + _POT
POT_ODDS_TABLE = np.divide(
    np.broadcast_to(_TO_CALL, _DENOM.shape), _DENOM,
    out=np.zeros((1001, 1001), dtype=np.float32),
    where=_DENOM > 0,
)
del _TO_CALL, _POT, _DENOM

# Precomputed hand strength for all 169 starting hands (13 ranks × 13 ranks × suited/offsuit)
PREFLOP_STRENGTH_CACHE = {}
def _init_preflop_cache():
    """
    Initialize precomputed hand strength for all starting hands.

    Pocket pairs are ALWAYS offsuit (two cards of one rank cannot share a
    suit), so the offsuit key must be stored for r1 == r2 as well.  Skipping it
    left every pair missing from the table, and the lookup below fell through
    to its 0.5 default — meaning aces, kings and deuces all presented to the
    network as an exactly average hand.
    """
    for r1 in RANKS:
        for r2 in RANKS:
            if r1 != r2:
                PREFLOP_STRENGTH_CACHE[(r1, r2, True)] = \
                    preflop_hand_strength([Card(r1, 'h'), Card(r2, 'h')])
            # Offsuit (and the only real form for a pair)
            PREFLOP_STRENGTH_CACHE[(r1, r2, False)] = \
                preflop_hand_strength([Card(r1, 'h'), Card(r2, 'd')])

def get_preflop_strength_fast(hole_cards: List[Card]) -> float:
    """
    Fast lookup of precomputed preflop hand strength.

    Every real two-card holding is present in the table, so a miss means the
    caller passed something malformed rather than an unusual hand.
    """
    if len(hole_cards) != 2:
        return 0.5
    c1, c2 = hole_cards
    suited = (c1.suit == c2.suit)
    strength = PREFLOP_STRENGTH_CACHE.get((c1.rank, c2.rank, suited))
    if strength is None:                      # malformed input, not a real hand
        return 0.5
    return strength

# Chen formula for preflop hand strength
def chen_formula(hole_cards: List[Card]) -> float:
    """
    Calculate Chen formula score for preflop hand strength.
    Returns a score from -1 to 20 (higher is better).
    """
    if len(hole_cards) != 2:
        return 0.0
    
    c1, c2 = hole_cards
    r1, r2 = RANK_ORDER[c1.rank], RANK_ORDER[c2.rank]
    
    # Ensure r1 >= r2 (high card first)
    if r2 > r1:
        r1, r2 = r2, r1
        c1, c2 = c2, c1
    
    # Base score from high card
    high_card_scores = {
        12: 10,  # A
        11: 8,   # K
        10: 7,   # Q
        9: 6,    # J
        8: 5, 7: 4.5, 6: 4, 5: 3.5, 4: 3, 3: 2.5, 2: 2, 1: 1.5, 0: 1
    }
    score = high_card_scores.get(r1, r1 / 2 + 1)
    
    # Pair bonus
    if r1 == r2:
        score = max(5, score * 2)
    
    # Suited bonus
    if c1.suit == c2.suit:
        score += 2
    
    # Gap penalty
    gap = r1 - r2 - 1
    if gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap == 3:
        score -= 4
    elif gap >= 4:
        score -= 5
    
    # Straight potential bonus (both cards <= Q and gap <= 2)
    if r1 <= 10 and gap <= 2 and r1 != r2:
        score += 1
    
    return score


def preflop_hand_strength(hole_cards: List[Card]) -> float:
    """
    Returns normalized preflop hand strength from 0.0 to 1.0.
    Based on Chen formula, normalized.
    """
    chen = chen_formula(hole_cards)
    # Chen scores range roughly from -1 to 20
    # Normalize to 0-1
    return max(0.0, min(1.0, (chen + 1) / 21))

# Initialize lookup cache on module load
_init_preflop_cache()


def hand_strength_vs_random(hole_cards: List[Card], community_cards: List[Card], 
                            num_simulations: int = 500) -> float:
    """
    Monte Carlo simulation of hand strength against random opponent hands.
    Returns win probability from 0.0 to 1.0.
    """
    from .cards import Deck
    import random
    
    if len(hole_cards) != 2:
        return 0.5
    
    wins = 0
    ties = 0
    
    # Cards that are already used
    used = set((c.rank, c.suit) for c in hole_cards + community_cards)
    
    for _ in range(num_simulations):
        # Create remaining deck
        remaining = [Card(r, s) for r in RANKS for s in SUITS 
                     if (r, s) not in used]
        random.shuffle(remaining)
        
        # Deal opponent cards
        opp_cards = remaining[:2]
        remaining = remaining[2:]
        
        # Complete community cards if needed
        need_community = 5 - len(community_cards)
        full_community = community_cards + remaining[:need_community]
        
        # Evaluate hands
        my_hand = evaluate_hand(hole_cards + full_community)
        opp_hand = evaluate_hand(opp_cards + full_community)
        
        if my_hand > opp_hand:
            wins += 1
        elif my_hand == opp_hand:
            ties += 1
    
    return (wins + ties * 0.5) / num_simulations


def get_state_vector(game, player_id: int, cache: Optional['FeatureCache'] = None) -> np.ndarray:
    """
    Returns state features as a flat vector for neural network input.

    There is exactly ONE feature layout in this project, defined by
    `FeatureCache.get_features()` and named by `get_feature_names()`.  Every
    trained genome in `checkpoints/` and `hall_of_fame/` was fitted against it,
    so inference MUST use the same layout or the network receives permuted
    inputs.  This function therefore always delegates to that definition.

    Passing `cache` is purely a performance optimisation: the fields it caches
    (position, preflop hand strength, starting stack, button) are invariant for
    the duration of a hand, so building a transient cache here yields values
    identical to a cache built at hand start.

    Args:
        game: PokerGame instance
        player_id: Player index
        cache: Optional FeatureCache to avoid re-deriving static features

    Returns:
        Feature vector as named by get_feature_names(), normalized.
    """
    if cache is not None:
        return cache.get_features(game)

    return FeatureCache(game, player_id).get_features(game)


#: Ranks in ascending order, for the straight-draw scan.
_RANK_ORDER = "23456789TJQKA"


def draw_and_texture(hole, board):
    """
    Four binary card features the made-hand score cannot express.

    ``made_hand_strength`` scores only the hand completed so far, so a flush
    draw and the same two overcards without one produce an identical number.
    Auditing the observation on 1,500 sampled states found postflop situations
    sharing a hand-strength value whose true equity differed by as much as
    0.907 — one holding a 5% underdog, the other a 95% favourite, arriving at
    the network as the same input. No amount of training recovers information
    the input never carried.

    These four cost a suit count and a rank scan, no simulation, and were
    measured to raise the postflop variance explained in true equity from 0.475
    to 0.637. Draws are not all of what is missing — kicker quality and board
    interaction remain invisible — but they are the part available for nothing.

    Returns:
        (flush_draw, open_ended, paired_board, three_to_a_flush), each 0.0/1.0.
    """
    if not board:
        return 0.0, 0.0, 0.0, 0.0

    cards = list(hole) + list(board)
    suits = [c.suit for c in cards]
    board_suits = [c.suit for c in board]

    flush_draw = 1.0 if any(suits.count(s) == 4 for s in set(suits)) else 0.0
    three_flush = 1.0 if any(board_suits.count(s) >= 3
                             for s in set(board_suits)) else 0.0

    ranks = sorted({_RANK_ORDER.index(c.rank) for c in cards})
    open_ended = 0.0
    for i in range(len(ranks) - 3):
        window = ranks[i:i + 4]
        if window[3] - window[0] == 3:          # four consecutive ranks
            open_ended = 1.0
            break

    board_ranks = [c.rank for c in board]
    paired_board = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0

    return flush_draw, open_ended, paired_board, three_flush


def get_feature_names() -> List[str]:
    """Returns the ordered list of feature names for documentation."""
    return [
        'position', 'in_position', 'players_behind',
        'stack_normalized', 'spr', 'opponent_all_in',
        'pot_odds', 'to_call_ratio',
        'round_preflop', 'round_flop', 'round_turn', 'round_river',
        'facing_bet', 'facing_raise',
        'hand_strength',
        'flush_draw', 'open_ended', 'paired_board', 'three_flush',
    ]


def get_action_mask(game, player_id: int) -> List[int]:
    """
    Engine-level legality mask, one slot per *engine* action type.

    Returns a list of 5 values: [fold, check, call, raise, all_in]
    1 = legal, 0 = illegal

    This mirrors `PokerGame.get_legal_actions()` and is NOT the mask a policy
    network consumes — networks act over the 6 abstract actions.  Use
    `get_abstract_action_mask()` for that; do not zero-pad this list to 6.
    """
    legal_actions = game.get_legal_actions(player_id)

    action_types = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all-in': 0}
    for action in legal_actions:
        action_types[action['type']] = 1

    return [
        action_types['fold'],
        action_types['check'],
        action_types['call'],
        action_types['raise'],
        action_types['all-in'],
    ]


def get_abstract_action_mask(game, player_id: int) -> np.ndarray:
    """
    Legality mask over the 6 abstract actions a policy network chooses between:

        0: fold          1: check/call    2: raise 0.5x pot
        3: raise 1x pot  4: raise 2x pot  5: all-in

    This is the single definition of the 6-slot mask.  It is the layout every
    trained genome was fitted against, so training, evaluation, the GUI and the
    RL environment must all use it.

    Returns:
        float32 array of shape (6,); 1.0 = legal, 0.0 = illegal
    """
    player = game.players[player_id]
    to_call = game.current_bet - player.bet

    mask = np.zeros(6, dtype=np.float32)
    mask[0] = 1.0  # fold is always available
    mask[1] = 1.0  # check/call is always available (a short call goes all-in)

    # Raise sizings are available only with chips left beyond the call amount.
    if player.stack > to_call and (player.stack - to_call) >= game.state.big_blind:
        mask[2] = 1.0
        mask[3] = 1.0
        mask[4] = 1.0

    if player.stack > 0:
        mask[5] = 1.0

    return mask


def get_raise_sizing_info(game, player_id: int) -> Dict[str, float]:
    """
    Returns normalized raise sizing information for AI decision making.
    
    Returns:
        min_raise_ratio: Minimum raise as ratio of pot
        max_raise_ratio: Maximum raise as ratio of pot (all-in)
        can_raise: Whether raising is possible
    """
    player = game.players[player_id]
    legal_actions = game.get_legal_actions(player_id)
    
    pot = max(1, game.state.pot.total)
    
    raise_info = None
    for action in legal_actions:
        if action['type'] == 'raise':
            raise_info = action
            break
    
    if raise_info is None:
        return {
            'min_raise_ratio': 0.0,
            'max_raise_ratio': 0.0,
            'can_raise': 0.0,
        }
    
    min_raise = raise_info.get('min', game.state.big_blind)
    max_raise = raise_info.get('max', player.stack)
    
    return {
        'min_raise_ratio': min(2.0, min_raise / pot),  # Cap at 2x pot
        'max_raise_ratio': min(10.0, max_raise / pot),  # Cap at 10x pot
        'can_raise': 1.0,
    }


# When True, the hand_strength feature reflects the player's best made hand
# using the community cards.  When False it is the preflop Chen score of the
# hole cards alone, which is what this project used to feed the network on
# every street — meaning agents played the flop, turn and river unable to tell
# a set from a busted draw, because the value depended only on the two hole
# cards and so was identical for every possible board.
#
# Set to False to reproduce the old observation for a like-for-like ablation.
BOARD_AWARE_STRENGTH = True


# Approximate probability that each made-hand category beats a random hand.
# These are heuristic anchors, not computed equities — their job is to put the
# postflop feature on the SAME 0-1 scale as the preflop Chen score, so the value
# does not jump when the flop lands.  Indexing follows HAND_RANKS
# (0 = High Card ... 9 = Royal Flush), plus a sentinel so the top category can
# interpolate against something.
_CATEGORY_STRENGTH = [0.18, 0.42, 0.62, 0.75, 0.83, 0.89, 0.94, 0.98, 0.995, 1.0, 1.0]


def made_hand_strength(hole_cards: List[Card], community_cards: List[Card]) -> float:
    """
    Strength of the player's best five-card hand, normalised to 0.0-1.0.

    Category dominates — a flush always beats any straight — with the top
    tiebreaking rank separating hands inside a category, so top pair reads higher
    than bottom pair.  Values are anchored to roughly how often each category
    beats a random hand, which keeps them commensurable with the preflop score:
    without that anchoring, flopping a set scored *lower* than the same hand did
    preflop, because a raw category index and a normalised Chen score are not the
    same scale.

    Uses the existing 7-card evaluator: ~8.7 us per call, roughly a thousandth of
    the Monte-Carlo equity estimate, and unlike the preflop score it responds to
    the board.  It does not value draws — a flush draw scores as whatever it has
    made so far.
    """
    cards = list(hole_cards) + list(community_cards)
    if len(cards) < 5:
        return get_preflop_strength_fast(hole_cards)

    # The fast evaluator, verified to agree with the reference on every one of
    # 3,000 random seven-card showdowns. This runs once per street per player
    # during training, and again for every situation sampled when fitting a card
    # abstraction, so the difference is not academic.
    result = evaluate_hand_fast(cards)
    base = _CATEGORY_STRENGTH[result.hand_rank]
    ceiling = _CATEGORY_STRENGTH[result.hand_rank + 1]

    # RANK_ORDER runs 0 (deuce) to 12 (ace).
    top_rank = result.tiebreaker[0] if result.tiebreaker else 0
    within = (top_rank / 12.0) * (ceiling - base) * 0.9

    return min(1.0, base + within)


class FeatureCache:
    """
    Cache static features per hand to avoid recomputation.

    Optimization: Features that don't change during a hand are computed once,
    then only dynamic features are updated per action.

    Performance: ~1.5-2× speedup by reducing redundant calculations.
    """
    __slots__ = ['position_norm', 'hand_strength', 'starting_stack',
                 'num_players', 'player_id', 'features', 'button',
                 '_strength_board_len', '_strength_cached',
                 '_draws_board_len', '_draws_cached']
    
    def __init__(self, game, player_id: int):
        """
        Compute static features once at hand start.
        
        Args:
            game: PokerGame instance (at hand start)
            player_id: Player to cache features for
        """
        player = game.players[player_id]
        self.num_players = len(game.players)
        self.player_id = player_id
        self.button = game.state.button
        
        # Static: Position (relative to button)
        position = (player_id - self.button) % self.num_players
        self.position_norm = position / max(1, self.num_players - 1)
        
        # Static: Hand strength (preflop) - use precomputed lookup
        self.hand_strength = get_preflop_strength_fast(player.hole_cards)

        # Board-aware strength is recomputed once per street, not per decision.
        self._strength_board_len = -1
        self._strength_cached = self.hand_strength
        self._draws_board_len = -1
        self._draws_cached = (0.0, 0.0, 0.0, 0.0)

        # Static: Starting stack
        self.starting_stack = player.stack + player.total_contributed
        
        # Preallocate feature array
        self.features = np.zeros(len(get_feature_names()), dtype=np.float32)
    
    def get_features(self, game) -> np.ndarray:
        """
        Get current feature vector with cached static values.
        Only recomputes dynamic features that change per action.
        
        Args:
            game: Current game state
        
        Returns:
            Feature vector ready for neural network input
        """
        player = game.players[self.player_id]
        
        # Dynamic: Current game state
        pot = game.state.pot.total
        to_call = max(0, game.current_bet - player.bet)
        my_stack = player.stack
        bb = game.state.big_blind
        
        # Count active players
        players_in_hand = sum(1 for p in game.players if not p.has_folded)
        
        # Betting round
        rounds = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        round_idx = rounds.get(game.state.betting_round, 0)
        
        # Static features (index 0-2)
        self.features[0] = self.position_norm
        
        # In position (dynamic - changes as players fold)
        in_position = 0.0
        if game.state.betting_round != 'preflop':
            last_to_act = None
            for i in range(self.num_players - 1, -1, -1):
                idx = (self.button + 1 + i) % self.num_players
                p = game.players[idx]
                if not p.has_folded and not p.is_all_in:
                    last_to_act = idx
                    break
            in_position = 1.0 if last_to_act == self.player_id else 0.0
        self.features[1] = in_position
        
        # Players behind (dynamic)
        players_behind = 0
        if players_in_hand > 1:
            current_idx = self.player_id
            for i in range(1, self.num_players):
                idx = (current_idx + i) % self.num_players
                p = game.players[idx]
                if not p.has_folded and not p.is_all_in and idx != self.player_id:
                    players_behind += 1
        self.features[2] = players_behind / max(1, self.num_players - 1)
        
        # Stack features (index 3-6)
        self.features[3] = my_stack / self.starting_stack  # Stack normalized
        
        spr = my_stack / pot if pot > 0 else 10.0
        self.features[4] = min(1.0, spr / 20.0)  # SPR
        
        commitment = player.total_contributed / self.starting_stack if self.starting_stack > 0 else 0
        
        # Whether an OPPONENT is all-in — a real decision input, since it caps
        # what can still be won and closes further betting against them.
        # This slot used to hold the acting player's own all-in flag, which is
        # necessarily 0: a player who is all-in never gets a turn.  Measured
        # over 22,423 decision states its standard deviation was exactly 0.000,
        # so the network had one input that could never carry information.
        self.features[5] = 1.0 if any(
            p.is_all_in and not p.has_folded and p.player_id != self.player_id
            for p in game.players
        ) else 0.0

        # Pot features (index 7-8) - use precomputed pot odds table
        # Use lookup table with 5-chip granularity
        tc_idx = min(1000, to_call // 5)
        pot_idx = min(1000, pot // 5)
        pot_odds = POT_ODDS_TABLE[tc_idx, pot_idx]
        self.features[6] = pot_odds
        self.features[7] = min(1.0, to_call / (my_stack + 1))
        
        # Game state (index 9-13)
        self.features[8] = 1.0 if round_idx == 0 else 0.0   # preflop
        self.features[9] = 1.0 if round_idx == 1 else 0.0  # flop
        self.features[10] = 1.0 if round_idx == 2 else 0.0  # turn
        self.features[11] = 1.0 if round_idx == 3 else 0.0  # river
        # Normalise by max table size (6) so HU ≈ 0.33, 6-max ≈ 1.0.
        
        # Action context (index 14-15)
        self.features[12] = 1.0 if to_call > 0 else 0.0      # facing bet
        self.features[13] = 1.0 if to_call > bb else 0.0     # facing raise
        
        # Hand strength (index 14) — recomputed once per street once the board
        # exists, so the value actually responds to the community cards.
        board = game.state.community_cards
        if BOARD_AWARE_STRENGTH and board:
            if self._strength_board_len != len(board):
                self._strength_board_len = len(board)
                self._strength_cached = made_hand_strength(player.hole_cards, board)
            self.features[14] = self._strength_cached
        else:
            self.features[14] = self.hand_strength

        # Draw and texture (indices 15-18) — recomputed with the board, for the
        # same reason and at the same moment as hand strength. Preflop there is
        # no board and all four are zero.
        if board:
            if self._draws_board_len != len(board):
                self._draws_board_len = len(board)
                self._draws_cached = draw_and_texture(player.hole_cards, board)
            self.features[15:19] = self._draws_cached
        else:
            self.features[15:19] = 0.0

        return self.features
