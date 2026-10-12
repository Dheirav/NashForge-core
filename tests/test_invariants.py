"""
Invariants that must hold for any result produced by this project to mean
anything.

Each test here corresponds to a defect that silently invalidated real training
runs.  They are cheap; run them before trusting any number.

    python -m pytest tests/ -q
"""
import numpy as np
import pytest

from engine import PokerGame, Action
from engine.features import (FeatureCache, get_state_vector, get_action_mask,
                             get_abstract_action_mask, get_feature_names)
from training.config import NetworkConfig, FitnessConfig, TrainingConfig, EvolutionConfig
from training.policy_network import PolicyNetwork, create_action_mask
from training.fitness import (play_hand, play_hands_batched, evaluate_matchup,
                              abstract_action_to_engine_action)

NET_CONFIG = NetworkConfig(hidden_sizes=[64, 32])
# Derived rather than written down: the observation width is defined by
# engine.get_feature_names(), and a constant here would silently disagree with
# it the next time the feature layer changes — which is exactly what happened
# when it went from 17 slots to 19.
GENOME_SIZE = PolicyNetwork(NET_CONFIG).genome_size


def make_net(rng):
    net = PolicyNetwork(NET_CONFIG)
    net.set_weights_from_genome(rng.normal(0, 0.5, GENOME_SIZE).astype(np.float32))
    return net


def random_hand(rng, num_players=2, stacks=None):
    return PokerGame(
        player_stacks=stacks or [1000] * num_players,
        small_blind=5, big_blind=10,
        seed=int(rng.integers(0, 2 ** 31)),
    )


# ---------------------------------------------------------------------------
# Chip conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 6])
def test_engine_conserves_chips(num_players):
    """
    The engine must neither create nor destroy chips, including when the pot is
    split into side pots by unequal all-ins.
    """
    rng = np.random.default_rng(0)
    for _ in range(300):
        stacks = list(rng.integers(500, 1501, num_players))
        game = PokerGame(list(stacks), 5, 10, seed=int(rng.integers(0, 2 ** 31)))
        guard = 0
        while not game.is_hand_over() and guard < 300:
            cur = game.state.current_player
            if cur is None:
                break
            legal = game.get_legal_actions(cur)
            if not legal:
                break
            choice = legal[rng.integers(len(legal))]
            if choice['type'] == 'raise':
                target = int(rng.integers(choice['min'], choice['max'] + 1))
                act = Action('raise', max(1, target - game.state.players[cur].bet))
            else:
                act = Action(choice['type'], choice.get('amount'))
            game.apply_action(cur, act)
            guard += 1
        game.resolve_showdown()
        assert sum(p.stack for p in game.players) == sum(stacks)


@pytest.mark.parametrize("num_players", [2, 6])
def test_hand_simulation_conserves_chips(num_players):
    """
    The training-path hand loops must conserve chips.  `play_hand` and
    `play_hands_batched` assert this internally, so completing without raising
    is the check.
    """
    rng = np.random.default_rng(1)
    nets = [make_net(rng) for _ in range(num_players)]

    for _ in range(150):
        play_hand(nets, random_hand(rng, num_players), rng)

    games = [random_hand(rng, num_players) for _ in range(32)]
    play_hands_batched([nets] * 32, games, rng)


@pytest.mark.parametrize("num_players", [2, 3, 6])
def test_fold_out_pot_is_awarded(num_players):
    """
    A hand won by folding must still pay out.

    Resolution used to be guarded by `betting_round == 'showdown'`, so on every
    fold-out the pot was simply discarded: the winner was never paid and every
    player was scored as having lost their contribution.  Most hands end this
    way, so this was the dominant term in every fitness value ever recorded.

    Driven deterministically rather than hoping a network folds.
    """
    from training.fitness import finish_hand, chip_deltas, hand_start_stacks

    game = PokerGame([1000] * num_players, 5, 10, seed=99)
    start = hand_start_stacks(game)
    assert start == [1000] * num_players

    # Everyone folds until a single player remains.
    while not game.is_hand_over():
        game.apply_action(game.state.current_player, Action('fold'))

    survivors = [p for p in game.players if not p.has_folded]
    assert len(survivors) == 1
    assert game.state.betting_round != 'showdown'   # the case that was skipped

    pot = game.state.pot.total
    assert pot > 0
    finish_hand(game)

    deltas = chip_deltas(game, start)               # asserts conservation
    winner = survivors[0].player_id
    assert deltas[winner] > 0, "winner of a fold-out was not paid"
    assert sum(deltas.values()) == 0


# ---------------------------------------------------------------------------
# Fitness measures the hero
# ---------------------------------------------------------------------------

def test_fitness_scores_the_hero_not_a_fixed_seat():
    """
    evaluate_matchup shuffles seats every hand.  It must follow the hero through
    that shuffle; reading a fixed seat index scored an opponent on roughly
    (n-1)/n of all hands.
    """
    import training.fitness as F

    rng = np.random.default_rng(2)
    hero = rng.normal(0, 0.5, GENOME_SIZE).astype(np.float32)
    opps = [rng.normal(0, 0.5, GENOME_SIZE).astype(np.float32) for _ in range(5)]
    key = float(hero[0])

    seen = []
    real = F.play_hands_batched

    def spy(networks_list, games, r, temperature=1.0):
        out = real(networks_list, games, r, temperature)
        for nets, game, changes in zip(networks_list, games, out):
            hero_seat = next(s for s, n in enumerate(nets)
                             if float(n.weights[0].flatten()[0]) == key)
            seen.append((hero_seat, changes, game.state.big_blind))
        return out

    F.play_hands_batched = spy
    try:
        delta_bb, hands = F.evaluate_matchup(
            hero, opps, NET_CONFIG,
            FitnessConfig(hands_per_matchup=120, num_players=6), seed=7)
    finally:
        F.play_hands_batched = real

    hero_total = sum(changes[seat] / bb for seat, changes, bb in seen)
    seat0_total = sum(changes[0] / bb for _, changes, bb in seen)

    assert hands == len(seen)
    assert delta_bb == pytest.approx(hero_total, abs=1e-6)
    # The hero genuinely moves around the table, so the old behaviour differs.
    assert len({s for s, _, _ in seen}) > 1
    assert delta_bb != pytest.approx(seat0_total, abs=1e-6)


def test_evaluation_conditions_match_the_config():
    """
    Stacks, blinds and antes must come from the config.  These were previously
    read with getattr() defaults for fields that did not exist, so every
    evaluation silently randomised them and no saved config recorded it.
    """
    import training.fitness as F

    rng = np.random.default_rng(3)
    hero = rng.normal(0, 0.5, GENOME_SIZE).astype(np.float32)
    opps = [rng.normal(0, 0.5, GENOME_SIZE).astype(np.float32) for _ in range(5)]

    seen = []
    real = F.play_hands_batched

    def spy(networks_list, games, r, temperature=1.0):
        for game in games:          # capture before the hand mutates stacks
            seen.append((tuple(p.stack + p.total_contributed for p in game.players),
                         game.state.small_blind, game.state.big_blind,
                         getattr(game.state, 'ante', 0)))
        return real(networks_list, games, r, temperature)

    cfg = FitnessConfig(hands_per_matchup=64, num_players=6,
                        starting_stack=1000, small_blind=5, big_blind=10, ante=0)
    F.play_hands_batched = spy
    try:
        F.evaluate_matchup(hero, opps, NET_CONFIG, cfg, seed=11)
    finally:
        F.play_hands_batched = real

    assert {s for stacks, _, _, _ in seen for s in stacks} == {1000}
    assert {(sb, bb, ante) for _, sb, bb, ante in seen} == {(5, 10, 0)}


# ---------------------------------------------------------------------------
# Training and inference must see the same thing
# ---------------------------------------------------------------------------

def test_features_identical_between_training_and_inference():
    """
    Training feeds networks FeatureCache.get_features(); evaluation, the GUI and
    the RL env call get_state_vector().  Two different 17-dim layouts once
    coexisted, so agents were evaluated on permuted inputs.
    """
    rng = np.random.default_rng(4)
    names = get_feature_names()
    for _ in range(40):
        game = random_hand(rng, 6)
        for pid in range(6):
            trained = FeatureCache(game, pid).get_features(game).copy()
            inferred = np.asarray(get_state_vector(game, pid))
            assert len(trained) == len(names)
            np.testing.assert_allclose(trained, inferred, rtol=0, atol=0)


def test_abstract_mask_has_one_definition():
    """The training helper and the engine must produce the same 6-slot mask."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        game = random_hand(rng, 6)
        for pid in range(6):
            np.testing.assert_array_equal(create_action_mask(game, pid),
                                          get_abstract_action_mask(game, pid))


def test_engine_mask_keeps_its_documented_five_slots():
    """
    get_action_mask is [fold, check, call, raise, all-in] — one slot per engine
    action type.  Callers that need the 6 abstract actions must use
    get_abstract_action_mask rather than zero-padding this to length 6, which
    made all-in permanently illegal and shifted every other slot's meaning.
    """
    rng = np.random.default_rng(6)
    game = random_hand(rng, 2)
    mask = get_action_mask(game, game.state.current_player)
    assert len(mask) == 5
    assert len(get_abstract_action_mask(game, game.state.current_player)) == 6


# ---------------------------------------------------------------------------
# Cards must actually vary between hands
# ---------------------------------------------------------------------------

def test_deck_differs_between_hands():
    """
    Every hand must get a fresh shuffle.

    `reset_hand()` used to rebuild the deck from one fixed seed, so a session
    contained exactly two distinct deals alternating with the button. Anything
    looping `reset_hand()` — `self_play.play_match`, and therefore every
    round-robin tournament, plus the GUI — scored agents on two repeated hands.

    Stacks are deep enough here that nobody busts on blinds and stops being
    dealt in.
    """
    game = PokerGame([100_000, 100_000], 5, 10, seed=42)
    deals = []
    for _ in range(200):
        deals.append(tuple(str(c) for p in game.players for c in p.hole_cards))
        game.reset_hand()
    assert len(set(deals)) == len(deals), "duplicate deals within one session"


def test_deck_sequence_is_reproducible():
    """A seed must still reproduce a whole session, not just its first hand."""
    def sequence():
        game = PokerGame([100_000, 100_000], 5, 10, seed=7)
        out = []
        for _ in range(40):
            out.append(tuple(str(c) for p in game.players for c in p.hole_cards))
            game.reset_hand()
        return out

    assert sequence() == sequence()
    first = PokerGame([100_000, 100_000], 5, 10, seed=7)
    assert sequence()[0] == tuple(str(c) for p in first.players for c in p.hole_cards)


def test_blinds_come_from_funded_seats():
    """
    Busted seats must not be handed a blind.

    They used to be, posting zero — which left the pot short, marked a busted
    player all-in, and with enough busted seats produced a hand where nobody
    posted anything.
    """
    game = PokerGame([1000, 0, 0, 1000, 0, 1000], 5, 10, seed=5)
    posted = {i: p.total_contributed for i, p in enumerate(game.players)
              if p.total_contributed > 0}
    assert game.state.pot.total == 15, f"blinds not fully posted: {posted}"
    for seat in posted:
        assert game.players[seat].stack + game.players[seat].total_contributed > 0


def test_lone_survivor_is_paid_without_evaluating_hands():
    """
    A hand won by folding can end before the board is complete. The winner must
    simply be paid; running the five-card comparator on a single candidate
    raised `Need at least 5 cards, got 2`.
    """
    game = PokerGame([1000, 1000, 1000], 5, 10, seed=3)
    game.state.community_cards = []
    for player in game.players[1:]:
        player.has_folded = True
    pot = game.state.pot.total
    game.state.pot.create_side_pots(game.players)
    winnings = game.resolve_showdown()          # must not raise
    assert winnings[0] == pot
    assert game.state.pot.total == 0, "pot not cleared after payout"


# ---------------------------------------------------------------------------
# Hand strength must actually rank hands
# ---------------------------------------------------------------------------

def test_hand_strength_responds_to_the_board():
    """
    The hand-quality feature must change when the community cards change.

    It was the Chen score of the two hole cards, cached for the whole hand, so
    it was identical for every possible board — the agent played every street
    after the flop unable to see it.
    """
    from engine.features import made_hand_strength
    from engine.cards import Card as C

    hole = [C('J', 'd'), C('J', 'c')]
    quads = made_hand_strength(hole, [C('J', 'h'), C('J', 's'), C('4', 'd')])
    trips = made_hand_strength(hole, [C('J', 'h'), C('6', 's'), C('2', 'd')])
    weak = made_hand_strength(hole, [C('A', 'h'), C('K', 'c'), C('Q', 'd')])

    assert quads > trips > weak
    assert len({quads, trips, weak}) == 3


def test_improving_never_lowers_hand_strength():
    """
    Flopping a set must not score below the same hand preflop. Raw category
    index and normalised Chen score are different scales; mixing them made
    strength *fall* as the hand improved.
    """
    from engine.features import made_hand_strength, get_preflop_strength_fast
    from engine.cards import Card as C

    hole = [C('J', 'd'), C('J', 'c')]
    assert made_hand_strength(hole, [C('J', 'h'), C('6', 's'), C('2', 'd')]) \
        > get_preflop_strength_fast(hole)


def test_opponent_all_in_feature_carries_information():
    """
    Slot 6 held the acting player's own all-in flag, which is necessarily 0 —
    an all-in player never gets a turn. Measured std was exactly 0.000, so the
    network had an input that could never inform it.
    """
    from engine.features import FeatureCache, get_feature_names

    idx = get_feature_names().index('opponent_all_in')
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(120):
        game = random_hand(rng, 4)
        caches = [FeatureCache(game, k) for k in range(4)]
        guard = 0
        while not game.is_hand_over() and guard < 100:
            cur = game.state.current_player
            if cur is None:
                break
            seen.append(caches[cur].get_features(game)[idx])
            legal = game.get_legal_actions(cur)
            if not legal:
                break
            choice = legal[rng.integers(len(legal))]
            game.apply_action(cur, Action(choice['type'], choice.get('amount')))
            guard += 1
    assert np.std(seen) > 0.0, "feature is constant and therefore inert"

def test_every_starting_hand_has_a_strength():
    """
    All 169 starting hands must be in the lookup table.

    Pocket pairs are always offsuit, and the table was built skipping the
    offsuit entry whenever both ranks matched — so every pair missed the table
    and fell through to the 0.5 default.  Aces and deuces were indistinguishable
    from an average hand, on 6% of all deals.
    """
    from engine.cards import RANKS
    from engine.features import get_preflop_strength_fast, preflop_hand_strength
    from engine.cards import Card as C

    for r1 in RANKS:
        for r2 in RANKS:
            suits = ('d', 'c') if r1 == r2 else ('h', 'd')
            hand = [C(r1, suits[0]), C(r2, suits[1])]
            assert get_preflop_strength_fast(hand) == pytest.approx(
                preflop_hand_strength(hand)), f"{r1}{r2} not in strength table"


def test_strong_pairs_outrank_weak_ones():
    """A hand-quality feature that cannot separate aces from deuces is inert."""
    from engine.cards import Card as C
    from engine.features import get_preflop_strength_fast as strength

    aces = strength([C('A', 'd'), C('A', 'c')])
    deuces = strength([C('2', 'd'), C('2', 'c')])
    trash = strength([C('7', 'h'), C('2', 'd')])
    assert aces > deuces > trash
    assert aces > 0.5, "pocket aces must not read as an average hand"


# ---------------------------------------------------------------------------
# Advertised actions must be performable
# ---------------------------------------------------------------------------

def test_legal_actions_are_satisfiable():
    """
    get_legal_actions must not offer a raise whose minimum exceeds its maximum.
    apply_action silently clamped such raises to an all-in, hiding the problem.
    """
    rng = np.random.default_rng(7)
    for _ in range(300):
        stacks = list(rng.integers(20, 1501, 3))
        game = PokerGame(list(stacks), 5, 10, seed=int(rng.integers(0, 2 ** 31)))
        guard = 0
        while not game.is_hand_over() and guard < 200:
            cur = game.state.current_player
            if cur is None:
                break
            for action in game.get_legal_actions(cur):
                if action['type'] == 'raise':
                    assert action['min'] <= action['max'], (
                        f"unsatisfiable raise offered: {action}")
            legal = game.get_legal_actions(cur)
            if not legal:
                break
            choice = legal[rng.integers(len(legal))]
            if choice['type'] == 'raise':
                target = int(rng.integers(choice['min'], choice['max'] + 1))
                act = Action('raise', max(1, target - game.state.players[cur].bet))
            else:
                act = Action(choice['type'], choice.get('amount'))
            game.apply_action(cur, act)
            guard += 1


# ---------------------------------------------------------------------------
# Reported statistics must describe what was measured
# ---------------------------------------------------------------------------

def _tiny_trainer(**evolution_kwargs):
    from training.evolution import EvolutionTrainer
    cfg = TrainingConfig(
        network=NET_CONFIG,
        evolution=EvolutionConfig(population_size=8, **evolution_kwargs),
        fitness=FitnessConfig(hands_per_matchup=24, matchups_per_agent=2, num_players=6),
        num_generations=2, seed=42, experiment_name='_invariant_test')
    trainer = EvolutionTrainer(cfg)
    trainer.initialize()
    return trainer


def test_generation_stats_describe_the_evaluated_population():
    """
    Statistics must be gathered before the population is replaced.  Collecting
    them afterwards averaged over the single surviving elite, so every logged
    'mean' equalled 'max' and every 'std' was exactly 0.
    """
    trainer = _tiny_trainer()
    stats = trainer.train_generation()
    assert stats['std_fitness'] > 0.0
    assert stats['mean_fitness'] != stats['max_fitness']
    assert stats['min_fitness'] <= stats['mean_fitness'] <= stats['max_fitness']


def test_sigma_decays_once_per_generation():
    """
    Population.evolve() already decays sigma; the trainer must not decay it a
    second time, which produced sigma * decay**(2g) instead of sigma * decay**g.
    """
    trainer = _tiny_trainer(mutation_sigma=0.1, mutation_decay=0.5)
    start = trainer.factory.current_sigma
    for _ in range(3):
        trainer.train_generation()
    assert trainer.factory.current_sigma == pytest.approx(start * 0.5 ** 3, rel=1e-9)


# ---------------------------------------------------------------------------
# Sanity floor: fitness magnitudes
# ---------------------------------------------------------------------------

def test_action_conversion_never_produces_an_illegal_action():
    """
    Every abstract action the mask marks legal must convert into an action the
    engine accepts, without falling back to the exception-swallowing fold path.
    """
    rng = np.random.default_rng(8)
    for _ in range(200):
        game = random_hand(rng, 6)
        guard = 0
        while not game.is_hand_over() and guard < 200:
            cur = game.state.current_player
            if cur is None:
                break
            player = game.players[cur]
            if player.has_folded or player.is_all_in:
                break
            mask = get_abstract_action_mask(game, cur)
            legal_idxs = np.flatnonzero(mask)
            idx = int(legal_idxs[rng.integers(len(legal_idxs))])
            action = abstract_action_to_engine_action(idx, game, cur)
            assert game.is_action_legal(cur, action), (
                f"abstract action {idx} -> {action} rejected by engine")
            game.apply_action(cur, action)
            guard += 1
