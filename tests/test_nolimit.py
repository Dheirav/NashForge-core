"""
Milestone 5: abstracted heads-up no-limit Hold'em.

The betting logic here is reimplemented rather than borrowed, because the
engine's version mutates in place and a traverser must be able to back up. That
makes these tests the only thing standing between a plausible-looking solver and
a game that quietly misallocates chips — which is precisely the failure this
project spent a session recovering from.

So the properties asserted are the accounting ones: chips are conserved, every
terminal is zero-sum, a fold pays exactly what the folder committed, and no line
runs forever. Card abstraction is checked to affect only the information-set
key, never the payoff.

    python -m pytest tests/test_nolimit.py -q
"""
import numpy as np
import pytest

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, RAISE_POT
from abstraction.buckets import CardAbstraction
from cfr import MCCFRSolver, VANILLA
from cfr.play import always_call_policy, play_hands, strategy_policy, uniform_policy
from games.base import CHANCE
from games.nolimit import NoLimitHoldem, STREET_NAMES

STARTING_STACK = 200
SMALL_BLIND, BIG_BLIND = 1, 2
TOTAL_CHIPS = 2 * STARTING_STACK


@pytest.fixture(scope="module")
def abstraction():
    """A deliberately tiny abstraction; these tests are about betting, not cards."""
    return CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                           samples=60, equity_samples=25).fit(np.random.default_rng(0))


@pytest.fixture(scope="module")
def game(abstraction):
    return NoLimitHoldem(abstraction, starting_stack=STARTING_STACK,
                         small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
                         raise_cap=1, equity_samples=25)


def play_random_hand(game, rng):
    """Drive one hand with random legal actions; returns the terminal state."""
    state = game.initial_state()
    for _ in range(200):
        if game.is_terminal(state):
            return state
        if game.is_chance(state):
            state = game.next_state(state, game.sample_chance(state, rng))
            continue
        actions = game.legal_actions(state)
        assert actions, f"no legal actions at {state.history!r}"
        state = game.next_state(state, actions[rng.integers(len(actions))])
    raise AssertionError(f"hand did not terminate: {state.history!r}")


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------

def test_chips_are_conserved(game):
    """Stacks plus contributions must always equal the chips brought to the table."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        state = play_random_hand(game, rng)
        assert sum(state.stacks) + sum(state.contributions) == TOTAL_CHIPS, state


def test_every_terminal_is_zero_sum(game):
    rng = np.random.default_rng(1)
    for _ in range(300):
        state = play_random_hand(game, rng)
        assert game.utility(state, 0) + game.utility(state, 1) == pytest.approx(0.0), state


def test_nobody_can_bet_more_than_their_stack(game):
    rng = np.random.default_rng(2)
    for _ in range(300):
        state = play_random_hand(game, rng)
        assert min(state.stacks) >= 0, state
        assert max(state.contributions) <= STARTING_STACK, state


def test_blinds_are_posted_before_the_deal(game):
    state = game.initial_state()
    assert state.contributions == (SMALL_BLIND, BIG_BLIND)
    assert state.stacks == (STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND)
    assert game.current_player(state) == CHANCE


def test_folding_preflop_forfeits_only_the_blind(game):
    """The small blind folding loses 1, not the whole stack."""
    rng = np.random.default_rng(3)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    assert game.current_player(state) == 0            # small blind acts first
    folded = game.next_state(state, FOLD)

    assert game.is_terminal(folded)
    assert game.utility(folded, 0) == -SMALL_BLIND
    assert game.utility(folded, 1) == SMALL_BLIND


def test_a_fold_pays_the_folder_s_contribution(game):
    """Whoever folds loses exactly what they put in, whatever their cards."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        state = play_random_hand(game, rng)
        if str(FOLD) not in state.history:
            continue
        losses = [game.utility(state, p) for p in (0, 1)]
        loser = 0 if losses[0] < 0 else 1
        assert losses[loser] == -state.contributions[loser], state.history


# ---------------------------------------------------------------------------
# Turn order and betting rules
# ---------------------------------------------------------------------------

def test_small_blind_acts_first_preflop_then_big_blind(game):
    rng = np.random.default_rng(5)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    assert game.current_player(state) == 0
    # The small blind completing does not end the street: the big blind has an option.
    called = game.next_state(state, CHECK_CALL)
    assert game.current_player(called) == 1
    assert not game.is_terminal(called)


def test_big_blind_acts_first_after_the_flop(game):
    rng = np.random.default_rng(6)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    state = game.next_state(state, CHECK_CALL)      # small blind completes
    state = game.next_state(state, CHECK_CALL)      # big blind checks: street closes
    assert game.current_player(state) == CHANCE     # deal the flop
    state = game.next_state(state, game.sample_chance(state, rng))
    assert game.current_player(state) == 1, "big blind is first to act postflop"


def test_a_single_check_does_not_close_a_street(game):
    """Postflop, one check leaves the other player to act."""
    rng = np.random.default_rng(7)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    state = game.next_state(state, CHECK_CALL)
    state = game.next_state(state, CHECK_CALL)
    state = game.next_state(state, game.sample_chance(state, rng))   # flop

    checked = game.next_state(state, CHECK_CALL)
    assert game.current_player(checked) == 0, "street closed after a single check"


def test_raises_are_capped(game):
    rng = np.random.default_rng(8)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    raised = game.next_state(state, RAISE_POT)
    # With cap 1 the opponent may only fold or call.
    assert set(game.legal_actions(raised)) <= {FOLD, CHECK_CALL}


def test_a_player_who_cannot_cover_may_only_fold_or_call(game):
    rng = np.random.default_rng(9)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    shoved = game.next_state(state, ALL_IN)
    assert set(game.legal_actions(shoved)) <= {FOLD, CHECK_CALL}


def test_hands_reach_every_street(game):
    """If the river is never reached, the street machinery is broken."""
    rng = np.random.default_rng(10)
    reached = set()
    for _ in range(400):
        state = play_random_hand(game, rng)
        reached.add(min(state.street, len(STREET_NAMES) - 1))
    assert reached == {0, 1, 2, 3}, reached


# ---------------------------------------------------------------------------
# Abstraction affects only what the solver sees
# ---------------------------------------------------------------------------

def test_information_sets_hide_the_opponent(game):
    """The key must contain a bucket and the public history, nothing else."""
    rng = np.random.default_rng(11)
    state = game.next_state(game.initial_state(),
                            game.sample_chance(game.initial_state(), rng))
    key = game.information_set(state, 0)
    bucket, _, history = key.partition("|")
    assert bucket.isdigit()
    assert history == state.history


def test_chance_cannot_be_enumerated(game):
    """
    No-limit chance is far too large to list. Raising is the honest answer, and
    is what forces a sampling solver rather than vanilla CFR.
    """
    with pytest.raises(NotImplementedError):
        game.chance_outcomes(game.initial_state())


def test_payoffs_do_not_depend_on_the_abstraction(abstraction):
    """
    Cards are dealt and settled for real; only the solver's view is bucketed.
    Two games differing only in granularity must agree on every payoff.
    """
    coarse = NoLimitHoldem(abstraction, raise_cap=1, equity_samples=25)
    fine = NoLimitHoldem(
        CardAbstraction(preflop_buckets=8, postflop_buckets=8, samples=60,
                        equity_samples=25).fit(np.random.default_rng(1)),
        raise_cap=1, equity_samples=25)

    rng = np.random.default_rng(12)
    for _ in range(60):
        state = play_random_hand(coarse, rng)
        assert coarse.utility(state, 0) == fine.utility(state, 0), state.history


# ---------------------------------------------------------------------------
# The solver runs on it
# ---------------------------------------------------------------------------

def test_mccfr_trains_and_finds_information_sets(game):
    solver = MCCFRSolver(game, rule=VANILLA, seed=0)
    solver.train(120)
    assert len(solver.nodes) > 50

    for key, probabilities in solver.average_strategy().items():
        assert probabilities.sum() == pytest.approx(1.0), key
        assert (probabilities >= 0.0).all(), key


def test_head_to_head_reports_uncertainty(game):
    """A chip count without a standard error cannot be compared to anything."""
    result = play_hands(game, [uniform_policy(), always_call_policy()], 200,
                        np.random.default_rng(0))
    assert result.hands == 200
    assert result.stderr > 0.0
    low, high = result.ci95
    assert low < result.mean < high


def test_alternating_seats_cancels_position(game):
    """
    Two identical policies must break even. Without alternating seats the small
    blind's positional disadvantage would show up as a spurious edge.
    """
    result = play_hands(game, [uniform_policy(), uniform_policy()], 400,
                        np.random.default_rng(0), alternate_seats=True)
    assert abs(result.mean) < 4 * max(result.stderr, 1e-9)


# ---------------------------------------------------------------------------
# Betting outside the abstraction
# ---------------------------------------------------------------------------

def test_an_off_tree_raise_moves_real_chips_but_records_an_abstract_one(game):
    """
    The pot must reflect what was actually wagered while the history names an
    action the opponent's strategy can answer. If the recorded action drove the
    chips instead, every payoff downstream of an off-tree bet would be wrong.
    """
    rng = np.random.default_rng(0)
    root = game.initial_state()
    dealt = game.next_state(root, game.sample_chance(root, rng))

    on_tree = game.next_state(dealt, RAISE_POT)          # 1.0x pot
    off_tree = game.raise_by_fraction(dealt, 0.7, RAISE_POT)

    # Perceived identically: same key, so the opponent answers the same way.
    assert off_tree.history == on_tree.history == str(RAISE_POT)

    # Paid differently: 0.7 pot is genuinely cheaper than 1.0 pot.
    actor = game.current_player(dealt)
    assert off_tree.contributions[actor] < on_tree.contributions[actor]
    assert off_tree.stacks[actor] > on_tree.stacks[actor]


def test_an_off_tree_raise_conserves_chips(game):
    """The accounting property that matters most, on the new path too."""
    rng = np.random.default_rng(1)
    root = game.initial_state()
    dealt = game.next_state(root, game.sample_chance(root, rng))

    for fraction in (0.25, 0.7, 1.4, 3.0):
        after = game.raise_by_fraction(dealt, fraction, RAISE_POT)
        assert sum(after.contributions) + sum(after.stacks) == TOTAL_CHIPS, fraction
        assert all(s >= 0 for s in after.stacks), fraction


def test_an_off_tree_raise_is_capped_by_the_stack(game):
    """A player cannot wager more than they have, whatever fraction is asked for."""
    rng = np.random.default_rng(2)
    root = game.initial_state()
    dealt = game.next_state(root, game.sample_chance(root, rng))

    actor = game.current_player(dealt)
    huge = game.raise_by_fraction(dealt, 50.0, RAISE_POT)
    assert huge.stacks[actor] == 0
    assert sum(huge.contributions) + sum(huge.stacks) == TOTAL_CHIPS


def test_the_recorded_action_must_be_one_the_abstraction_has(game):
    """
    Recording a fold or a call as the perception of a raise would corrupt the
    history, and the corruption would only surface much later as a strategy
    lookup that quietly missed.
    """
    rng = np.random.default_rng(3)
    root = game.initial_state()
    dealt = game.next_state(root, game.sample_chance(root, rng))

    for invalid in (FOLD, CHECK_CALL):
        with pytest.raises(ValueError):
            game.raise_by_fraction(dealt, 0.7, invalid)


def test_a_bucket_survives_its_cache_being_cleared():
    """
    The bucket memo is a pure optimisation, which is what makes capping it safe.

    `bucket_for` seeds its generator from `hash(key)`, so the same hole on the
    same board buckets identically whether or not the memo happens to hold it.
    The docstring's determinism guarantee comes from that seeding, not from the
    cache, and this is the property the size cap depends on: clearing costs
    recomputation, never a different answer.

    Without a cap the memo is unbounded. Keyed on (hole, board) it spans roughly
    1,326 x 2.1 million, so it never saturates: it reached 2,800,227 entries and
    491 MB by 60,000 iterations, which is what capped `bucket_sweep.py` at a
    1280-second budget.
    """
    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=40, equity_samples=8)
    abstraction.fit(np.random.default_rng(0))
    game = NoLimitHoldem(abstraction, equity_samples=8)

    holdings = [((0, 1), ()), ((0, 1), (2, 3, 4)), ((5, 6), (2, 3, 4, 7)),
                ((8, 9), (2, 3, 4, 7, 10))]
    first = [game.bucket_for(hole, board) for hole, board in holdings]

    game._bucket_cache.clear()
    again = [game.bucket_for(hole, board) for hole, board in holdings]
    assert first == again, "a cleared memo changed the bucket"


def test_the_bucket_cache_stops_growing_at_its_limit():
    """The cap is what bounds a long run's memory; assert it actually binds."""
    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=40, equity_samples=8)
    abstraction.fit(np.random.default_rng(0))
    game = NoLimitHoldem(abstraction, equity_samples=8, bucket_cache_limit=8)

    for card in range(40):
        game.bucket_for((card, card + 1), ())
    assert len(game._bucket_cache) <= 8, (
        f"cache holds {len(game._bucket_cache)} entries past a limit of 8")
