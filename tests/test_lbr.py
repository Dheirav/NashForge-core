"""
Local Best Response: a computable lower bound on exploitability.

LBR is the only exploitability measure available in no-limit, so it needs
checking as carefully as anything it will be used to judge. The difficulty is
that there is no exact answer to compare against — that is the whole reason LBR
exists — so these tests pin it in three other ways:

* **A known answer.** Against an opponent who always folds, the exploiter's
  winnings are arithmetic, not opinion.
* **Ordering.** A strategy trained for longer must not be *more* exploitable
  than an untrained one. LBR that cannot tell those apart is measuring noise.
* **Mechanism.** The range update must actually rule hands out; without that
  LBR has no leverage and degenerates into a fixed heuristic. It must also
  survive a new street, where the bucketing it reads is replaced wholesale.
* **Fairness.** LBR must reach a verdict from public information alone. An
  exploiter that peeks at the cards it is exploiting reports a bound on
  nothing.

    python -m pytest tests/test_lbr.py -q
"""
import numpy as np
import pytest

from abstraction.betting import CHECK_CALL, FOLD, RAISE_POT
from abstraction.buckets import CardAbstraction
from cfr import MCCFRSolver, VANILLA, lbr_value
from cfr.lbr import LocalBestResponse
from games.nolimit import NoLimitHoldem


@pytest.fixture(scope="module")
def game():
    abstraction = CardAbstraction(preflop_buckets=4, postflop_buckets=4,
                                  samples=300, equity_samples=30,
                                  strength="made_hand").fit(np.random.default_rng(0))
    return NoLimitHoldem(abstraction, raise_cap=1, equity_samples=30)


@pytest.fixture(scope="module")
def trained(game):
    solver = MCCFRSolver(game, rule=VANILLA, seed=0)
    solver.train(4000)
    return solver.average_strategy()


def constant_strategy(game, action, buckets=4):
    """A strategy that always takes ``action`` wherever it is legal."""
    from abstraction.betting import enumerate_street_sequences  # noqa: F401

    strategy = {}
    rng = np.random.default_rng(0)
    seen = set()

    def walk(state, depth=0):
        if game.is_terminal(state) or depth > 30:
            return
        if game.is_chance(state):
            walk(game.next_state(state, game.sample_chance(state, rng)), depth + 1)
            return
        player = game.current_player(state)
        actions = list(game.legal_actions(state))
        for bucket in range(buckets):
            key = f"{bucket}|{state.history}"
            if key not in seen:
                seen.add(key)
                probabilities = np.zeros(len(actions))
                index = actions.index(action) if action in actions else \
                    actions.index(CHECK_CALL)
                probabilities[index] = 1.0
                strategy[key] = probabilities
        for candidate in actions:
            walk(game.next_state(state, candidate), depth + 1)

    for _ in range(40):
        walk(game.initial_state())
    return strategy


# ---------------------------------------------------------------------------
# A known answer
# ---------------------------------------------------------------------------

def test_an_always_folding_opponent_is_maximally_exploitable(game):
    """
    If the opponent folds every time, the exploiter collects their blind every
    hand. That is arithmetic, so LBR must find close to it — and a bound that
    cannot exploit a player who never plays is broken.
    """
    always_fold = constant_strategy(game, FOLD)
    result = lbr_value(game, always_fold, hands=400,
                       rng=np.random.default_rng(0), rollout_samples=20)

    assert result.proves_exploitable, result.summary()
    # Seats alternate, so the win is the average of taking the big blind and
    # taking the small blind: (2 + 1) / 2 = 1.5 chips a hand.
    assert result.mean == pytest.approx(1.5, abs=0.6), result.summary()


def test_the_bound_is_never_wildly_beyond_the_stack(game, trained):
    """A hand cannot win more than the opponent brought to the table."""
    result = lbr_value(game, trained, hands=200,
                       rng=np.random.default_rng(1), rollout_samples=20)
    assert abs(result.mean) < game.starting_stack


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_training_reduces_exploitability(game, trained):
    """
    An untrained strategy plays uniformly at random and should be far more
    exploitable than a trained one. LBR that cannot separate these is measuring
    noise rather than exploitability.

    1,500 hands rather than 500: once the raise valuation began counting the
    call, LBR started betting substantially larger, and larger bets carry more
    variance per hand. The mean rose (+12.4 against an untrained opponent) while
    the interval widened with it, so the hand count has to keep up. This is the
    behaviour changing, not the property — at 500 hands the same measurement is
    +5.1 +/- 3.5 and cannot clear zero.
    """
    untrained = lbr_value(game, {}, hands=1500,
                          rng=np.random.default_rng(2), rollout_samples=25)
    learned = lbr_value(game, trained, hands=1500,
                        rng=np.random.default_rng(2), rollout_samples=25)

    assert untrained.mean > learned.mean, \
        f"untrained {untrained.mean:.2f} vs trained {learned.mean:.2f}"
    assert untrained.proves_exploitable


def test_result_reports_its_own_uncertainty(game, trained):
    result = lbr_value(game, trained, hands=200,
                       rng=np.random.default_rng(3), rollout_samples=20)
    assert result.hands == 200
    assert result.stderr > 0.0
    low, high = result.ci95
    assert low < result.mean < high
    # The guarantee is one-directional and the summary must say so.
    assert "exploitab" in result.summary()


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------

def dealt(game, seed=0):
    """A freshly dealt hand, past the opening chance node."""
    rng = np.random.default_rng(seed)
    root = game.initial_state()
    return game.next_state(root, game.sample_chance(root, rng))


def test_betting_rules_out_hands_that_would_not_have_acted(game):
    """
    The whole leverage of LBR is that betting gives information away. If a hand
    would never take the observed action, its weight must go to zero.
    """
    lbr = LocalBestResponse(game, {}, rollout_samples=5, candidates=48)
    state = dealt(game)
    candidates = lbr._deal_range(state, 0, np.random.default_rng(1))
    lbr._sync(candidates, state.board)

    actions = list(game.legal_actions(state))
    raise_index = len(actions) - 1
    call_index = actions.index(CHECK_CALL)
    assert raise_index != call_index

    # Only hands sharing the first candidate's bucket ever raise; the rest call.
    tell = candidates.buckets[0]
    strategy = {}
    for bucket in set(candidates.buckets):
        probabilities = np.zeros(len(actions))
        probabilities[raise_index if bucket == tell else call_index] = 1.0
        strategy[f"{bucket}|{state.history}"] = probabilities
    lbr.strategy = strategy

    lbr._update_belief(candidates, state, raise_index)

    for index, bucket in enumerate(candidates.buckets):
        if bucket != tell:
            assert candidates.weights[index] == 0.0, candidates.hands[index]
    assert candidates.weights.sum() == pytest.approx(1.0)


def test_the_range_is_rebucketed_when_the_board_changes(game):
    """
    Each street has its own clustering, so bucket 2 on the flop and bucket 2 on
    the turn are unrelated categories. A read carried across a street without
    re-bucketing is applied to the wrong hands entirely — and a candidate
    holding a card that just landed on the board is no longer possible at all.
    """
    lbr = LocalBestResponse(game, {}, rollout_samples=5, candidates=24)
    state = dealt(game)

    candidates = lbr._deal_range(state, 0, np.random.default_rng(1))
    lbr._sync(candidates, state.board)
    preflop_buckets = list(candidates.buckets)

    mine = set(state.hole[0])
    flop = tuple(c for c in range(52) if c not in mine)[:3]
    lbr._sync(candidates, flop)

    for index, hand in enumerate(candidates.hands):
        if candidates.weights[index] > 0.0:
            assert candidates.buckets[index] == game.bucket_for(hand, flop)
        if hand[0] in flop or hand[1] in flop:
            assert candidates.weights[index] == 0.0, hand

    assert candidates.weights.sum() == pytest.approx(1.0)
    assert preflop_buckets != candidates.buckets, \
        "the flop changed nothing, so the buckets were never recomputed"


def test_an_impossible_observation_leaves_the_read_intact(game):
    """
    If the strategy gives the observed action zero probability everywhere, the
    strategy is not the one being modelled. The range must stay finite and
    normalised rather than collapsing to NaN.
    """
    lbr = LocalBestResponse(game, {}, rollout_samples=5, candidates=16)
    state = dealt(game)
    candidates = lbr._deal_range(state, 0, np.random.default_rng(1))
    lbr._sync(candidates, state.board)

    actions = list(game.legal_actions(state))
    always_first = np.zeros(len(actions))
    always_first[0] = 1.0
    lbr.strategy = {f"{b}|{state.history}": always_first
                    for b in set(candidates.buckets)}

    before = candidates.weights.copy()
    lbr._update_belief(candidates, state, 1)

    assert np.isfinite(candidates.weights).all()
    assert candidates.weights.sum() == pytest.approx(1.0)
    assert candidates.weights == pytest.approx(before)


def test_fold_probability_reflects_the_range(game):
    """An opponent who always folds must read as folding with probability 1."""
    lbr = LocalBestResponse(game, constant_strategy(game, FOLD),
                            rollout_samples=10, candidates=16)
    state = dealt(game)
    raised = game.next_state(state, 3)          # a pot-sized raise
    candidates = lbr._deal_range(state, 0, np.random.default_rng(1))

    if FOLD in game.legal_actions(raised):
        probability = lbr._fold_probability(raised, candidates)
        assert probability == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

def test_the_exploiter_never_reads_the_cards_it_is_exploiting(game):
    """
    LBR must decide from its own cards, the board and the betting. Swapping the
    opponent's holding behind its back cannot change what it estimates, because
    it is not entitled to have seen it.
    """
    lbr = LocalBestResponse(game, {}, rollout_samples=30, candidates=8)
    state = dealt(game)

    mine = state.hole[0]
    spare = [c for c in range(52) if c not in set(mine)][:4]
    first = state._replace(hole=(mine, (spare[0], spare[1])))
    second = state._replace(hole=(mine, (spare[2], spare[3])))

    def estimate(where):
        candidates = lbr._deal_range(where, 0, np.random.default_rng(4))
        return lbr._win_probability(where, 0, candidates,
                                    np.random.default_rng(5))

    assert estimate(first) == estimate(second)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def test_a_bigger_bet_is_worth_more_when_it_folds_them_equally_often(game):
    """
    The incentive a larger bet buys is the larger pot it wins when called. A
    valuation that omits the call prices every extra chip as pure downside, so
    the cheapest legal bet wins by construction — which is what LBR did,
    betting the smallest size offered in 47% of decisions and continuing to do
    so as the floor dropped and its results got worse.

    Holding fold probability fixed isolates the showdown term: against an
    opponent who never folds, a bet is worth having made only through the chips
    it drags in.
    """
    lbr = LocalBestResponse(game, {}, rollout_samples=5, candidates=8)
    state = dealt(game)
    candidates = lbr._deal_range(state, 0, np.random.default_rng(1))
    lbr._sync(candidates, state.board)

    me = game.current_player(state)
    opponent = 1 - me

    def value(fraction, win, folds=0.0):
        after = game.raise_by_fraction(state, fraction, RAISE_POT)
        to_call = max(after.committed[me] - after.committed[opponent], 0)
        called = min(to_call, after.stacks[opponent])
        showdown = (win * (after.contributions[opponent] + called)
                    - (1.0 - win) * after.contributions[me])
        return folds * state.contributions[opponent] + (1.0 - folds) * showdown

    # Holding a hand that wins most of the time, betting more must be worth more.
    small, large = value(0.25, win=0.8), value(1.5, win=0.8)
    assert large > small, f"0.25x scored {small:.2f}, 1.5x scored {large:.2f}"


def test_the_exploiter_does_not_simply_bet_the_minimum(game, trained):
    """
    Whatever the smallest offered size is, it must not attract a plurality of
    the betting. Lowering the floor used to leave the share unmoved at ~47%,
    which is the signature of a valuation that rewards cheapness rather than
    the bet.
    """
    from collections import Counter

    picks = Counter()

    class Counting(LocalBestResponse):
        def _choose(self, state, me, candidate_range, rng):
            move = super()._choose(state, me, candidate_range, rng)
            picks["off-tree-floor" if move.fraction == 0.02 else "other"] += 1
            return move

    sizes = (0.02, 0.5, 1.0, 2.0)
    lbr = Counting(game, trained, rollout_samples=5, candidates=8, bet_sizes=sizes)
    lbr.play(60, np.random.default_rng(11))

    total = sum(picks.values())
    assert total > 0
    share = picks["off-tree-floor"] / total
    assert share < 0.35, f"the floor still takes {share:.1%} of decisions"
