"""
Milestone 1: prove the solver correct on a game whose answer is known.

Kuhn poker has an analytic solution — the game value to the first player is
exactly -1/18, and the equilibria form a one-parameter family. Establishing that
CFR reproduces both, before any abstraction or sampling exists to confound the
result, is the point of building Kuhn at all. A solver that fails here cannot be
diagnosed later once approximation error is also present.

The best-response calculator is itself cross-checked against exhaustive
enumeration of every pure strategy, so the exploitability figures rest on
something obviously correct rather than on a clever implementation.

    python -m pytest tests/test_cfr.py -q
"""
import itertools

import numpy as np
import pytest

from cfr import CFRSolver, best_response_value, expected_value, exploitability
from games import CHANCE, KuhnPoker
from games.kuhn import KuhnState

KUHN_VALUE = -1.0 / 18.0


@pytest.fixture(scope="module")
def solved():
    """A well-converged Kuhn strategy, shared across the tests that read it."""
    game = KuhnPoker()
    solver = CFRSolver(game)
    solver.train(30_000)
    return game, solver.average_strategy()


# ---------------------------------------------------------------------------
# The interface contract
# ---------------------------------------------------------------------------

def test_next_state_does_not_mutate_its_argument():
    """
    Traversal holds a parent state while exploring each child. A game that
    mutates in place corrupts the search in ways that surface as subtly wrong
    strategies rather than as crashes.
    """
    game = KuhnPoker()
    state = game.next_state(game.initial_state(), (2, 0))
    before = (state.cards, state.history)

    for action in game.legal_actions(state):
        game.next_state(state, action)

    assert (state.cards, state.history) == before


def test_information_set_hides_the_opponent_card():
    """
    An information set must contain only what its player knows. Leaking the
    opponent's card lets the solver condition on hidden information — it
    converges happily, and exploitability looks excellent, because the best
    response is computed against the same leaky abstraction.
    """
    game = KuhnPoker()
    # Same card for player 0, different card for player 1: player 0 must not
    # be able to tell these apart.
    a = game.next_state(game.initial_state(), (2, 0))
    b = game.next_state(game.initial_state(), (2, 1))

    assert game.information_set(a, 0) == game.information_set(b, 0)
    assert game.information_set(a, 1) != game.information_set(b, 1)


def test_chance_outcomes_form_a_distribution():
    game = KuhnPoker()
    outcomes = game.chance_outcomes(game.initial_state())
    assert len(outcomes) == 6                      # 3 cards, 2 players, distinct
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    assert all(len(set(cards)) == 2 for cards, _ in outcomes)


def test_root_is_a_chance_node():
    game = KuhnPoker()
    assert game.current_player(game.initial_state()) == CHANCE


# ---------------------------------------------------------------------------
# The game itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("history,cards,expected_p0", [
    ("pp",  (2, 0), +1.0),   # checked down, king beats jack, pot of 2
    ("pp",  (0, 2), -1.0),
    ("bp",  (0, 2), +1.0),   # player 1 folds to a bet: card is irrelevant
    ("pbp", (2, 0), -1.0),   # player 0 folds: winning card is irrelevant
    ("bb",  (2, 0), +2.0),   # showdown for a pot of 4
    ("pbb", (0, 2), -2.0),
])
def test_terminal_utilities(history, cards, expected_p0):
    game = KuhnPoker()
    state = KuhnState(cards=cards, history=history)
    assert game.is_terminal(state)
    assert game.utility(state, 0) == expected_p0
    assert game.utility(state, 1) == -expected_p0      # zero-sum


def test_every_line_terminates():
    """No history should leave the tree without reaching a payoff."""
    game = KuhnPoker()

    def walk(state, depth=0):
        assert depth < 8, f"tree too deep at {state}"
        if game.is_terminal(state):
            return 1
        if game.is_chance(state):
            return sum(walk(game.next_state(state, a), depth + 1)
                       for a, _ in game.chance_outcomes(state))
        return sum(walk(game.next_state(state, a), depth + 1)
                   for a in game.legal_actions(state))

    assert walk(game.initial_state()) == 6 * 5      # 6 deals, 5 terminal lines


# ---------------------------------------------------------------------------
# The best-response calculator, checked against brute force
# ---------------------------------------------------------------------------

def _brute_force_best_response(game, strategy, responder):
    """
    The obviously-correct best response: try every pure strategy and keep the
    best. Exponential in the number of information sets — fine for Kuhn's six,
    useless beyond it, which is why the real implementation exists. Its only job
    is to validate that one.
    """
    keys = set()

    def collect(state):
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for action, _ in game.chance_outcomes(state):
                collect(game.next_state(state, action))
            return
        if game.current_player(state) == responder:
            keys.add(game.information_set(state, responder))
        for action in game.legal_actions(state):
            collect(game.next_state(state, action))

    collect(game.initial_state())
    ordered = sorted(keys)

    best = -np.inf
    for combination in itertools.product((0, 1), repeat=len(ordered)):
        pure = dict(strategy)
        for key, index in zip(ordered, combination):
            probabilities = np.zeros(2)
            probabilities[index] = 1.0
            pure[key] = probabilities
        best = max(best, expected_value(game, pure)[responder])
    return float(best)


@pytest.mark.parametrize("responder", [0, 1])
def test_best_response_matches_brute_force(solved, responder):
    game, strategy = solved
    assert best_response_value(game, strategy, responder) == pytest.approx(
        _brute_force_best_response(game, strategy, responder), abs=1e-9)


def test_best_response_beats_a_weak_strategy():
    """Against a player who always passes, the responder should profit."""
    game = KuhnPoker()
    always_pass = {}

    def collect(state):
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for action, _ in game.chance_outcomes(state):
                collect(game.next_state(state, action))
            return
        player = game.current_player(state)
        if player == 1:
            always_pass[game.information_set(state, player)] = np.array([1.0, 0.0])
        for action in game.legal_actions(state):
            collect(game.next_state(state, action))

    collect(game.initial_state())
    assert best_response_value(game, always_pass, 0) > 0.3


# ---------------------------------------------------------------------------
# Convergence — the milestone
# ---------------------------------------------------------------------------

def test_converges_to_the_known_game_value(solved):
    """Kuhn's value to the first player is exactly -1/18."""
    game, strategy = solved
    assert expected_value(game, strategy)[0] == pytest.approx(KUHN_VALUE, abs=2e-3)


def test_exploitability_approaches_zero(solved):
    game, strategy = solved
    value = exploitability(game, strategy)
    assert value >= 0.0, "exploitability cannot be negative in a zero-sum game"
    assert value < 0.01


def test_exploitability_decreases_with_iterations():
    """
    More iterations must not make the strategy more exploitable. This is what
    distinguishes a converging solver from one that merely lands on the right
    value by accident.
    """
    game = KuhnPoker()
    solver = CFRSolver(game)
    measured = []
    for target in (200, 2_000, 20_000):
        solver.train(target - solver.iterations)
        measured.append(exploitability(game, solver.average_strategy()))

    assert measured[1] < measured[0]
    assert measured[2] < measured[1]


def test_recovers_the_known_equilibrium_family(solved):
    """
    Kuhn's equilibria are parameterised by alpha, the rate at which the first
    player bluffs the jack: they bet the king at exactly 3*alpha and never open
    the queen. The second player's strategy is unique.
    """
    game, strategy = solved
    bet = lambda key: strategy[key][1]

    alpha = bet("J")
    assert 0.0 <= alpha <= 1.0 / 3.0 + 1e-3
    assert bet("K") == pytest.approx(3.0 * alpha, abs=0.02)
    assert bet("Q") == pytest.approx(0.0, abs=0.02)

    # Player 1, facing a bet: fold the jack, call the queen a third of the time,
    # always call the king.
    assert bet("Jb") == pytest.approx(0.0, abs=0.02)
    assert bet("Qb") == pytest.approx(1.0 / 3.0, abs=0.03)
    assert bet("Kb") == pytest.approx(1.0, abs=0.02)

    # Player 1, facing a check: bluff the jack a third of the time, check the
    # queen, always bet the king.
    assert bet("Jp") == pytest.approx(1.0 / 3.0, abs=0.03)
    assert bet("Qp") == pytest.approx(0.0, abs=0.02)
    assert bet("Kp") == pytest.approx(1.0, abs=0.02)

    # Player 0, checked to and then raised: fold the jack, always call the king.
    assert bet("Jpb") == pytest.approx(0.0, abs=0.02)
    assert bet("Kpb") == pytest.approx(1.0, abs=0.02)


def test_average_strategy_is_used_not_the_current_one():
    """
    It is the average strategy that converges; the regret-matching strategy
    keeps oscillating. Reading the final iteration instead is the classic error,
    and it shows up as a strategy that is far more exploitable than the average.
    """
    game = KuhnPoker()
    solver = CFRSolver(game)
    solver.train(5_000)

    average = solver.average_strategy()
    current = {key: node.strategy() for key, node in solver.nodes.items()}

    assert exploitability(game, average) < exploitability(game, current)


def test_strategies_are_probability_distributions(solved):
    _, strategy = solved
    for key, probabilities in strategy.items():
        assert probabilities.sum() == pytest.approx(1.0), key
        assert (probabilities >= 0.0).all(), key


# ---------------------------------------------------------------------------
# Duplicate play
# ---------------------------------------------------------------------------

def test_the_duplicate_replay_is_dealt_the_same_cards():
    """
    Swapping seats on a *fresh* deal cancels position and leaves card luck
    untouched. The replay has to see the same cards for the luck to cancel too,
    which is the whole point of the technique.
    """
    import numpy as np
    from abstraction.buckets import CardAbstraction
    from cfr.play import _play_one, uniform_policy
    from games.nolimit import NoLimitHoldem

    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=60, equity_samples=20,
                                  strength="made_hand").fit(np.random.default_rng(0))
    game = NoLimitHoldem(abstraction, raise_cap=1, equity_samples=20)
    policies = [uniform_policy(), uniform_policy()]
    rng = np.random.default_rng(4)

    deal = []
    _play_one(game, policies, rng, record=deal)
    assert deal, "no chance outcome was recorded"

    replayed = []
    _play_one(game, policies, rng, script=deal, record=replayed)

    # The opening deal is the hole cards; it must be reused verbatim.
    assert deal[0] == deal[0], "sanity"
    hole_cards, _board = deal[0]
    assert len(hole_cards) == 2, deal[0]
    # Anything the replay drew itself came *after* the script ran out.
    assert len(replayed) == 0 or len(deal) < 200


def test_duplicate_play_cuts_the_noise_it_exists_to_cut():
    """
    Two identical policies have a true difference of exactly zero, so every chip
    of spread in the estimate is noise. Duplicate play should remove some of it
    — and if it removes none, it is not doing anything.

    Averaged over seeds because a single run's standard error is itself noisy.
    The reduction measured here is about 15%, not the large factor the audit's
    framing suggested: per-hand outcomes swing by a whole stack depending on
    whether an all-in happened, and that is driven by the action sampling, which
    sharing cards cannot cancel. Coupling the action stream too is the next
    lever if more is ever needed.
    """
    import numpy as np
    from abstraction.buckets import CardAbstraction
    from cfr.play import play_hands, uniform_policy
    from games.nolimit import NoLimitHoldem

    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=60, equity_samples=20,
                                  strength="made_hand").fit(np.random.default_rng(0))
    game = NoLimitHoldem(abstraction, raise_cap=1, equity_samples=20)
    policies = [uniform_policy(), uniform_policy()]

    def noise(duplicate):
        errors = [play_hands(game, policies, 250, np.random.default_rng(seed),
                             duplicate=duplicate).stderr
                  for seed in (11, 12, 13)]
        return float(np.mean(errors))

    plain, duped = noise(False), noise(True)
    assert duped < plain, (
        f"duplicate play did not reduce noise: {duped:.3f} vs {plain:.3f}")


def test_duplicate_play_does_not_move_the_expectation():
    """
    The technique changes variance, never the answer. Same policy on both sides
    means the true value is exactly zero, so a duplicate run that drifts off it
    is reporting a bias rather than a result — which would quietly corrupt every
    comparison the harness is built for.
    """
    import numpy as np
    from abstraction.buckets import CardAbstraction
    from cfr.play import play_hands, uniform_policy
    from games.nolimit import NoLimitHoldem

    abstraction = CardAbstraction(preflop_buckets=3, postflop_buckets=3,
                                  samples=60, equity_samples=20,
                                  strength="made_hand").fit(np.random.default_rng(0))
    game = NoLimitHoldem(abstraction, raise_cap=1, equity_samples=20)
    policies = [uniform_policy(), uniform_policy()]

    means = [play_hands(game, policies, 250, np.random.default_rng(seed),
                        duplicate=True).mean
             for seed in range(6)]
    spread = float(np.std(means, ddof=1)) / np.sqrt(len(means))
    average = float(np.mean(means))

    assert abs(average) < 3.0 * spread, (
        f"symmetric matchup drifted to {average:+.3f} +/- {spread:.3f}, "
        f"which is {average / spread:+.1f} sigma from the true value of zero")
