"""
Milestone 2: Leduc Hold'em on the same interface, and the solver unchanged.

Kuhn proved the solver correct against an analytic answer. Leduc has no closed
form, so the checks here are of two kinds:

* **Structural facts that can be counted.** The tree must contain exactly 288
  information sets — the figure quoted throughout the literature for this
  formulation — every chance node must carry a distribution, and every terminal
  must be zero-sum. These catch rule errors, which is where the bugs in a
  betting implementation live.

* **Behaviour of the solver on it.** Exploitability must fall monotonically. The
  best-response calculator was cross-checked against brute force on Kuhn, so
  here it is the measuring instrument rather than the thing measured — brute
  force is no longer available at 2^288 pure strategies.

The point of Leduc is also to test the interface itself: ``CFRSolver`` and
``exploitability`` are used here with no Leduc-specific code at all. If they had
needed any, the interface would have been shaped wrong.

    python -m pytest tests/test_leduc.py -q
"""
import numpy as np
import pytest

from cfr import CFRSolver, exploitability
from games import LeducHoldem
from games.base import CHANCE
from games.leduc import BET_SIZES, LeducState, RAISE_CAP, _contributions, rank_of

ANTE = 1


@pytest.fixture(scope="module")
def game():
    return LeducHoldem()


@pytest.fixture(scope="module")
def tree(game):
    """Walk the whole game once and collect what the structural tests assert on."""
    stats = {
        "chance": 0, "decisions": 0, "terminals": 0,
        "infosets": {0: set(), 1: set()},
        "zero_sum": True, "max_depth": 0,
        "probabilities_ok": True,
    }

    def walk(state, depth=0):
        stats["max_depth"] = max(stats["max_depth"], depth)
        assert depth < 40, f"tree does not terminate at {state}"

        if game.is_terminal(state):
            stats["terminals"] += 1
            if abs(game.utility(state, 0) + game.utility(state, 1)) > 1e-9:
                stats["zero_sum"] = False
            return
        if game.is_chance(state):
            stats["chance"] += 1
            outcomes = game.chance_outcomes(state)
            if abs(sum(p for _, p in outcomes) - 1.0) > 1e-9:
                stats["probabilities_ok"] = False
            for action, _ in outcomes:
                walk(game.next_state(state, action), depth + 1)
            return

        stats["decisions"] += 1
        player = game.current_player(state)
        stats["infosets"][player].add(game.information_set(state, player))
        for action in game.legal_actions(state):
            walk(game.next_state(state, action), depth + 1)

    walk(game.initial_state())
    return stats


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_information_set_count_matches_the_literature(tree):
    """
    288 information sets for two-round Leduc with a 6-card deck, keyed on rank.

    A rule error — a wrong raise cap, a missing betting line, a round that
    closes at the wrong point — moves this number, so it is the single most
    useful assertion about the implementation.
    """
    assert len(tree["infosets"][0]) == 144
    assert len(tree["infosets"][1]) == 144
    assert len(tree["infosets"][0]) + len(tree["infosets"][1]) == 288


def test_chance_structure(game, tree):
    assert tree["probabilities_ok"]
    # One deal of two distinct cards from six, then one board card from the
    # remaining four, for each of the five ways round one can close without a
    # fold: cc, rc, rrc, crc, crrc.
    assert tree["chance"] == 1 + 30 * 5

    deals = game.chance_outcomes(game.initial_state())
    assert len(deals) == 30
    assert all(len(set(cards)) == 2 for cards, _ in deals)


def test_every_terminal_is_zero_sum(tree):
    assert tree["zero_sum"]


def test_root_is_a_chance_node(game):
    assert game.current_player(game.initial_state()) == CHANCE


# ---------------------------------------------------------------------------
# Betting rules
# ---------------------------------------------------------------------------

def test_raises_are_capped_per_round(game):
    state = game.next_state(game.initial_state(), (4, 0))
    # Two raises exhaust the cap; only fold or call may follow.
    for _ in range(RAISE_CAP):
        assert "r" in game.legal_actions(state)
        state = game.next_state(state, "r")
    assert set(game.legal_actions(state)) == {"f", "c"}


def test_folding_is_not_offered_with_nothing_to_call(game):
    """Legal but strictly dominated, so it is kept out of the tree."""
    state = game.next_state(game.initial_state(), (4, 0))
    assert "f" not in game.legal_actions(state)


@pytest.mark.parametrize("history,expected", [
    ("",       [1, 1]),                       # antes only
    ("cc",     [1, 1]),                       # checked through
    ("rc",     [1 + 2, 1 + 2]),               # bet and called, round one
    ("rrc",    [1 + 4, 1 + 4]),               # bet, raise, call
    ("rf",     [1 + 2, 1]),                   # folded to a bet: no call posted
    ("cc/rc",  [1 + 4, 1 + 4]),               # round two bets are 4
    ("rc/rrc", [1 + 2 + 8, 1 + 2 + 8]),       # both rounds, capped
])
def test_contributions_are_derived_from_history(history, expected):
    """
    The pot is a pure function of the betting history. Storing it alongside the
    history is what lets the two drift apart.
    """
    contributed, _ = _contributions(history)
    assert contributed == expected


def test_bet_sizes_double_on_the_second_round():
    assert BET_SIZES == (2, 4)


# ---------------------------------------------------------------------------
# Showdown
# ---------------------------------------------------------------------------

def test_pairing_the_board_beats_a_higher_card(game):
    """A jack that pairs the board beats an unpaired king."""
    # Cards 0,1 are jacks; 2,3 queens; 4,5 kings.
    state = LeducState(hole=(0, 4), board=1, history="cc/cc")
    assert game.is_terminal(state)
    assert game.utility(state, 0) > 0, "paired jack should beat an unpaired king"
    assert game.utility(state, 1) == -game.utility(state, 0)


def test_higher_card_wins_when_neither_pairs(game):
    state = LeducState(hole=(4, 0), board=2, history="cc/cc")
    assert game.utility(state, 0) > 0
    state = LeducState(hole=(0, 4), board=2, history="cc/cc")
    assert game.utility(state, 0) < 0


def test_equal_ranks_split(game):
    """The two suited copies of a rank tie when neither pairs the board."""
    state = LeducState(hole=(0, 1), board=4, history="cc/cc")
    assert game.utility(state, 0) == 0.0
    assert game.utility(state, 1) == 0.0


def test_folding_forfeits_only_what_was_committed(game):
    """Player 1 folds to an opening bet: player 0 wins the ante, not the bet."""
    state = LeducState(hole=(0, 4), board=-1, history="rf")
    assert game.is_terminal(state)
    assert game.utility(state, 0) == 1.0        # opponent's ante
    assert game.utility(state, 1) == -1.0

    # Card strength is irrelevant to a fold.
    swapped = LeducState(hole=(4, 0), board=-1, history="rf")
    assert game.utility(swapped, 0) == 1.0


def test_rank_of_collapses_suits():
    assert [rank_of(c) for c in range(6)] == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# The solver, unchanged, on a bigger game
# ---------------------------------------------------------------------------

def test_information_sets_hide_the_opponent_card(game):
    """Same hole card for player 0, different for player 1: indistinguishable."""
    a = game.next_state(game.initial_state(), (4, 0))
    b = game.next_state(game.initial_state(), (4, 2))
    assert game.information_set(a, 0) == game.information_set(b, 0)
    assert game.information_set(a, 1) != game.information_set(b, 1)


def test_information_sets_hide_the_board_until_it_is_dealt(game):
    """The board must not appear in any round-one information set."""
    state = game.next_state(game.initial_state(), (4, 0))
    key = game.information_set(state, 0)
    assert key.startswith("K:"), key           # rank, no board, then history


def test_solver_runs_unchanged_and_reduces_exploitability(game):
    """
    The whole point of the interface: CFRSolver and exploitability are applied
    here with no Leduc-specific code.
    """
    solver = CFRSolver(game)
    measured = []
    for target in (10, 60, 240):
        solver.train(target - solver.iterations)
        measured.append(exploitability(game, solver.average_strategy()))

    assert all(value >= 0.0 for value in measured), measured
    assert measured[1] < measured[0], measured
    assert measured[2] < measured[1], measured


def test_strategies_are_probability_distributions(game):
    solver = CFRSolver(game)
    solver.train(20)
    for key, probabilities in solver.average_strategy().items():
        assert probabilities.sum() == pytest.approx(1.0), key
        assert (probabilities >= 0.0).all(), key
