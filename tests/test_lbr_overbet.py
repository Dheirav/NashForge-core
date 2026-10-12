"""
LBR's own off-tree sizes, read the way the live bridge reads them.

Without the option LBR translates its sizes onto the legal sized raises, which
clamps any overbet to 2x pot. `chipzen.bridge._as_abstract` instead reads a bet
of 1.5 times the largest size or more, and any bet that costs the stack, as
all-in, and `_close_street` re-reads the street if the hand goes on. These tests
pin that with `bridge_translation=True` the bot is asked what `replay` would ask
it, and that without it every earlier number reproduces to the last digit.

    venv/bin/python -m pytest tests/test_lbr_overbet.py -q
"""
import numpy as np
import pytest

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, RAISE_POT, RAISE_TWO
from abstraction.buckets import CardAbstraction
from cfr.lbr import DEFAULT_BET_SIZES, LocalBestResponse, Move, lbr_value
from chipzen.bridge import replay
from games.nolimit import NoLimitHoldem
from test_lbr_thirdraise import ShyOfShoves

BETWEEN = (0.33, 0.66, 1.33, 2.66, 5.33)     # scripts/lbr_ladder.py --between-sizes
HOLE = ((0, 13), (25, 38))


@pytest.fixture(scope="module")
def abstraction():
    return CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=300,
                           equity_samples=30, strength="made_hand").fit(np.random.default_rng(0))


@pytest.fixture(scope="module")
def game(abstraction):
    return NoLimitHoldem(abstraction, starting_stack=400, small_blind=1, big_blind=2,
                         raise_cap=(4, 2, 1), equity_samples=30)


def dealt(game):
    return game.next_state(game.initial_state(), (HOLE, ()))


def arena_for(bot, entries, phase, stacks):
    """The arena's turn_request for seat ``bot``; ``stacks`` is by seat."""
    history = [{"seat": 0, "action": "post_small_blind", "amount": 1, "phase": "preflop"},
               {"seat": 1, "action": "post_big_blind", "amount": 2, "phase": "preflop"}]
    history += [{"seat": s, "action": a, "amount": n, "phase": p} for s, a, n, p in entries]
    return {"action_history": history, "phase": phase,
            "your_stack": stacks[bot], "opponent_stacks": [stacks[1 - bot]]}


def bridge_reads(game, state, me, fraction):
    """The history `chipzen.bridge.replay` builds for LBR's raise of ``fraction`` here."""
    after = game.raise_by_fraction(state, fraction, RAISE_TWO)    # chips only; the label is ignored
    entries = [(0, "raise", 6, "preflop")] if state.history == str(RAISE_POT) else []
    entries.append((me, "raise", after.committed[me], "preflop"))
    hand = replay(arena_for(1 - me, entries, "preflop", after.stacks), seat=1 - me,
                  rng=np.random.default_rng(0), schedule=game.raise_cap)
    return hand.node.history


# ---------------------------------------------------------------------------

def test_off_reproduces_every_earlier_number(game):
    """
    Pinned from the code before this option existed, same game and seeds, for
    the --between-sizes menu and the default one, shared and paired generators.
    """
    strategy = ShyOfShoves(game)
    r = lbr_value(game, strategy, hands=60, rng=np.random.default_rng(5),
                  rollout_samples=10, candidates=8, bet_sizes=BETWEEN)
    assert (r.mean, r.stderr) == (27.366666666666667, 11.858087818207936)
    r = LocalBestResponse(game, strategy, 10, 8, bet_sizes=BETWEEN).play(60, paired_seed=21)
    assert (r.mean, r.stderr, float(r.values.sum())) == (36.016666666666666, 14.679326360924948, 2161.0)
    r = lbr_value(game, strategy, hands=60, rng=np.random.default_rng(5),
                  rollout_samples=10, candidates=8)
    assert (r.mean, r.stderr) == (8.95, 13.418474805345609)
    r = LocalBestResponse(game, strategy, 10, 8).play(60, paired_seed=21)
    assert (r.mean, r.stderr) == (10.008333333333333, 12.83062462242219)


@pytest.mark.parametrize("fraction", (2.75, 3.5, 5.33))
@pytest.mark.parametrize("opened", (False, True))
def test_an_overbet_is_heard_as_the_bridge_hears_it(game, fraction, opened):
    """
    An open (depth 0, sizes up to 2x) or a re-raise over a pot open (depth 1, 2x
    and all-in). With the option the label is the one `replay` builds from the
    same chips; without it every overbet is 2x pot. 2.75 sits under the
    bridge's 1.5 x 2x = 3.0 line and stays 2x either way, 3.5 and 5.33 are over it.
    """
    state = dealt(game)
    if opened:
        state = game.next_state(state, RAISE_POT)
    me = game.current_player(state)
    on = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
    off = LocalBestResponse(game, ShyOfShoves(game), 10, 8)
    heard = on._apply_move(state, Move(None, fraction), np.random.default_rng(0))
    assert heard.history == bridge_reads(game, state, me, fraction)
    assert heard.history[-1] == (str(ALL_IN) if fraction >= 3.0 else str(RAISE_TWO))
    old = off._apply_move(state, Move(None, fraction), np.random.default_rng(0))
    assert old.history[-1] == str(RAISE_TWO)
    # Same chips either way: only the strategy's reading moves.
    assert heard.contributions == old.contributions and heard.stacks[me] > 0


def test_a_bet_that_costs_the_stack_is_all_in_to_the_bridge(abstraction):
    """6bb deep a 2.75 pot open takes every chip: the bridge reads all-in, translation 2x."""
    short = NoLimitHoldem(abstraction, starting_stack=12, small_blind=1, big_blind=2,
                          raise_cap=(4, 2, 1), equity_samples=30)
    state = dealt(short)
    on = LocalBestResponse(short, ShyOfShoves(short), 10, 8, bridge_translation=True)
    heard = on._apply_move(state, Move(None, 2.75), np.random.default_rng(0))
    assert heard.stacks[0] == 0 and heard.history == str(ALL_IN)
    assert on.probe_stats["capped_allins"] == 1 and on.probe_stats["overbet_allins"] == 0
    old = LocalBestResponse(short, ShyOfShoves(short), 10, 8)._apply_move(
        state, Move(None, 2.75), np.random.default_rng(0))
    assert old.history == str(RAISE_TWO)


def test_the_bot_answers_the_all_in_node_and_the_street_is_re_read(game):
    """
    A 3.5 pot open. The bot is asked at "5", where this strategy folds all but
    its top bucket, not at "4", where it re-raises. Called, the street closes
    and the next one is looked up on the bridge's re-read line, "41/".
    """
    lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
    lbr._bridge = None
    state = dealt(game)
    heard = lbr._apply_move(state, Move(None, 3.5), np.random.default_rng(0))
    assert heard.history == "5" and heard.contributions == (16, 2)
    actions = list(game.legal_actions(heard))
    assert actions == [FOLD, CHECK_CALL]
    assert np.array_equal(lbr._policy(0, heard.history, len(actions)), [1.0, 0.0])
    assert np.array_equal(lbr._policy(3, heard.history, len(actions)), [0.0, 1.0])

    called = game.next_state(heard, CHECK_CALL)
    lbr._street_closes(called)
    flop = game.next_state(called, (2, 3, 4))
    hand = replay(arena_for(1, [(0, "raise", 16, "preflop"), (1, "call", 14, "preflop")],
                            "flop", (400 - 16, 400 - 16)),
                  seat=1, rng=np.random.default_rng(0), schedule=game.raise_cap)
    assert flop.history == hand.node.history == "51/"
    assert lbr._alt_history(flop.history) == hand.alt_history == "41/"
    assert lbr.probe_stats["overbet_allins"] == 1 and lbr.probe_stats["collapsed_streets"] == 1


@pytest.mark.parametrize("opened", (False, True))
def test_pricing_uses_the_bridges_distribution(game, opened):
    """
    `_bridge_distribution` lays out what `_as_abstract` samples; draw it many
    times and the frequencies must match, so pricing and play hear the same thing.
    """
    state = dealt(game)
    if opened:
        state = game.next_state(state, RAISE_POT)
    me = game.current_player(state)
    lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
    rng = np.random.default_rng(1)
    for fraction in BETWEEN + DEFAULT_BET_SIZES:
        readings, weights = lbr._bridge_distribution(state, me, fraction)
        draws = [lbr._bridge_perceive(state, me, fraction, lbr._bridge_hand(me), rng)
                 for _ in range(2000)]
        for reading, weight in zip(readings, weights):
            assert abs(np.mean([d == reading for d in draws]) - weight) < 0.04, (fraction, reading)
        assert set(draws) <= set(readings)


def test_paired_runs_count_what_the_bridge_read(game):
    """
    Over whole hands with the default menu: the bot answers every pseudo all-in
    LBR's 3.5 pot bets made, and a call collapses a street unless it was on the river.
    """
    strategy = ShyOfShoves(game)
    off = LocalBestResponse(game, strategy, 10, 8).play(200, paired_seed=11)
    lbr = LocalBestResponse(game, strategy, 10, 8, bridge_translation=True)
    on = lbr.play(200, paired_seed=11)
    stats = lbr.probe_stats
    assert stats["probes"] == 0 and stats["overbet_allins"] > 0
    assert stats["folded"] + stats["called"] == stats["overbet_allins"]
    # At most, not equal: a called river pseudo all-in has no next street to re-read, so it counts as
    # called and collapses nothing. Equality held only while this strategy happened never to call one.
    assert stats["collapsed_streets"] <= stats["called"]
    assert stats["alt_misses"] == 0
    # Some hands never reach a size the two readings differ on, and pairing
    # leaves those exactly equal.
    assert 0 < ((on.values - off.values) == 0).sum() < on.values.size
