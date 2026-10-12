"""
LBR's third-raise probe: a small raise where the tree's menu is all-in only.

The live bridge reads any raise at that depth as all-in (`chipzen.bridge._as_abstract`)
and, if the hand goes on, re-reads the street (`_close_street`). These tests pin
that LBR asks the strategy exactly what the bridge would ask it, that the chips
moved are the real ones, and that with the option off nothing changes at all.

    venv/bin/python -m pytest tests/test_lbr_thirdraise.py -q
"""
import numpy as np
import pytest

from abstraction.betting import (ALL_IN, CHECK_CALL, FOLD, RAISE_POT, RAISE_TWO,
                                 legal_actions)
from abstraction.buckets import CardAbstraction
from cfr.lbr import LocalBestResponse, Move, lbr_value
from chipzen.bridge import replay
from games.nolimit import NoLimitHoldem

PROBES = (0.4, 0.6, 1.0)
STACK = 400


class ShyOfShoves:
    """
    A strategy built from its key rather than stored: re-raise 2x pot over a
    preflop open, fold to anything read as all-in unless holding the top
    bucket, and otherwise check or call.

    It is the shape of the leak: it folds to a shove often enough that a shove
    looks attractive, and the shove risks a whole stack where the bridge would
    have read a 0.4 pot raise the same way.
    """

    def __init__(self, game, top=3):
        self.game, self.top = game, top

    def get(self, key, default=None):
        bucket, history = key.split("|", 1)
        street = history.count("/")
        acts = history.split("/")[-1]
        raises = sum(c in "2345" for c in acts)
        last = int(acts[-1]) if acts else None
        facing = (last is not None and last >= 2) or (street == 0 and not acts)
        actions = list(legal_actions(raises, facing, self.game.raise_cap, last, street))
        p = np.zeros(len(actions))
        if last == ALL_IN:
            p[actions.index(CHECK_CALL if int(bucket) >= self.top else FOLD)] = 1.0
        elif street == 0 and raises == 1 and RAISE_TWO in actions:
            p[actions.index(RAISE_TWO)] = 1.0
        else:
            p[actions.index(CHECK_CALL)] = 1.0
        return p


@pytest.fixture(scope="module")
def game():
    abstraction = CardAbstraction(preflop_buckets=4, postflop_buckets=4,
                                  samples=300, equity_samples=30,
                                  strength="made_hand").fit(np.random.default_rng(0))
    # (4, 2, 1): the third raise of a street is all-in only, as on the deep rungs.
    return NoLimitHoldem(abstraction, starting_stack=STACK, small_blind=1, big_blind=2,
                         raise_cap=(4, 2, 1), equity_samples=30)


def facing_reraise(game, hole=((0, 13), (25, 38))):
    """SB opens pot, BB re-raises 2x pot: SB (seat 0) to make the third raise."""
    state = game.next_state(game.initial_state(), (hole, ()))
    state = game.next_state(state, RAISE_POT)
    return game.next_state(state, RAISE_TWO)


def arena(entries, phase, bot_stack, lbr_stack):
    """The arena's turn_request for the bot (seat 1), as `chipzen.bridge.replay` reads it."""
    history = [{"seat": 0, "action": "post_small_blind", "amount": 1, "phase": "preflop"},
               {"seat": 1, "action": "post_big_blind", "amount": 2, "phase": "preflop"}]
    history += [{"seat": s, "action": a, "amount": n, "phase": p} for s, a, n, p in entries]
    return {"action_history": history, "phase": phase,
            "your_stack": bot_stack, "opponent_stacks": [lbr_stack]}


# ---------------------------------------------------------------------------

def test_off_reproduces_the_result_from_before_the_change(game):
    """
    Pinned from the code before this option existed, same game, same seed:
    mean 8.95, stderr 13.418474805345609. Off must not move a single draw.
    """
    result = lbr_value(game, ShyOfShoves(game), hands=60, rng=np.random.default_rng(5),
                       rollout_samples=10, candidates=8)
    assert result.mean == 8.95
    assert result.stderr == 13.418474805345609


def test_probes_change_nothing_where_the_tree_has_no_all_in_only_menu():
    """A one-raise tree never offers all-in alone, so probes on and off are the same run."""
    abstraction = CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=300,
                                  equity_samples=30, strength="made_hand").fit(np.random.default_rng(0))
    flat = NoLimitHoldem(abstraction, starting_stack=STACK, raise_cap=1, equity_samples=30)
    strategy = ShyOfShoves(flat)
    off = lbr_value(flat, strategy, hands=40, rng=np.random.default_rng(3), rollout_samples=10, candidates=8)
    on = lbr_value(flat, strategy, hands=40, rng=np.random.default_rng(3), rollout_samples=10, candidates=8,
                   offtree_third_raise=PROBES)
    assert (off.mean, off.stderr) == (on.mean, on.stderr)


def test_probes_are_offered_only_where_the_menu_is_all_in_only(game):
    lbr = LocalBestResponse(game, ShyOfShoves(game), rollout_samples=10, candidates=8,
                            offtree_third_raise=PROBES)
    state = facing_reraise(game)
    assert tuple(game.legal_actions(state)) == (FOLD, CHECK_CALL, ALL_IN)
    rng = np.random.default_rng(0)
    moves, _ = lbr._candidates(state, 0, lbr._deal_range(state, 0, rng), rng)
    assert [m.fraction for m in moves if m.probe] == list(PROBES)

    # One raise in, the menu is 2x pot and all-in: LBR's ordinary sizes cover it.
    opened = game.next_state(game.next_state(game.initial_state(), (((0, 13), (25, 38)), ())), RAISE_POT)
    moves, _ = lbr._candidates(opened, 1, lbr._deal_range(opened, 1, rng), rng)
    assert not any(m.probe for m in moves)

    # And with the option off, the all-in-only node has no probe either.
    plain = LocalBestResponse(game, ShyOfShoves(game), rollout_samples=10, candidates=8)
    moves, _ = plain._candidates(state, 0, plain._deal_range(state, 0, rng), rng)
    assert not any(m.probe for m in moves)


def test_a_probe_that_would_be_all_in_is_not_offered():
    """Short, 1.0 pot already costs the stack: that is the shove LBR always had."""
    abstraction = CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=300,
                                  equity_samples=30, strength="made_hand").fit(np.random.default_rng(0))
    short = NoLimitHoldem(abstraction, starting_stack=80, raise_cap=(4, 2, 1), equity_samples=30)
    lbr = LocalBestResponse(short, ShyOfShoves(short), rollout_samples=10, candidates=8,
                            offtree_third_raise=PROBES)
    state = facing_reraise(short)
    rng = np.random.default_rng(0)
    moves, _ = lbr._candidates(state, 0, lbr._deal_range(state, 0, rng), rng)
    offered = [m.fraction for m in moves if m.probe]
    assert offered == [f for f in PROBES if short._raise_cost(state, 0, f) < state.stacks[0]]
    assert 1.0 not in offered


def test_the_bot_hears_what_the_bridge_reads_and_the_chips_are_real(game):
    """
    The same hand through LBR and through `chipzen.bridge.replay`: the key the
    bot answers, and after the call the re-read history, must be the bridge's.
    """
    lbr = LocalBestResponse(game, ShyOfShoves(game), rollout_samples=10, candidates=8,
                            offtree_third_raise=PROBES)
    lbr._bridge = None
    state = facing_reraise(game)
    assert state.contributions == (6, 30)

    probed = lbr._apply_move(state, Move(None, 0.4, probe=True), np.random.default_rng(0))
    # Real chips: call 24, then 0.4 of the 60 pot after the call.
    assert probed.contributions == (54, 30)
    assert probed.stacks[0] == STACK - 54 > 0
    hand = replay(arena([(0, "raise", 6, "preflop"), (1, "raise", 30, "preflop"),
                         (0, "raise", 54, "preflop")], "preflop", STACK - 30, STACK - 54),
                  seat=1, rng=np.random.default_rng(0), schedule=game.raise_cap)
    assert probed.history == hand.node.history == "345"
    assert tuple(game.legal_actions(probed)) == (FOLD, CHECK_CALL)

    # A fold pays LBR the 30 the bot put in, not anything sized off a shove.
    folded = game.next_state(probed, FOLD)
    assert game.is_terminal(folded) and game.utility(folded, 0) == 30.0

    # A call closes the street with both stacks live, and the hand goes on.
    called = game.next_state(probed, CHECK_CALL)
    assert called.contributions == (54, 54)
    lbr._street_closes(called)
    flop = game.next_state(called, (2, 3, 4))
    hand = replay(arena([(0, "raise", 6, "preflop"), (1, "raise", 30, "preflop"),
                         (0, "raise", 54, "preflop"), (1, "call", 24, "preflop")],
                        "flop", STACK - 54, STACK - 54),
                  seat=1, rng=np.random.default_rng(0), schedule=game.raise_cap)
    assert flop.history == hand.node.history == "3451/"
    assert lbr._alt_history(flop.history) == hand.alt_history == "341/"

    # The bot is asked on the true line first and, where the tree ended it, on the re-read one.
    width = len(game.legal_actions(flop))
    table = {"2|341/": np.full(width, 1.0 / width)}
    lbr.strategy = table
    assert lbr._policy(2, flop.history, width) is table["2|341/"]


def test_the_probe_is_priced_on_its_real_size(game):
    """At the constructed node, the probe's priced state carries 54 chips, not a stack."""
    lbr = LocalBestResponse(game, ShyOfShoves(game), rollout_samples=10, candidates=8,
                            offtree_third_raise=(0.4,))
    state = facing_reraise(game)
    perceived = lbr._bridge_perceive(state, 0, 0.4, lbr._bridge_hand(0), np.random.default_rng(0))
    assert perceived == ALL_IN
    after = game.raise_by_fraction(state, 0.4, perceived)
    shove = game.next_state(state, ALL_IN)
    assert after.contributions[0] == 54 and shove.contributions[0] == STACK


def test_against_a_strategy_that_folds_to_shoves_probes_find_more(game):
    """
    One decision, then whole hands on paired seeds. The strategy folds three
    quarters of its range to anything read as all-in, so a 0.4 pot raise wins
    the same folds as a shove and risks a ninth of it when called.
    """
    strategy = ShyOfShoves(game)
    state = facing_reraise(game, hole=((5, 13), (25, 38)))     # LBR holds 7h 2d; the bot's cards are never read
    values = {}
    for probes in ((), PROBES):
        lbr = LocalBestResponse(game, strategy, rollout_samples=40, candidates=16,
                                offtree_third_raise=probes)
        rng = np.random.default_rng(9)
        moves, vals = lbr._candidates(state, 0, lbr._deal_range(state, 0, rng), rng)
        values[probes] = (moves, vals)
    off_best = max(values[()][1])
    on_moves, on_vals = values[PROBES]
    assert max(on_vals) > off_best
    assert on_moves[int(np.argmax(on_vals))].probe

    off = LocalBestResponse(game, strategy, rollout_samples=10, candidates=8).play(300, paired_seed=11)
    lbr_on = LocalBestResponse(game, strategy, rollout_samples=10, candidates=8, offtree_third_raise=PROBES)
    on = lbr_on.play(300, paired_seed=11)
    difference = on.values - off.values
    assert lbr_on.probe_stats["probes"] > 0
    # Hands where no probe changed anything are identical, which is what pairing buys.
    assert (difference == 0).sum() > 250
    assert difference.mean() > 0 and on.mean > off.mean
