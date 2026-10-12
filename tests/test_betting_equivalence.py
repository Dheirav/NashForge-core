"""
The two implementations of no-limit betting, checked against each other.

This project has poker's betting written twice. `engine.PokerGame` is the one
the audit found sound and the one every published figure is measured in — the
panel, Phase 4, the crossover, the GUI, both Slumbot runs. `games/nolimit.py`
is a second implementation, rebuilt deliberately and for a good reason, which
its own docstring gives:

    **Rebuilt:** the betting. ``engine.PokerGame`` mutates state in place and
    consumes a dealt deck, so a traverser cannot branch over an action and back
    up.

CFR needs to branch; the engine cannot. So the solver is *trained* in the
traversal game and *scored* in the engine, and nothing has ever checked that the
two agree about chips.

Why this matters more than it sounds
------------------------------------
If the two disagree, every CFR strategy here was fitted for a game slightly
different from the one it is measured in. That would not make the numbers wrong,
but it would change what they are numbers about — including the abstraction
crossover, which is this project's headline result.

It surfaced from a discrepancy: `train_nolimit.py` reported the 150,000-iteration
solver *worse* against always-call than the 4,000-iteration one, where
`evaluation.benchmark` reported it better. Two paths, opposite conclusions, same
two strategies. The first guess was a lookup bug in `cfr/play.py`; measuring it
showed that guard never fires. The real difference is that the two paths play
different implementations.

Fixed on 2 September
--------------------
`training/fitness.py` sized a pot-fraction raise off the pot *before* the call
and `games/nolimit.py` off the pot after it, so every raise in the engine was
about 20% smaller than the same abstract action in the game the solver trains
in. The engine now follows the traversal game's convention, which is the
standard one. These tests are what found it and what keep it fixed.

What is compared, and what is not
----------------------------------
Only the betting, which is the part that was rebuilt. Both are driven through
identical abstract action sequences and their chip state compared at every step:
pot, per-player contributions, and stacks. Cards are irrelevant to that and are
left out, so a disagreement here is arithmetic rather than dealing.

Showdown and card handling are not covered. The engine's are audited; the
traversal game's are checked by CFR reproducing Kuhn's −1/18 and exact Leduc
exploitability, which would not happen if its payoffs were wrong.
"""
import numpy as np
import pytest

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD
from abstraction.buckets import CardAbstraction
from engine import PokerGame, get_abstract_action_mask
from evaluation.benchmark import _constrain
from games.nolimit import NoLimitHoldem
from training.fitness import abstract_action_to_engine_action

STACK, SMALL_BLIND, BIG_BLIND, RAISE_CAP = 200, 1, 2, 1


@pytest.fixture(scope="module")
def traversal():
    """The traversal game. Its abstraction is irrelevant to betting arithmetic."""
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4)
    abstraction.fit(np.random.default_rng(0))
    return NoLimitHoldem(abstraction, starting_stack=STACK,
                         big_blind=BIG_BLIND, raise_cap=RAISE_CAP,
                         equity_samples=4)


def engine_chips(actions):
    """
    Drive the engine through `actions`; return its chip state, or None if the
    sequence ends the hand early.
    """
    game = PokerGame([STACK, STACK], small_blind=SMALL_BLIND,
                     big_blind=BIG_BLIND, seed=7, enable_history=False)
    street, raises = game.state.betting_round, 0
    for action in actions:
        if game.is_hand_over():
            return None
        player = game.state.current_player
        if player is None:
            return None
        if game.state.betting_round != street:
            street, raises = game.state.betting_round, 0
        to_call = game.current_bet - game.players[player].bet
        mask = _constrain(get_abstract_action_mask(game, player), to_call,
                          raises, RAISE_CAP)
        if not mask[action]:
            return None                      # not legal here; nothing to compare
        if action in (2, 3, 4, ALL_IN):
            raises += 1
        game.apply_action(player, abstract_action_to_engine_action(action, game, player))
    # Pot and stacks only. The per-street bet field is deliberately excluded:
    # the engine clears it at a street boundary and at the end of a hand, the
    # traversal game carries it, and comparing the two compares reset semantics
    # rather than arithmetic. Stacks already say what each player has paid.
    return (game.state.pot.total, tuple(p.stack for p in game.players))


def traversal_chips(game, actions):
    """The same sequence through the traversal game, or None if it does not fit."""
    state = game.initial_state()
    # Deal, so the root stops being a chance node. Which cards is immaterial
    # here: nothing below showdown reads them.
    state = state._replace(hole=((0, 1), (2, 3)))
    for action in actions:
        if game.is_terminal(state):
            return None
        if game.current_player(state) < 0:
            return None                      # a chance node: street change
        if action not in game.legal_actions(state):
            return None
        state = game.next_state(state, action)
    return (sum(state.contributions), tuple(state.stacks))


def sequences(depth):
    """Every action sequence up to `depth`, the betting tree enumerated."""
    if depth == 0:
        yield ()
        return
    for head in sequences(depth - 1):
        for action in range(6):
            yield head + (action,)


def survey(traversal, depth):
    """Every comparable sequence at `depth`, split by whether the two agree."""
    agree, differ = [], []
    for actions in sequences(depth):
        engine = engine_chips(actions)
        if engine is None:
            continue
        theirs = traversal_chips(traversal, actions)
        if theirs is None:
            continue
        (differ if engine != theirs else agree).append((actions, engine, theirs))
    return agree, differ


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_two_betting_implementations_agree_on_chips(traversal, depth):
    """
    Same actions in, same chips out, over the enumerated betting tree.

    They did not, until 2 September. `training/fitness.py` sized a pot-fraction
    raise off `game.state.pot.total`, the pot *before* the call, where
    `games/nolimit.py` sizes it off the pot after -- the standard convention, in
    which a pot-sized raise means calling first and then betting the pot
    including your call. Every raise in the engine was therefore about 20%
    smaller than the same abstract action in the game the solver trains in, and
    a CFR strategy was scored making bets it had not been fitted for.

    Everything else agreed to the chip even then, which is what made the
    divergence hard to notice: fold, check, call and all-in were never wrong.

    Sequences either implementation rejects are skipped rather than asserted
    about. The point is to compare where both agree a line is legal, not to
    re-check the legal-action rules, which both take from `abstraction.betting`.
    """
    agree, differ = survey(traversal, depth)
    assert agree, f"no comparable sequences at depth {depth}"
    assert not differ, (
        f"{len(differ)} of {len(agree) + len(differ)} sequences disagree at "
        f"depth {depth}. First: actions={differ[0][0]} "
        f"engine(pot,stacks)={differ[0][1]} traversal={differ[0][2]}")


def test_the_blinds_are_posted_the_same_way(traversal):
    """The root, before anyone acts. If this differs, everything after it does."""
    engine = engine_chips(())
    theirs = traversal_chips(traversal, ())
    assert engine == theirs, f"engine={engine} traversal={theirs}"


def test_a_call_costs_the_same_in_both(traversal):
    """The simplest line there is, isolated so a failure names itself."""
    assert engine_chips((CHECK_CALL,)) == traversal_chips(traversal, (CHECK_CALL,))


def test_folding_leaves_the_same_chips_behind(traversal):
    engine = engine_chips((FOLD,))
    theirs = traversal_chips(traversal, (FOLD,))
    if engine is not None and theirs is not None:
        assert engine == theirs
