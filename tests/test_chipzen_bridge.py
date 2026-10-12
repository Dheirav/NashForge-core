"""
The Chipzen bridge, checked against the protocol's own worked hand.

Every amount below is taken from section 4 of POKER-GAME-STATE-PROTOCOL.md:
stacks of 1,000, blinds 5 and 10, seat 0 the dealer and small blind. A bridge
bug is indistinguishable from a weak strategy from the outside, so the numbers
here are the arena's, not this project's.
"""
import numpy as np
import pytest

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, RAISE_HALF, RAISE_POT, RAISE_TWO
from chipzen.bridge import cards, legal_mask, replay, to_chipzen
from abstraction.betting import ALL_IN, RAISE_POT  # noqa: E402,F811
from chipzen.bridge import Node  # noqa: E402

SB = {"seat": 0, "action": "post_small_blind", "amount": 5, "phase": "preflop", "is_timeout": False}
BB = {"seat": 1, "action": "post_big_blind", "amount": 10, "phase": "preflop", "is_timeout": False}


def entry(seat, action, amount, phase):
    return {"seat": seat, "action": action, "amount": amount, "phase": phase,
            "is_timeout": False}


def state(seat, phase, history, your_stack, opp_stack, pot, to_call,
          min_raise=0, max_raise=0, board=(), hole=("As", "Kh")):
    return {
        "hand_number": 1, "phase": phase, "board": list(board),
        "your_hole_cards": list(hole), "pot": pot, "your_stack": your_stack,
        "opponent_stacks": [opp_stack], "to_call": to_call,
        "min_raise": min_raise, "max_raise": max_raise,
        "action_history": [SB, BB] + list(history),
    }


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# Message 4: seat 0 to act preflop, only the blinds in.
def test_the_pot_starts_at_the_blinds(rng):
    hand = replay(state(0, "preflop", [], 995, 990, 15, 5, 20, 995), 0, rng)
    assert hand.node.pot == 15
    assert hand.node.committed == [5, 10]
    assert hand.node.to_call == 5
    assert hand.node.history == ""
    assert hand.node.to_act == 0


def test_start_stacks_are_recovered_from_now_plus_contributions(rng):
    hand = replay(state(0, "preflop", [], 995, 990, 15, 5), 0, rng)
    assert hand.start_stacks == [1000, 1000]
    assert hand.big_blind == 10
    assert hand.effective_bb == 100.0


# Message 7: seat 1 to act after seat 0 raised to 30.
def test_a_raise_to_a_level_costs_the_difference(rng):
    hand = replay(state(1, "preflop", [entry(0, "raise", 30, "preflop")],
                        990, 970, 40, 20, 50, 990), 1, rng)
    assert hand.node.pot == 40
    assert hand.node.committed == [30, 10]
    assert hand.node.to_call == 20
    assert hand.start_stacks == [1000, 1000]


def test_a_pot_sized_open_reads_as_the_pot_raise(rng):
    # Raise to 30 from the small blind: call 5 into 15, then 20 on top of 20.
    hand = replay(state(1, "preflop", [entry(0, "raise", 30, "preflop")],
                        990, 970, 40, 20), 1, rng)
    assert hand.node.history == str(RAISE_POT)


# Message 11: the flop, seat 1 (big blind) to act first.
def test_a_new_street_clears_committed_keeps_the_pot_and_adds_a_slash(rng):
    # A call's amount is what it adds: the big blind calls 20 to match a raise to 30.
    hand = replay(state(1, "flop", [entry(0, "raise", 30, "preflop"),
                                    entry(1, "call", 20, "preflop")],
                        970, 970, 60, 0, 10, 970, board=("Qs", "7h", "3d")), 1, rng)
    assert hand.node.pot == 60
    assert hand.node.committed == [0, 0]
    assert hand.node.prior == [30, 30]
    assert hand.node.street == 1
    assert hand.node.history == f"{RAISE_POT}{CHECK_CALL}/"


def test_the_phase_field_advances_a_street_the_history_has_not_reached(rng):
    # Same as above: no flop action yet, the phase alone says we are on it.
    hand = replay(state(1, "flop", [entry(0, "raise", 30, "preflop"),
                                    entry(1, "call", 30, "preflop")],
                        970, 970, 60, 0), 1, rng)
    assert hand.node.history.endswith("/")
    assert hand.node.raises_this_street == 0


# Message 17: seat 1 facing a 40 bet on the flop after checking.
def test_a_bet_into_an_unopened_pot_is_a_fraction_of_that_pot(rng):
    hand = replay(state(1, "flop", [entry(0, "raise", 30, "preflop"),
                                    entry(1, "call", 20, "preflop"),
                                    entry(1, "check", 0, "flop"),
                                    entry(0, "raise", 40, "flop")],
                        970, 930, 100, 40, 80, 970, board=("Qs", "7h", "3d")), 1, rng)
    assert hand.node.pot == 100
    assert hand.node.to_call == 40
    assert hand.node.raises_this_street == 1
    street = hand.node.history.split("/")[1]
    assert street[0] == str(CHECK_CALL)
    # 40 into 60 is two thirds of the pot: between half and full, never beyond.
    assert street[1] in (str(RAISE_HALF), str(RAISE_POT))


def test_a_re_raise_is_measured_against_the_pot_after_the_call(rng):
    # Seat 0 opens to 30; seat 1 re-raises to 90. Calling 20 makes the pot 60,
    # and 60 more on top of that is exactly one pot.
    hand = replay(state(0, "preflop", [entry(0, "raise", 30, "preflop"),
                                       entry(1, "raise", 90, "preflop")],
                        970, 910, 120, 60), 0, rng, schedule=2)
    assert hand.node.history == f"{RAISE_POT}{RAISE_POT}"
    assert hand.node.raises_this_street == 2


def test_a_re_raise_is_translated_onto_the_sizes_the_schedule_allows(rng):
    # Under (4, 2) the second raise may only be two times pot or all-in, so a
    # small re-raise must land on RAISE_TWO rather than on a size the solver
    # never stored at that depth.
    hand = replay(state(0, "preflop", [entry(0, "raise", 30, "preflop"),
                                       entry(1, "raise", 70, "preflop")],
                        970, 930, 100, 40), 0, rng, schedule=(4, 2))
    assert hand.node.history == f"{RAISE_POT}{RAISE_TWO}"


def test_a_bet_to_everything_the_bettor_has_is_the_all_in_action(rng):
    # Seat 1 started with 1,000 and bets to 1,000 preflop.
    hand = replay(state(0, "preflop", [entry(0, "raise", 30, "preflop"),
                                       entry(1, "raise", 1000, "preflop")],
                        970, 0, 1030, 970), 0, rng, schedule=2)
    assert hand.node.history[-1] == str(ALL_IN)
    assert hand.node.misses == 0


def test_a_bet_beyond_the_abstraction_is_counted_and_read_as_a_shove(rng):
    # 40 into a pot of 15 after calling 5: (35 - 5) / 20 = 1.5 pots is fine;
    # 200 into it is (195 - 5) / 20 = 9.5 pots, beyond the largest size.
    hand = replay(state(1, "preflop", [entry(0, "raise", 200, "preflop")],
                        990, 800, 210, 190), 1, rng)
    assert hand.node.history == str(ALL_IN)
    assert hand.node.misses == 1


def test_cards_convert_to_the_engine_s_own_type(rng):
    hole, board = cards(state(0, "flop", [], 0, 0, 0, 0, board=("Qs", "7h", "3d")))
    assert [c.rank for c in hole] == ["A", "K"]
    assert [c.suit for c in board] == ["s", "h", "d"]


# Outgoing.
def test_a_pot_raise_from_the_small_blind_is_the_protocol_s_own_thirty(rng):
    s = state(0, "preflop", [], 995, 990, 15, 5, 20, 995)
    hand = replay(s, 0, rng)
    assert to_chipzen(RAISE_POT, hand.node, s) == {"action": "raise", "params": {"amount": 30}}


def test_passive_actions_pick_the_legal_word(rng):
    s = state(0, "preflop", [], 995, 990, 15, 5, 20, 995)
    hand = replay(s, 0, rng)
    assert to_chipzen(CHECK_CALL, hand.node, s) == {"action": "call", "params": {}}
    assert to_chipzen(FOLD, hand.node, s) == {"action": "fold", "params": {}}
    s2 = state(1, "flop", [entry(0, "raise", 30, "preflop"), entry(1, "call", 30, "preflop")],
               970, 970, 60, 0, 10, 970)
    assert to_chipzen(CHECK_CALL, replay(s2, 1, rng).node, s2) == {"action": "check", "params": {}}


def test_all_in_goes_out_as_max_raise_and_a_raise_never_exceeds_it(rng):
    s = state(0, "preflop", [], 995, 990, 15, 5, 20, 995)
    hand = replay(s, 0, rng)
    assert to_chipzen(ALL_IN, hand.node, s)["params"]["amount"] == 995
    s["max_raise"] = 25
    assert to_chipzen(RAISE_TWO, hand.node, s)["params"]["amount"] == 25


def test_a_raise_below_the_minimum_is_lifted_to_it(rng):
    s = state(1, "flop", [entry(0, "raise", 30, "preflop"), entry(1, "call", 30, "preflop")],
              970, 970, 60, 0, 100, 970)
    hand = replay(s, 1, rng)
    assert to_chipzen(RAISE_HALF, hand.node, s)["params"]["amount"] == 100


# The mask.
def test_the_arena_s_valid_actions_override_the_tree(rng):
    s = state(0, "preflop", [], 995, 990, 15, 5)
    node = replay(s, 0, rng).node
    mask = legal_mask(node, ["fold", "call"])
    assert mask[FOLD] == 1 and mask[CHECK_CALL] == 1
    assert not mask[2:].any()


def test_folding_is_removed_when_there_is_nothing_to_call(rng):
    s = state(1, "flop", [entry(0, "raise", 30, "preflop"), entry(1, "call", 30, "preflop")],
              970, 970, 60, 0)
    node = replay(s, 1, rng).node
    mask = legal_mask(node, ["check", "raise"])
    assert mask[FOLD] == 0
    assert mask[CHECK_CALL] == 1 and mask[RAISE_POT] == 1


def test_the_schedule_removes_raises_the_solver_never_saw(rng):
    s = state(0, "preflop", [entry(0, "raise", 30, "preflop"),
                             entry(1, "raise", 90, "preflop")], 970, 910, 120, 60)
    node = replay(s, 0, rng, schedule=1).node
    mask = legal_mask(node, ["fold", "call", "raise"], schedule=1)
    assert not mask[2:].any()
    node = replay(s, 0, rng, schedule=(4, 2)).node
    mask = legal_mask(node, ["fold", "call", "raise"], schedule=(4, 2))
    assert not mask[2:].any(), "two raises in, (4, 2) is exhausted too"



# --- the call convention, and the short-stack technicalities (15 September) ---

def test_a_call_s_amount_is_the_increment_not_the_level(rng):
    """
    Hand 4 against an arena opponent: blinds 50/100, they raise to 260, we
    call 160. Every one of 3,128 logged calls carries the increment. Read as a
    level, the pot was 420 instead of 520 and the flop's pot-sized bet was
    keyed as two times pot: a line never taken, in a quarter of decisions.
    """
    from chipzen.bridge import _contributions
    history = [{"seat": 1, "action": "post_small_blind", "amount": 50, "phase": "preflop"},
               {"seat": 0, "action": "post_big_blind", "amount": 100, "phase": "preflop"},
               {"seat": 1, "action": "raise", "amount": 260, "phase": "preflop"},
               {"seat": 0, "action": "call", "amount": 160, "phase": "preflop"},
               {"seat": 0, "action": "raise", "amount": 520, "phase": "flop"}]
    st = {"hand_number": 4, "phase": "flop", "board": ["Ad", "8s", "Qs"], "your_hole_cards": ["9d", "Qc"],
          "pot": 1040, "your_stack": 9740, "opponent_stacks": [9220], "to_call": 520,
          "min_raise": 1040, "max_raise": 9740, "action_history": history}
    hand = replay(st, 1, rng)
    assert _contributions(history) == [780, 260]
    assert hand.node.pot == 1040 and hand.node.to_call == 520
    assert hand.node.history.startswith("31/3")       # a pot-sized bet reads as pot, not 2x
    assert hand.effective_bb == pytest.approx(min(9740 + 260, 9220 + 780) / 100)


def test_a_short_blind_post_does_not_set_the_level(rng):
    """35 logged hands had a big blind posted short; the depth read collapsed to 1bb."""
    st = state(1, "preflop", [], 24, 5000, 74, 26)
    st["action_history"] = [{"seat": 0, "action": "post_small_blind", "amount": 50, "phase": "preflop"},
                            {"seat": 1, "action": "post_big_blind", "amount": 24, "phase": "preflop"}]
    hand = replay(st, 1, rng)
    assert hand.big_blind == 100


def test_the_raise_ceiling_wins_over_the_floor():
    """Short, min_raise can exceed max_raise; sending min_raise was more than our stack."""
    node = Node(pot=300, committed=[0, 100], to_act=0)
    out = to_chipzen(RAISE_POT, node, {"pot": 300, "to_call": 100, "min_raise": 200, "max_raise": 150})
    assert out == {"action": "raise", "params": {"amount": 150}}
    out = to_chipzen(ALL_IN, node, {"pot": 300, "to_call": 100, "min_raise": 0, "max_raise": 0})
    assert out == {"action": "call", "params": {}}


def test_an_unknown_post_is_dead_money_and_an_unknown_verb_is_passive(rng):
    st = state(1, "preflop", [], 990, 990, 20, 0)
    st["action_history"] = st["action_history"] + [{"seat": 0, "action": "post_ante_x", "amount": 5, "phase": "preflop"},
                                                   {"seat": 0, "action": "timeout", "amount": 0, "phase": "preflop"}]
    hand = replay(st, 1, rng)                      # no TranslationError
    assert hand.node.pot == 20
    assert hand.node.history.endswith(str(CHECK_CALL))


def test_a_pseudo_all_in_that_closed_a_street_is_re_read_as_the_largest_sized_raise(rng):
    # A burst hand at 100bb: SB opens 250, BB 3-bets 1,250,
    # SB 4-bets 2,250 with 7,750 behind, BB calls (scaled here to the fixture's 5/10 blinds); the flop is dealt. Under the
    # (4, 2, 1) taper the third raise can only be read as all-in, so the flop
    # keyed a history the tree treats as terminal and the rule played the
    # hand. With the street closed and the hand continuing, the 4-bet is
    # re-read: the taper has no sized third raise, so it collapses into a
    # call and the flop keys the post-3-bet node.
    hist = [entry(0, "raise", 25, "preflop"), entry(1, "raise", 125, "preflop"),
            entry(0, "raise", 225, "preflop"), entry(1, "call", 100, "preflop")]
    facing = replay(state(1, "preflop", hist[:3], 875, 775, 350, 100), 1, rng, schedule=(4, 2, 1))
    assert facing.node.history.endswith(str(ALL_IN)), "facing the 4-bet it is still a fold-or-call spot"
    assert facing.pseudo_allins and facing.collapsed == 0
    flop = replay(state(1, "flop", hist, 775, 775, 450, 0, board=("2c", "7d", "Ts")), 1, rng, schedule=(4, 2, 1))
    # The true history keeps the all-in; the re-read one is offered beside it.
    # The open's size is translated stochastically, so only the tails are pinned.
    assert flop.node.history[1:] == "4" + str(ALL_IN) + str(CHECK_CALL) + "/", flop.node.history
    assert flop.alt_history[1:] == "4" + str(CHECK_CALL) + "/", flop.alt_history
    assert flop.collapsed == 1 and flop.node.pot == 450
    # Under a schedule with a sized third raise the same hand keys that raise.
    flop2 = replay(state(1, "flop", hist, 775, 775, 450, 0, board=("2c", "7d", "Ts")), 1, rng, schedule=3)
    assert flop2.node.history.endswith(str(CHECK_CALL) + "/") and flop2.collapsed == 0 and flop2.alt_history is None
    # A real all-in is left alone: the hand did end.
    allin = [entry(0, "raise", 25, "preflop"), entry(1, "raise", 125, "preflop"),
             entry(0, "raise", 1000, "preflop"), entry(1, "call", 875, "preflop")]
    ended = replay(state(1, "flop", allin, 0, 0, 2000, 0, board=("2c", "7d", "Ts")), 1, rng, schedule=(4, 2, 1))
    assert ended.node.history[1:] == "4" + str(ALL_IN) + str(CHECK_CALL) + "/" and ended.alt_history is None
