"""
Review tests for `bridge_translation` (branch lbr-overbet), written to cover what
tests/test_lbr_overbet.py leaves out: postflop and re-raise readings checked
against `chipzen.bridge.replay` on a grid of sizes around the 1.5x line, every
rung schedule the ladders use (one of which puts the line at 1.5 pot, not 3.0),
chip rounding, stack caps, the all-in-only depth, the re-read lookup taken (and
missed) in a played hand, the counters with both options on, and off-arm pins
computed from HEAD's cfr/lbr.py.

    venv/bin/python -m pytest tests/test_review_lbr_overbet.py -q
"""
import numpy as np
import pytest

from abstraction.betting import (ALL_IN, CHECK_CALL, FOLD, RAISE_HALF, RAISE_POT,
                                 RAISE_TWO, StreetSchedule, raise_sizes_at)
from abstraction.buckets import CardAbstraction
from abstraction.translation import translation_distribution
from cfr.lbr import DEFAULT_BET_SIZES, LocalBestResponse, Move, lbr_value
from chipzen.bridge import replay
from games.nolimit import RAISE_FRACTION, NoLimitHoldem
from test_lbr_thirdraise import ShyOfShoves

BETWEEN = (0.33, 0.66, 1.33, 2.66, 5.33)
HOLE = ((0, 13), (25, 38))
PHASES = ("preflop", "flop", "turn", "river")
FLOP, TURN, RIVER = (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)

#: Every raise schedule the ladder169l rungs were solved under (cap2_*.json,
#: surveyed 5 Oct). "ns" has pot as its largest size, so its line is 1.5 pot.
SCHEDULES = {
    "uniform2": 2,
    "t421": (4, 2, 1),
    "t432": (4, 3, 2),
    "ns": ((2, 3, 5), (2, 3, 5), (5,)),
    "tapered_a": StreetSchedule((((2, 3, 4, 5), (2, 4, 5), (5,)),) * 3
                            + (((2, 3, 4, 5), (4, 5), (5,)),)),
}


@pytest.fixture(scope="module")
def abstraction():
    return CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=300,
                           equity_samples=30, strength="made_hand").fit(np.random.default_rng(0))


def make(abstraction, cap=(4, 2, 1), stack=400):
    return NoLimitHoldem(abstraction, starting_stack=stack, small_blind=1, big_blind=2,
                         raise_cap=cap, equity_samples=30)


@pytest.fixture(scope="module")
def game(abstraction):
    return make(abstraction)


class Line:
    """
    A hand walked through the game and, alongside, the arena's action list for it,
    so `replay` can be asked what it reads at the same chips.
    """

    def __init__(self, game):
        self.game = game
        self.state = game.next_state(game.initial_state(), (HOLE, ()))
        self.entries = []

    def act(self, action):
        g, s = self.game, self.state
        seat = g.current_player(s)
        phase = PHASES[s.street]
        after = g.next_state(s, action)
        if action == CHECK_CALL:
            to_call = s.committed[1 - seat] - s.committed[seat]
            self.entries.append((seat, "call", to_call, phase) if to_call > 0
                                else (seat, "check", 0, phase))
        else:
            self.entries.append((seat, "raise", after.committed[seat], phase))
        self.state = after
        return self

    def deal(self, board):
        self.state = self.game.next_state(self.state, board)
        return self

    def replay_of(self, me, fraction, rng):
        """The history `replay` builds after ``me`` raises ``fraction`` here."""
        g, s = self.game, self.state
        after = g.raise_by_fraction(s, fraction, ALL_IN)       # chips only
        entries = self.entries + [(me, "raise", after.committed[me], PHASES[s.street])]
        bot = 1 - me
        history = [{"seat": 0, "action": "post_small_blind", "amount": 1, "phase": "preflop"},
                   {"seat": 1, "action": "post_big_blind", "amount": 2, "phase": "preflop"}]
        history += [{"seat": a, "action": b, "amount": n, "phase": p} for a, b, n, p in entries]
        request = {"action_history": history, "phase": PHASES[s.street],
                   "your_stack": after.stacks[bot], "opponent_stacks": [after.stacks[me]]}
        return replay(request, seat=bot, rng=rng, schedule=g.raise_cap)


class Same:
    """A generator stand-in whose every uniform draw is ``u``."""

    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def spots(game):
    """Open preflop, re-raise preflop, bet the flop, raise the flop."""
    yield "open", Line(game)
    yield "reraise", Line(game).act(RAISE_POT)
    yield "flop bet", Line(game).act(CHECK_CALL).act(CHECK_CALL).deal(FLOP)
    flop = Line(game).act(CHECK_CALL).act(CHECK_CALL).deal(FLOP)
    if RAISE_POT in game.legal_actions(flop.state):
        yield "flop raise", flop.act(RAISE_POT)


def line_for(game, state):
    """1.5 times the largest sized raise the schedule has here, or None."""
    depth = sum(1 for a in state.history.split("/")[-1] if a in "2345")
    sized = [a for a in raise_sizes_at(game.raise_cap, depth, state.street) if a != ALL_IN]
    return 1.5 * RAISE_FRACTION[sized[-1]] if sized else None


GRID = tuple(sorted(set(np.round(np.arange(0.1, 6.01, 0.1), 2).tolist()
                        + [1.49, 1.5, 1.51, 2.66, 2.75, 2.9, 2.99, 3.0, 3.01, 3.5, 5.33])))


# ---------------------------------------------------------------------------
# What the bot is asked: the label LBR writes equals replay's, everywhere.

@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_every_size_reads_as_replay_reads_it_on_every_schedule(abstraction, name):
    """
    Same chips, same generator seed: LBR's label must be the one replay builds.
    Postflop and at a raise as well as the preflop open, on every schedule a
    ladder rung uses, for a grid that straddles the line on each of them.
    """
    game = make(abstraction, SCHEDULES[name])
    for spot, line in spots(game):
        me = game.current_player(line.state)
        for fraction in GRID:
            for seed in (0, 1, 2):
                lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
                heard = lbr._apply_move(line.state, Move(None, fraction), np.random.default_rng(seed))
                # replay also translates the bot's own earlier raise, which can
                # spend a draw; every draw it makes is LBR's first one, and the
                # earlier raise is an exact tree size, so only LBR's raise uses it.
                hand = line.replay_of(me, fraction, Same(np.random.default_rng(seed).random()))
                assert heard.history == hand.node.history, (name, spot, fraction, seed)


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_where_the_pseudo_all_in_line_sits_by_schedule(abstraction, name):
    """
    The write-up says 2.75 pot is never affected and the line is 3.0. That holds
    for every schedule whose largest sized raise is 2x pot, which is all of them
    but "ns", where the largest size is pot and so 1.6, 2.0, 2.66 and 2.75 are
    pseudo all-ins too. The test pins the line per schedule, which is the
    correct statement of the claim.
    """
    game = make(abstraction, SCHEDULES[name])
    for spot, line in spots(game):
        threshold = line_for(game, line.state)
        assert threshold == (1.5 if name == "ns" else 3.0), (name, spot)
        lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
        for fraction in (2.0, 2.75):
            heard = lbr._apply_move(line.state, Move(None, fraction), np.random.default_rng(0))
            assert (heard.history[-1] == str(ALL_IN)) == (name == "ns"), (name, spot, fraction)


def test_rounding_to_the_chip_can_push_a_size_over_the_line(game):
    """
    2.9 pot over a 4-chip pot costs round(11.6) = 12 chips, which the bridge sees
    as exactly 3.0 pot: a pseudo all-in, counted as an overbet. 2.9 over a pot of
    12 is 34.8 -> 35 chips, 2.917 pot, and stays 2x.
    """
    lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
    state = Line(game).state
    heard = lbr._apply_move(state, Move(None, 2.9), np.random.default_rng(0))
    assert heard.history == str(ALL_IN) and heard.stacks[0] > 0
    assert lbr.probe_stats["overbet_allins"] == 1
    reraise = Line(game).act(RAISE_POT).state
    heard = lbr._apply_move(reraise, Move(None, 2.9), np.random.default_rng(0))
    assert heard.history[-1] == str(RAISE_TWO)


def test_a_bet_below_the_smallest_size_and_the_big_blind_floor(game):
    """
    A 0.1 pot open costs the one big blind minimum, 0.5 pot to the bridge, which
    is the half-pot node either way. A 0.33 re-raise where only 2x and all-in
    exist is heard as 2x by both translations.
    """
    on = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
    off = LocalBestResponse(game, ShyOfShoves(game), 10, 8)
    state = Line(game).state
    assert on._bridge_view(state, 0, 0.1)[0] == 0.5
    assert on._apply_move(state, Move(None, 0.1), np.random.default_rng(0)).history == str(RAISE_HALF)
    reraise = Line(game).act(RAISE_POT).state
    for lbr in (on, off):
        assert lbr._apply_move(reraise, Move(None, 0.33), np.random.default_rng(0)).history[-1] \
            == str(RAISE_TWO)


def test_a_stack_capped_reraise_under_the_line_is_a_real_all_in(abstraction):
    """
    10bb deep, over a pot open, a 1.33 pot re-raise is under the 3.0 line but
    costs the whole stack. The bridge reads all-in (capped, no chips kept), the
    old translation 2x. Same chips both ways.
    """
    short = make(abstraction, stack=20)
    reraise = Line(short).act(RAISE_POT).state
    on = LocalBestResponse(short, ShyOfShoves(short), 10, 8, bridge_translation=True)
    heard = on._apply_move(reraise, Move(None, 1.33), np.random.default_rng(0))
    old = LocalBestResponse(short, ShyOfShoves(short), 10, 8)._apply_move(
        reraise, Move(None, 1.33), np.random.default_rng(0))
    assert heard.stacks[1] == 0 and heard.history == "35" and old.history == "34"
    assert heard.contributions == old.contributions
    assert on.probe_stats["capped_allins"] == 1 and on.probe_stats["overbet_allins"] == 0
    assert on._bridge.pseudo_allins == []


# ---------------------------------------------------------------------------
# Pricing: the distribution, and that it is what the candidates are priced on.

@pytest.mark.parametrize("name", ("t421", "ns", "tapered_a"))
def test_pricing_distribution_matches_sampling_postflop_and_by_schedule(abstraction, name):
    """The author's frequency check, extended to the flop and to two other schedules."""
    game = make(abstraction, SCHEDULES[name])
    rng = np.random.default_rng(3)
    for spot, line in spots(game):
        state = line.state
        me = game.current_player(state)
        lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bridge_translation=True)
        for fraction in (0.33, 0.66, 0.8, 1.25, 1.33, 1.6, 2.66, 3.5):
            readings, weights = lbr._bridge_distribution(state, me, fraction)
            assert abs(weights.sum() - 1.0) < 1e-12
            draws = [lbr._bridge_perceive(state, me, fraction, lbr._bridge_hand(me), rng)
                     for _ in range(1500)]
            assert set(draws) <= set(readings), (name, spot, fraction)
            for reading, weight in zip(readings, weights):
                assert abs(np.mean([d == reading for d in draws]) - weight) < 0.045, \
                    (name, spot, fraction, reading)


def test_candidates_priced_identically_where_the_two_readings_agree(game):
    """
    At the open, 0.5, 1.0, 2.0 and 2.75 pot cost round numbers of chips and both
    translations hear them the same, so each must be priced to the same value
    on and off; 3.5 is heard as all-in on and 2x off, and its value must move.
    """
    state = Line(game).state
    strategy = ShyOfShoves(game)
    values = {}
    for flag in (False, True):
        lbr = LocalBestResponse(game, strategy, 30, 16, bet_sizes=(0.5, 1.0, 2.0, 2.75, 3.5),
                                bridge_translation=flag)
        candidate_range = lbr._deal_range(state, 0, np.random.default_rng(4))
        moves, vals = lbr._candidates(state, 0, candidate_range, np.random.default_rng(9))
        values[flag] = {m.fraction: v for m, v in zip(moves, vals) if m.fraction is not None}
    for fraction in (0.5, 1.0, 2.0, 2.75):
        assert values[True][fraction] == values[False][fraction], fraction
    assert values[True][3.5] != values[False][3.5]


def test_the_all_in_only_depth_is_untouched_and_probes_still_combine(game):
    """
    The third raise under (4, 2, 1) has no sized raise: LBR's own sizes are not
    offered there with the option on or off, and the candidate values are the
    same. Adding offtree_third_raise brings the probes back on top.
    """
    state = Line(game).act(RAISE_POT).act(RAISE_TWO).state
    assert list(game.legal_actions(state)) == [FOLD, CHECK_CALL, ALL_IN]
    strategy = ShyOfShoves(game)
    out = {}
    for flag, third in ((False, ()), (True, ()), (True, (0.4, 1.0))):
        lbr = LocalBestResponse(game, strategy, 20, 8, bridge_translation=flag,
                                offtree_third_raise=third)
        candidate_range = lbr._deal_range(state, 0, np.random.default_rng(4))
        out[(flag, third)] = lbr._candidates(state, 0, candidate_range, np.random.default_rng(9))
    assert out[(False, ())] == out[(True, ())]
    moves, _ = out[(True, (0.4, 1.0))]
    assert [m.fraction for m in moves if m.probe] == [0.4, 1.0]
    assert not [m for m in moves if m.fraction is not None and not m.probe]


def test_on_is_a_no_op_for_on_tree_sizes(abstraction):
    """
    With only the tree's own sizes and stacks too deep to cap, nothing the bridge
    reads differs, so whole paired runs must be identical, which also checks that
    asking the strategy through `_policy` once the bridge record exists is the
    same lookup as `information_set`.
    """
    deep = make(abstraction, stack=10 ** 6)
    strategy = ShyOfShoves(deep)
    off = LocalBestResponse(deep, strategy, 10, 8, bet_sizes=(0.5, 1.0, 2.0)).play(40, paired_seed=3)
    lbr = LocalBestResponse(deep, strategy, 10, 8, bet_sizes=(0.5, 1.0, 2.0), bridge_translation=True)
    on = lbr.play(40, paired_seed=3)
    assert np.array_equal(off.values, on.values)
    assert lbr.probe_stats["overbet_allins"] == lbr.probe_stats["capped_allins"] == 0


# ---------------------------------------------------------------------------
# A called pseudo all-in in a played hand: the re-read line is what is asked.

class TableLike(ShyOfShoves):
    """
    ShyOfShoves with the real table's shape: no entry for a line that goes on
    after an all-in is called (the tree ends there), so only the re-read history
    can answer. ``alt`` maps a re-read street to a forced action; ``miss`` makes
    the re-read lookup miss as well.
    """

    def __init__(self, game, alt=None, miss=False, call_all_in=True):
        super().__init__(game, top=0 if call_all_in else 99)
        self.alt, self.miss = alt or {}, miss

    def get(self, key, default=None):
        bucket, history = key.split("|", 1)
        streets = history.split("/")
        if any(s.endswith(str(ALL_IN) + str(CHECK_CALL)) for s in streets[:-1]):
            return None
        if self.miss and history in self.alt:
            return None
        p = super().get(key, default)
        if history in self.alt:
            p = np.zeros_like(p)
            p[self.alt[history]] = 1.0
        return p


class Scripted(LocalBestResponse):
    """LBR whose moves are given in order, so a hand can be steered exactly."""

    def __init__(self, *args, script=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.script = list(script)
        self.seen = []

    def _choose(self, state, me, candidate_range, rng):
        self.seen.append(state.history)
        return self.script.pop(0)


def test_after_a_called_preflop_overbet_the_bot_is_asked_on_the_re_read_line(game):
    """
    LBR in the small blind opens 3.5 pot: "5". The bot calls. On the flop the
    bot (big blind, first to act) has no entry at "51/" and is asked at "41/",
    where this table bets half pot; LBR folds. Without the re-read it would play
    uniformly. The hand's history shows which happened.
    """
    strategy = TableLike(game, alt={"41/": 1})        # RAISE_HALF, index 1 of (1, 2, 3, 4, 5)
    lbr = Scripted(game, strategy, 10, 8, bridge_translation=True,
                   script=[Move(None, 3.5), Move(FOLD, None)])
    result = lbr._one(0, np.random.default_rng(0))
    assert result == -16.0 and lbr.seen == ["", "51/2"]
    assert lbr.probe_stats["overbet_allins"] == 1 and lbr.probe_stats["called"] == 1
    assert lbr.probe_stats["collapsed_streets"] == 1
    assert lbr.probe_stats["alt_hits"] == 1 and lbr.probe_stats["alt_misses"] == 0


def test_a_re_read_lookup_that_misses_falls_back_to_uniform_and_is_counted(game):
    """Same line with "41/" absent too: uniform play, one miss recorded."""
    strategy = TableLike(game, alt={"41/": 2}, miss=True)
    seen = []
    for seed in range(12):
        lbr = Scripted(game, strategy, 10, 8, bridge_translation=True,
                       script=[Move(None, 3.5)] + [Move(FOLD, None)] * 3)
        lbr._one(0, np.random.default_rng(seed))
        assert lbr.probe_stats["alt_misses"] >= 1 and lbr.probe_stats["alt_hits"] == 0
        seen.append(lbr.probe_stats["alt_misses"])
    assert sum(seen) >= 12


def test_a_called_reraise_overbet_is_re_read_as_2x_like_replay(game):
    """
    The bot opens pot, LBR in the big blind re-raises 3.5 pot (depth 1, line 3.0),
    the bot calls: the flop's true line is "351/" and the re-read one "341/",
    the same as replay builds from the arena's chips.
    """
    line = Line(game).act(RAISE_POT)
    lbr = LocalBestResponse(game, ShyOfShoves(game, top=0), 10, 8, bridge_translation=True)
    heard = lbr._apply_move(line.state, Move(None, 3.5), np.random.default_rng(0))
    assert heard.history == "35"
    line.entries.append((1, "raise", heard.committed[1], "preflop"))
    line.state = heard
    line.act(CHECK_CALL)
    lbr._street_closes(line.state)
    line.deal(FLOP)
    assert lbr._alt_history(line.state.history) == "341/"
    request = {"action_history": [{"seat": 0, "action": "post_small_blind", "amount": 1, "phase": "preflop"},
                                  {"seat": 1, "action": "post_big_blind", "amount": 2, "phase": "preflop"}]
               + [{"seat": a, "action": b, "amount": n, "phase": p} for a, b, n, p in line.entries],
               "phase": "flop", "your_stack": line.state.stacks[0], "opponent_stacks": [line.state.stacks[1]]}
    hand = replay(request, seat=0, rng=np.random.default_rng(0), schedule=game.raise_cap)
    assert hand.node.history == line.state.history and hand.alt_history == "341/"


def test_a_called_flop_overbet_is_re_read_on_the_turn(game):
    """Limped, LBR (big blind) bets 3.5 pot on the flop, called: turn re-read "11/41/"."""
    strategy = TableLike(game)
    lbr = Scripted(game, strategy, 10, 8, bridge_translation=True,
                   script=[Move(CHECK_CALL, None), Move(None, 3.5), Move(FOLD, None)] + [Move(FOLD, None)] * 3)
    lbr._one(1, np.random.default_rng(0))
    assert lbr.probe_stats["overbet_allins"] == 1 and lbr.probe_stats["called"] == 1
    assert lbr.probe_stats["collapsed_streets"] == 1
    assert lbr._alt_history("11/51/") == "11/41/"


def test_a_called_river_overbet_is_counted_called_but_collapses_nothing(game):
    """
    There is no next street to re-read, so collapsed_streets < called is
    correct. tests/test_lbr_overbet.py asserts collapsed_streets == called,
    which holds only because its strategy never calls a river overbet; the
    real run has 773 collapses for 775 calls in chunk 1.
    """
    strategy = TableLike(game)
    lbr = Scripted(game, strategy, 10, 8, bridge_translation=True,
                   script=[Move(CHECK_CALL, None)] * 3 + [Move(None, 3.5)])
    lbr._one(1, np.random.default_rng(0))
    assert lbr.probe_stats["overbet_allins"] == 1 and lbr.probe_stats["called"] == 1
    assert lbr.probe_stats["collapsed_streets"] == 0


# ---------------------------------------------------------------------------
# Counters and the off arm.

def test_both_options_together_every_pseudo_all_in_is_answered(game):
    """
    With probes and bridge translation on, every probe and every overbet read as
    a pseudo all-in is answered by the bot once, and probes stay in their own
    counter.
    """
    lbr = LocalBestResponse(game, ShyOfShoves(game), 10, 8, bet_sizes=BETWEEN,
                            offtree_third_raise=(0.4, 0.6, 1.0), bridge_translation=True)
    lbr.play(150, paired_seed=13)
    s = lbr.probe_stats
    assert s["probes"] > 0 and s["overbet_allins"] > 0
    assert s["folded"] + s["called"] == s["probes"] + s["overbet_allins"]
    assert s["collapsed_streets"] <= s["called"]


@pytest.mark.parametrize("cap, sizes, third, paired, expected", [
    # Computed 5 Oct with HEAD's cfr/lbr.py (git show HEAD:cfr/lbr.py), the
    # code before this branch: hands=60, rollout_samples=10, candidates=8.
    ((4, 2, 1), BETWEEN, (0.4, 0.6, 1.0), True, 32.34166666666667),
    ((4, 2, 1), DEFAULT_BET_SIZES, (0.4, 0.6, 1.0), False, 33.65833333333333),
    (2, BETWEEN, (), False, 46.56666666666667),
    (2, DEFAULT_BET_SIZES, (), True, -4.125),
    (((2, 3, 5), (2, 3, 5), (5,)), BETWEEN, (), True, 57.19166666666667),
    (((2, 3, 5), (2, 3, 5), (5,)), DEFAULT_BET_SIZES, (), False, 24.35),
])
def test_off_reproduces_head_on_other_schedules_and_with_probes(abstraction, cap, sizes, third,
                                                              paired, expected):
    game = make(abstraction, cap)
    strategy = ShyOfShoves(game)
    if paired:
        r = LocalBestResponse(game, strategy, 10, 8, bet_sizes=sizes,
                              offtree_third_raise=third).play(60, paired_seed=21)
    else:
        r = lbr_value(game, strategy, hands=60, rng=np.random.default_rng(5), rollout_samples=10,
                      candidates=8, bet_sizes=sizes, offtree_third_raise=third)
    assert r.mean == expected
