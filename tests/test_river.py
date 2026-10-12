"""
River endgame solving: the hand set's showdown arithmetic, the tree in live
chips, CFR+ on toy ranges where the answer is known, and the arena hook.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD  # noqa: E402
from cfr.river import HandSet, RiverSolver, blueprint_ranges, build_tree, decide_river  # noqa: E402
from slumbot.bridge import parse_cards  # noqa: E402

BOARD = ["2s", "6d", "9c", "Qh", "Ad"]       # no flush, no straight possible: the nuts is AA


@pytest.fixture(scope="module")
def hands():
    return HandSet.build(parse_cards(BOARD))


def index(hands, *cards):
    return hands.index_of[tuple(sorted(c.index for c in parse_cards(list(cards))))]


def test_the_board_leaves_1081_hands_and_ranks_the_nuts_on_top(hands):
    assert hands.size == 1081
    kk = index(hands, "As", "Ac")
    assert hands.ranks[kk] == hands.ranks.max()
    assert hands.ranks[index(hands, "3h", "4h")] < hands.ranks[index(hands, "Ah", "Qc")]


def test_showdown_mass_respects_card_removal(hands):
    """
    Against a uniform opponent, the nuts beats every compatible hand and a
    hand's own cards are not counted against it: KK's beaten mass equals the
    number of hands sharing no card with it, less the ties (none).
    """
    reach = np.ones((1, hands.size))
    diff = hands.showdown(reach)[0]
    kk = index(hands, "As", "Ac")
    compatible = hands.compatible_mass(reach)[0]
    assert diff[kk] == pytest.approx(compatible[kk])
    # Compatible hands from the 45 unseen cards: C(45, 2).
    assert compatible[kk] == pytest.approx(45 * 44 / 2)
    assert diff.max() == diff[kk]
    assert diff[index(hands, "3h", "4h")] < 0


def test_the_tree_is_built_in_live_chips_and_caps_raises():
    root, decisions = build_tree(pot_before=1000, to_call=0, stacks=(5000, 5000),
                                 raises_so_far=0, schedule=2, first_player=0)
    assert root.actions[0] == CHECK_CALL and FOLD not in root.actions
    assert ALL_IN in root.actions
    # After a bet and a raise the cap is reached: only fold or call remain.
    bet = root.children[1]
    raise_ = bet.children[2]
    assert raise_.actions == [FOLD, CHECK_CALL]
    assert raise_.children[1].kind == "showdown"
    # Facing a bet at the root: fold is offered, and a call goes to showdown.
    facing, _ = build_tree(1000, 500, (4500, 5000), 1, 2, 0)
    assert facing.actions[:2] == [FOLD, CHECK_CALL]
    assert facing.children[1].kind == "showdown"
    assert facing.children[1].contrib == (1000, 1000)


def test_the_nuts_bets_and_air_folds_to_a_pot_bet(hands):
    """Known answers: value bets with the best hand, folds air facing a bet from everything."""
    uniform = np.ones(hands.size)
    root, decisions = build_tree(1000, 0, (5000, 5000), 0, 2, 0)
    solver = RiverSolver(hands, root, decisions, (uniform.copy(), uniform.copy()))
    solver.solve(150, 30)
    nuts = solver.root_strategy(index(hands, "As", "Ac"))
    assert nuts[CHECK_CALL] < 0.3 and sum(p for a, p in nuts.items() if a >= 2) > 0.7
    assert abs(sum(nuts.values()) - 1.0) < 1e-9
    facing, decisions = build_tree(1000, 1000, (4000, 5000), 1, 2, 0)
    solver = RiverSolver(hands, facing, decisions, (uniform.copy(), uniform.copy()))
    solver.solve(150, 30)
    air = solver.root_strategy(index(hands, "3h", "4h"))
    assert air[FOLD] > 0.7
    assert solver.root_strategy(index(hands, "As", "Ac"))[FOLD] < 0.05


def test_four_hundred_iterations_fit_the_clock(hands):
    uniform = np.ones(hands.size)
    root, decisions = build_tree(1500, 0, (8000, 8000), 0, 2, 0)
    solver = RiverSolver(hands, root, decisions, (uniform.copy(), uniform.copy()))
    started = time.perf_counter()
    done = solver.solve(400, 30)
    assert done == 400
    assert time.perf_counter() - started < 15


def test_ranges_are_uniform_under_an_empty_blueprint_and_shrink_under_a_fold(hands):
    from abstraction.buckets import CardAbstraction
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4).fit(np.random.default_rng(0))
    board = parse_cards(BOARD)
    ours, theirs = blueprint_ranges(hands, {}, abstraction, 1, "11/11/11/", board, True)
    # Preflop: two decisions each at uniform 1/n over the legal list; every hand alive.
    assert ours.min() > 0 and theirs.min() > 0
    assert np.allclose(ours, ours[0]) and np.allclose(theirs, theirs[0])
    # A blueprint that always folds a bucket preflop removes those hands from our range.
    strategy = {f"{b}|": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) for b in range(2)}
    ours2, _ = blueprint_ranges(hands, strategy, abstraction, 1, "11/11/11/", board, True)
    assert ours2.max() > ours.max()          # a call taken with certainty raises reach


def test_the_arena_entry_point_returns_a_legal_action():
    state = {"pot": 2000, "to_call": 1000, "your_stack": 6000, "opponent_stacks": [5000],
             "board": BOARD, "your_hole_cards": ["As", "Ac"], "min_raise": 2000, "max_raise": 6000}
    from abstraction.buckets import CardAbstraction
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4).fit(np.random.default_rng(0))
    legal = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    decision = decide_river(state, parse_cards(["As", "Ac"]), parse_cards(BOARD), "11/11/11/3",
                            {}, abstraction, 2, True, np.random.default_rng(0), legal,
                            iterations=100, budget_s=20)
    assert decision.choice in decision.distribution and legal[decision.choice]
    assert decision.distribution.get(FOLD, 0.0) < 0.05
    assert decision.hands == 1081 and decision.iterations == 100


def test_facing_a_bet_from_the_top_half_the_answer_follows_equity(hands):
    """
    At the root facing a bet the opponent has no further decision before
    showdown, so our answer is a best response to their range: pure per hand,
    ordered by equity against it. Against a range of the top half, king-high
    (40th percentile) has nothing and folds; queens up (79th) beats most of the
    range and continues; aces up raises for value.
    """
    ours = np.ones(hands.size)
    top_half = (hands.lo >= hands.size // 2).astype(float)
    facing, decisions = build_tree(1000, 1000, (4000, 5000), 1, 2, 0)
    solver = RiverSolver(hands, facing, decisions, (ours, top_half))
    solver.solve(200, 30)
    king_high = solver.root_strategy(index(hands, "Kh", "Jd"))
    queens = solver.root_strategy(index(hands, "Qs", "Jd"))
    aces = solver.root_strategy(index(hands, "Ah", "Qc"))
    assert king_high[FOLD] > 0.9
    assert queens[FOLD] < 0.1
    assert aces[FOLD] < 0.05 and sum(p for a, p in aces.items() if a >= 2) > 0.5


def test_the_native_solver_matches_the_python_reference_and_is_much_faster(hands):
    # The C++ core (native/src/river.hpp) does the same arithmetic as the
    # Python; both are run on the same tree and ranges and their root
    # strategies pinned to each other. Floating-point order differs, so the
    # pin is loose enough for that and tight enough to catch a wrong sign,
    # a missing blocker correction or an off-by-one in the rank order.
    from cfr.river import NativeRiverSolver
    rng = np.random.default_rng(4)
    ranges = (rng.random(hands.size), rng.random(hands.size))
    for args in ((1000, 0, (5000, 5000), 0, 2, 0), (1000, 1000, (4000, 5000), 1, 2, 0),
                 (600, 300, (900, 2500), 1, (4, 2, 1), 0)):
        root, decisions = build_tree(*args)
        py = RiverSolver(hands, root, decisions, (ranges[0].copy(), ranges[1].copy()))
        cc = NativeRiverSolver(hands, root, decisions, (ranges[0].copy(), ranges[1].copy()))
        py.solve(60, 60); cc.solve(60, 60)
        assert cc.iterations == 60
        worst = 0.0
        for h in (index(hands, "As", "Ac"), index(hands, "3h", "4h"), index(hands, "Kd", "Qd")):
            a, b = py.root_strategy(h), cc.root_strategy(h)
            assert list(a) == list(b)
            worst = max(worst, max(abs(a[k] - b[k]) for k in a))
        assert worst < 0.02, f"native and python root strategies differ by {worst:.3f} on {args}"
    root, decisions = build_tree(1500, 0, (8000, 8000), 0, 2, 0)
    cc = NativeRiverSolver(hands, root, decisions, (np.ones(hands.size), np.ones(hands.size)))
    started = time.perf_counter()
    assert cc.solve(400, 30) == 400
    assert time.perf_counter() - started < 1.0, "400 native iterations should take well under a second"


def test_the_blend_mixes_the_opponent_range_with_any_two_cards(monkeypatch):
    # 30 Sept: blend 0 is the blueprint's reach as before; 1 is every hand the board allows, equally.
    import cfr.river as river
    seen = {}
    real = river.make_solver
    def capture(hands, root, decisions, ranges, native=None):
        seen["theirs"] = np.asarray(ranges[1], dtype=float)
        return real(hands, root, decisions, ranges, native)
    monkeypatch.setattr(river, "make_solver", capture)
    state = {"pot": 2000, "to_call": 1000, "your_stack": 6000, "opponent_stacks": [5000],
             "board": BOARD, "your_hole_cards": ["As", "Ac"], "min_raise": 2000, "max_raise": 6000}
    from abstraction.buckets import CardAbstraction
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4).fit(np.random.default_rng(0))
    legal = np.ones(6)
    args = (state, parse_cards(["As", "Ac"]), parse_cards(BOARD), "11/11/11/3", {}, abstraction, 2, True,
            np.random.default_rng(0), legal)
    plain = decide_river(*args, iterations=50, budget_s=20)
    assert plain.range_source == "blueprint"
    mixed = decide_river(*args, iterations=50, budget_s=20, blend=1.0)
    assert mixed.range_source == "blend 1"
    assert abs(seen["theirs"].sum() - 1.0) < 1e-9 and np.allclose(seen["theirs"], seen["theirs"][0])
    half = decide_river(*args, iterations=50, budget_s=20, blend=0.5)
    assert half.range_source == "blend 0.5" and abs(seen["theirs"].sum() - 1.0) < 1e-9
    assert half.choice in half.distribution and legal[half.choice]
