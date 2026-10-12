"""
The per-node visit counter: off changes nothing, on counts what it says.

Kuhn is small enough that several counts are fixed by the traversal itself,
not by the random stream, so they are checked exactly rather than roughly:
external sampling walks every traverser action and samples one action for the
other player, which pins how many times each history is entered.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

native = pytest.importorskip("pokerbot_native", reason="run native/build.sh to build the C++ core")

from cfr.visits import (Visits, classify, from_solver, load_visits, street_and_depth,  # noqa: E402
                        visits_path, write_visits)

CARDS = ("0", "1", "2")


def _sum(visits, history, which):
    return sum(visits[f"{c}|{history}"][which] for c in CARDS if f"{c}|{history}" in visits)


@pytest.mark.parametrize("threads", [1, 3])
def test_kuhn_counts_match_what_the_traversal_forces(threads):
    T = 5000
    v = native.solve_kuhn_visits(T, 11, threads)
    AVG, REG = 0, 1
    # Player 0's root is entered once per iteration in each traversal: as the
    # sampled player in player 1's (average) and as the traverser in its own (regret).
    assert _sum(v, "", AVG) == T
    assert _sum(v, "", REG) == T
    # In player 0's traversal both of its actions are walked, so player 1 is
    # sampled once at "p" and once at "b" every iteration.
    assert _sum(v, "p", AVG) == T
    assert _sum(v, "b", AVG) == T
    # ...and per card, since the same deal reaches both.
    for c in CARDS:
        assert v[f"{c}|p"][AVG] == v[f"{c}|b"][AVG]
    # In player 1's traversal player 0 samples one of the two, so player 1's
    # regret updates at "p" and "b" add up to T.
    assert _sum(v, "p", REG) + _sum(v, "b", REG) == T
    # Player 0 reaches "pb" as the sampled player exactly when, in player 1's
    # traversal, it sampled a pass and player 1 (walking both) bet: once per
    # regret update player 1 made at "p".
    assert _sum(v, "pb", AVG) == _sum(v, "p", REG)
    # Locks cover the counters: the threaded run loses nothing, which the
    # equalities above already require exactly.


def test_kuhn_skips_the_counts_the_average_skips():
    """Samples before average_from never reach the average, so they are not counted as avg visits."""
    T, skip = 4000, 1000
    v = native.solve_kuhn_visits(T, 3, 1, skip)
    assert _sum(v, "", 0) == T - skip
    assert _sum(v, "", 1) == T     # regret updates run from the first iteration


def _small(seed=3):
    return native.NoLimitSolver([0] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], seed)


def test_counting_does_not_change_the_strategy_and_off_counts_nothing():
    off, on = _small(), _small()
    on.set_count_visits(True)
    assert not off.count_visits() and on.count_visits()
    off.train(400)
    on.train(400)
    assert off.average_strategy() == on.average_strategy()
    _, avg, reg = off.visits_flat(32)
    assert not np.asarray(avg).any() and not np.asarray(reg).any()


@pytest.mark.parametrize("threads", [1, 2])
def test_no_limit_root_counts_equal_the_iterations(threads):
    solver = _small(5)
    solver.set_count_visits(True)
    solver.train(600, threads)
    v = from_solver(solver)
    roots = np.array([k.split(b"|", 1)[1].rstrip(b"\0") == b"" for k in v.keys])
    assert v.avg[roots].sum() == 600 and v.regret[roots].sum() == 600
    # A node that exists was entered at least once one way or the other.
    assert ((v.avg + v.regret) > 0).all()


def test_the_counts_line_up_with_the_flat_strategy_and_round_trip(tmp_path):
    solver = _small()
    solver.set_count_visits(True)
    solver.train(300)
    key_bytes, _, _ = solver.average_strategy_flat(32)
    flat_keys = np.asarray(key_bytes).view("S32").reshape(-1)
    v = from_solver(solver, averaged_from=0, meta={"note": "test"})
    assert np.array_equal(v.keys, flat_keys)
    path = visits_path(str(tmp_path / "rung.pkl"))
    assert path.endswith("rung.visits.npz")
    assert visits_path(str(tmp_path / "rung.pkl"), 2000).endswith("rung.visits.2000.npz")
    write_visits(path, v)
    back = load_visits(path)
    assert np.array_equal(back.keys, v.keys)
    assert np.array_equal(back.avg, v.avg) and back.avg.dtype == np.uint32
    assert np.array_equal(back.regret, v.regret)
    assert back.iterations == 300 and back.averaged_iterations == 300
    assert back.meta == {"note": "test"}


def test_street_and_raise_depth_from_the_key():
    assert street_and_depth("12|") == (0, 0)
    assert street_and_depth("12|23") == (0, 2)
    assert street_and_depth(b"3|231/1\0\0") == (1, 0)
    assert street_and_depth("3|21/14/5") == (2, 1)
    assert street_and_depth("3|11/2/1/32") == (3, 2)
    streets, depths = classify(np.array([b"1|", b"1|2/23"], dtype="S32"))
    assert streets.tolist() == [0, 1] and depths.tolist() == [0, 2]


def test_the_report_rule_needs_every_leg():
    from scripts.cfr import visit_report as report

    avg = np.array([0, 5, 50, 500, 5000, 50000], dtype=np.uint32)
    reg = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint32)
    row = report.summarise(avg, reg)
    assert row["under_10"] == pytest.approx(2 / 6) and row["zero"] == pytest.approx(1 / 6)
    # Weighted by avg x regret, nearly all the weight sits on the busiest node.
    assert row["rw_p10"] == 50000
    v = Visits(np.array([b"1|", b"1|2", b"1|2/1", b"1|1/1/1", b"1|23", b"1|1"], dtype="S32"),
               avg, reg, 1000, 1000)
    rows = report.breakdown(v)
    assert rows["all"]["nodes"] == 6 and rows["preflop d1"]["nodes"] == 1

    class A:  # noqa: D401 - a stand-in for argparse's namespace
        bar, h2h_delta, h2h_se, lbr_delta, lbr_se = 100, None, None, None, None
    # Leg (a) is judged on the worst line: the whole tree passing is not enough.
    legs, _ = report.rule(rows, A)
    assert rows["all"]["rw_p10"] >= 100 and not legs["a_visits"]["pass"]
    assert legs["a_visits"]["line"] == "preflop d1"
    busy = {"all": rows["all"], "preflop d0": rows["all"], "flop d2": rows["all"]}
    legs, verdict = report.rule(busy, A)
    assert legs["a_visits"]["pass"] and verdict == "not measured"
    rows = busy
    A.h2h_delta, A.h2h_se, A.lbr_delta, A.lbr_se = 0.4, 0.7, -1.0, 4.5
    assert report.rule(rows, A)[1] == "done"
    A.lbr_delta = 6.0
    assert report.rule(rows, A)[1] == "not done"
    A.lbr_delta, A.h2h_se = -1.0, 2.0       # too wide to call a sub-point move
    assert report.rule(rows, A)[0]["b_h2h"]["pass"] is None
