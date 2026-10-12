"""Leg (d) of the stopping rule: scripts/fold_jumps.py on small hand-built strategies."""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fold_jumps import (RANKS, actions_at, all_in_histories, check, combo_weights,  # noqa: E402
                                line_probability, neighbour_pairs, own_reach, parse_label, report, verdict)

CAP = 2                    # two raises: the small blind's raise can be re-raised all-in
LABELS = ["AA", "KK", "QQ", "AKs", "AQs", "KQs", "AKo", "AQo"]
AA, KK, QQ, AKS, AQS, KQS, AKO, AQO = range(len(LABELS))


def all_labels():
    out = []
    for high in range(12, -1, -1):
        for low in range(high, -1, -1):
            if high == low:
                out.append(RANKS[high] * 2)
            else:
                out += [f"{RANKS[high]}{RANKS[low]}s", f"{RANKS[high]}{RANKS[low]}o"]
    return out


def one_hot(actions, action, p=1.0):
    row = np.zeros(len(actions))
    row[actions.index(action)] = p
    if p < 1.0:                                     # the rest on the first other action
        row[[i for i in range(len(actions)) if actions[i] != action][0]] = 1.0 - p
    return row


def line(fold_at_shove, shove_share=0.5, root_raise=None):
    """
    Every class opens a half-pot raise ('2'), the big blind re-raises all-in
    ('25') with `shove_share` of every class, and the small blind folds there
    with `fold_at_shove[class]`.
    """
    strategy = {}
    root, after_raise, facing = actions_at("", CAP), actions_at("2", CAP), actions_at("25", CAP)
    for klass in range(len(LABELS)):
        strategy[f"{klass}|"] = one_hot(root, 2, (root_raise or {}).get(klass, 1.0))
        strategy[f"{klass}|2"] = one_hot(after_raise, 5, shove_share) if shove_share > 0 else one_hot(after_raise, 1)
        fold = fold_at_shove[klass]
        strategy[f"{klass}|25"] = np.array([fold, 1.0 - fold])
    assert facing == (0, 1)
    return strategy


MONOTONE = {AA: 0.0, KK: 0.0, QQ: 0.1, AKS: 0.0, AQS: 0.2, KQS: 0.6, AKO: 0.1, AQO: 0.7}


def test_labels_parse_both_ways():
    assert parse_label("AKs") == (12, 11, True)
    assert parse_label("72o") == (5, 0, False)
    assert parse_label("TT") == (8, 8, None)


def test_neighbours_on_the_full_grid():
    labels = all_labels()
    pairs = neighbour_pairs(labels)
    # 12 pair steps, then for each suitedness 66 steps of the high card and 66 of the low one.
    assert len(pairs) == 12 + 2 * (66 + 66)
    assert len(set(pairs)) == len(pairs)
    for strong, weak in pairs:
        s, w = parse_label(labels[strong]), parse_label(labels[weak])
        assert s[2] == w[2]                                       # same suitedness, pairs with pairs
        assert s[0] >= w[0] and s[1] >= w[1]                      # card for card at least as high
        assert (s[0] - w[0]) + (s[1] - w[1]) == (2 if s[2] is None else 1)
    names = {(labels[a], labels[b]) for a, b in pairs}
    assert ("AKs", "AQs") in names and ("AQs", "KQs") in names and ("KK", "QQ") in names
    assert ("AKs", "KQs") not in names                            # two steps, not a neighbour
    assert not any(b == "KK" and a.endswith("s") for a, b in names)   # KQs one up is not KK


def test_neighbours_of_a_partial_list():
    pairs = {(LABELS[a], LABELS[b]) for a, b in neighbour_pairs(LABELS)}
    assert pairs == {("AA", "KK"), ("KK", "QQ"), ("AKs", "AQs"), ("AQs", "KQs"), ("AKo", "AQo")}


def test_combo_weights():
    w = combo_weights(LABELS)
    assert w.sum() == pytest.approx(1.0)
    assert w[AA] / w[AKS] == pytest.approx(6 / 4) and w[AKO] / w[AKS] == pytest.approx(3)


def test_all_in_histories_are_preflop_and_facing_a_shove():
    strategy = {"0|": [1], "0|5": [1, 0], "0|25": [1, 0], "0|2": [1], "0|25/1": [1], "0|1/5": [1, 0]}
    assert all_in_histories(strategy) == ["5", "25"]


def test_own_reach_counts_only_the_players_own_actions():
    strategy = line(MONOTONE, shove_share=0.3, root_raise={AKS: 0.4})
    # The small blind at '25' acted once ('2' at the root); the big blind's 0.3 is not its reach.
    assert own_reach(strategy, CAP, AKS, "25") == pytest.approx(0.4)
    assert own_reach(strategy, CAP, AA, "25") == pytest.approx(1.0)
    # The big blind's own reach to '25' is its shove share.
    assert own_reach(strategy, CAP, AA, "25", player=1) == pytest.approx(0.3)
    assert own_reach(strategy, CAP, 99, "25") == 0.0                    # no entry for the class


def test_line_probability_is_the_opponents_range_weighted_reach():
    strategy = line(MONOTONE, shove_share=0.3)
    assert line_probability(strategy, CAP, "25", combo_weights(LABELS)) == pytest.approx(0.3)


def test_monotone_folds_flag_nothing():
    result = check(line(MONOTONE), CAP, LABELS)
    assert result["nodes"] == 1 and result["pairs_compared"] == 5
    assert result["flagged"] == [] and result["excess"] == 0.0 and result["weighted_excess"] == 0.0


def test_a_stronger_hand_folding_more_is_flagged_and_weighted():
    folds = {**MONOTONE, AKS: 0.9, AQS: 0.1}
    result = check(line(folds, shove_share=0.5), CAP, LABELS)
    assert [(r["stronger"], r["weaker"]) for r in result["flagged"]] == [("AKs", "AQs")]
    row = result["flagged"][0]
    assert row["excess"] == pytest.approx(0.8)
    assert row["line"] == pytest.approx(0.5) and row["reach_stronger"] == pytest.approx(1.0)
    assert result["weighted_excess"] == pytest.approx(0.8 * 0.5)


def test_threshold_is_inclusive_and_small_gaps_pass():
    folds = {**MONOTONE, AKS: 0.35, AQS: 0.2}          # a 0.15 gap, under the 0.2 default
    assert check(line(folds), CAP, LABELS)["flagged"] == []
    folds = {**MONOTONE, AKS: 0.45, AQS: 0.2}
    assert len(check(line(folds), CAP, LABELS, threshold=0.25)["flagged"]) == 1


def test_a_class_that_does_not_reach_the_node_is_not_compared():
    folds = {**MONOTONE, AKS: 0.9, AQS: 0.1}
    result = check(line(folds, root_raise={AKS: 0.01}), CAP, LABELS)
    assert result["flagged"] == [] and result["pairs_compared"] == 4


def test_an_untouched_uniform_entry_is_counted_not_compared():
    strategy = line(MONOTONE)
    strategy[f"{AKS}|25"] = np.array([0.5, 0.5])
    result = check(strategy, CAP, LABELS)
    assert result["uniform_entries"] == 1 and result["pairs_compared"] == 4


def test_a_line_the_opponent_never_takes_carries_no_weight():
    folds = {**MONOTONE, AKS: 0.9, AQS: 0.1}
    strategy = line(folds, shove_share=0.0)
    # Nobody shoves at '2', so '25' is never trained: still listed, weighted to nothing.
    for klass in range(len(LABELS)):
        strategy[f"{klass}|25"] = np.array([folds[klass], 1.0 - folds[klass]])
    result = check(strategy, CAP, LABELS)
    assert len(result["flagged"]) == 1 and result["weighted_excess"] == 0.0


def test_verdict():
    good, bad = {**MONOTONE, AKS: 0.5, AQS: 0.2}, {**MONOTONE, AKS: 0.9, AQS: 0.1}
    at_good, at_bad = check(line(good), CAP, LABELS), check(line(bad), CAP, LABELS)
    assert verdict(at_bad, at_good) == "pass"                 # 2T better
    assert verdict(at_good, at_good) == "pass"                # level is not worse
    assert verdict(at_good, at_bad) == "FAIL"                 # 2T worse
    assert verdict(at_good, check({}, CAP, LABELS)) == "not measured"


class _Abstraction:
    def __init__(self, labels):
        self._preflop = {}
        for index, label in enumerate(labels):
            high, low = label[0], label[1]
            self._preflop[(high, low, label.endswith("s"))] = index


def test_report_reads_a_saved_solve(tmp_path):
    path = tmp_path / "rung.pkl"
    folds = {**MONOTONE, AKS: 0.9, AQS: 0.1}
    with open(path, "wb") as handle:
        pickle.dump({"strategy": line(folds), "args": {"raise_cap": CAP, "stack": 140, "big_blind": 2},
                     "abstraction": _Abstraction(LABELS)}, handle)
    result = report(str(path), 0.2, 0.05)
    assert result["depth_bb"] == 70
    assert [(r["stronger"], r["weaker"]) for r in result["flagged"]] == [("AKs", "AQs")]


# ---- the cost check (6 October): price, equity, cost, verdict ---------------------------------------------------

from scripts.fold_jumps import (CLASSES, EQUITY_TABLE, cost_check, cost_verdict, equity_table,  # noqa: E402
                                labels_of, price)

T421 = (4, 2, 1)


@pytest.mark.parametrize("history, stack, expected", [
    ("15", 24, (22, 48, 0)),      # limp, shove: 11bb into a 24bb final pot, 45.8%
    ("25", 24, (20, 48, 0)),      # half-pot open to 2bb (1 to call, then 2 more), shove
    ("35", 36, (30, 72, 0)),      # pot open to 3bb at 18bb
    ("45", 36, (26, 72, 0)),      # twice-pot open to 5bb
    ("5", 24, (22, 48, 1)),       # the small blind open-shoves; the big blind has 1bb in
    ("245", 140, (120, 280, 1)),  # open 2bb, three-bet to 10bb, four-bet all-in, the big blind to call
    ("1445", 140, (90, 280, 0)),  # limp, raise to 5bb, re-raise to 25bb, shove
])
def test_price_replays_the_trainers_chip_rules(history, stack, expected):
    assert price(history, stack, 2, T421) == expected


def test_price_refuses_a_history_the_tree_cannot_hold():
    with pytest.raises(ValueError, match="not legal"):
        price("2222", 140, 2, T421)               # a fourth preflop raise past (4, 2, 1)'s third
    with pytest.raises(ValueError, match="not legal"):
        price("55", 24, 2, T421)                  # an all-in cannot be raised
    with pytest.raises(ValueError):
        price("11", 24, 2, T421)                  # limp, check: the flop, not a preflop decision


def test_the_equity_table_is_oriented_mine_against_theirs():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "cfr", "ladder169", "nolimit_12bb.pkl")
    if not os.path.exists(path) or not os.path.exists(EQUITY_TABLE):
        pytest.skip("the 169-class rung or the all-in table is not on this machine")
    from cfr.flat import load_strategy
    labels = labels_of(load_strategy(path))
    assert len(labels) == CLASSES
    eq, at = equity_table(), {label: i for i, label in enumerate(labels)}
    assert eq[at["AA"], at["72o"]] > 0.85 and eq[at["72o"], at["AA"]] < 0.15
    assert 0.40 < eq[at["AKs"], at["QQ"]] < 0.50                  # the classic race, a little under half
    assert np.allclose(eq + eq.T, 1.0, atol=0.06)                  # sampled, so a few hundredths of noise


def _priced_line(fold_at_shove, shove_share=0.5):
    """`line` above, at 12bb: everyone opens half pot ('2'), the big blind shoves, the small blind folds or calls."""
    return line(fold_at_shove, shove_share)


def _equity(value_for):
    """An 8 by 8 equity matrix where each of our classes has the same equity against every opponent class."""
    eq = np.zeros((len(LABELS), len(LABELS)))
    for klass, value in value_for.items():
        eq[klass, :] = value
    return eq


def test_cost_is_the_fold_share_of_a_winning_call_and_the_call_share_of_a_losing_one():
    # '25' at 12bb: call 20 chips into a 48-chip pot, so a call needs 41.7%. AA (60%) should call, KK (30%) fold.
    folds = {k: 0.0 for k in range(len(LABELS))}
    folds[AA], folds[KK] = 0.25, 0.4                              # AA folds a winner a quarter of the time, KK calls a loser 60%
    eq = _equity({k: 0.417 * 48 / 48 for k in range(len(LABELS))})
    eq[AA, :], eq[KK, :] = 0.6, 0.3
    result = cost_check(_priced_line(folds), CAP, LABELS, eq, 24, 2)
    node = next(n for n in result["nodes"] if n["history"] == "25")
    assert node["call_bb"] == 10 and node["pot_bb"] == 24 and node["needs"] == pytest.approx(20 / 48)
    assert node["line"] == pytest.approx(0.5)
    w = combo_weights(LABELS)
    ev_aa, ev_kk = (0.6 * 48 - 20) / 2, (0.3 * 48 - 20) / 2       # +4.4bb, -2.8bb
    want_aa = w[AA] * 1.0 * 0.5 * ev_aa * 0.25 * 100
    want_kk = w[KK] * 1.0 * 0.5 * -ev_kk * 0.6 * 100
    rows = {r["class"]: r for r in node["top"]}
    assert rows["AA"]["cost"] == pytest.approx(want_aa) and rows["KK"]["cost"] == pytest.approx(want_kk)
    assert rows["AA"]["ev_call_bb"] == pytest.approx(ev_aa) and rows["KK"]["ev_call_bb"] == pytest.approx(ev_kk)


def test_a_best_reply_costs_nothing_and_a_line_nobody_takes_costs_nothing():
    eq = _equity({k: (0.6 if k in (AA, AKS) else 0.3) for k in range(len(LABELS))})
    best = {k: (0.0 if k in (AA, AKS) else 1.0) for k in range(len(LABELS))}
    assert cost_check(_priced_line(best), CAP, LABELS, eq, 24, 2)["cost"] == pytest.approx(0.0)
    worst = {k: 1.0 - v for k, v in best.items()}
    assert cost_check(_priced_line(worst), CAP, LABELS, eq, 24, 2)["cost"] > 0
    nobody = line(worst, shove_share=0.0)
    for klass in range(len(LABELS)):                              # entries exist, but nobody ever shoves at '2'
        nobody[f"{klass}|25"] = np.array([worst[klass], 1.0 - worst[klass]])
    result = cost_check(nobody, CAP, LABELS, eq, 24, 2)
    assert result["cost"] == 0.0 and result["nodes"][0]["line"] == 0.0


def test_the_cost_scales_with_the_classs_own_reach():
    eq = _equity({k: 0.6 for k in range(len(LABELS))})
    folds = {k: 0.0 for k in range(len(LABELS))}
    folds[AA] = 1.0                                                # AA folds a 60% call every time
    full = cost_check(_priced_line(folds), CAP, LABELS, eq, 24, 2)["cost"]
    strategy = line(folds, shove_share=0.5, root_raise={AA: 0.5})  # AA opens only half the time
    half = cost_check(strategy, CAP, LABELS, eq, 24, 2)["cost"]
    assert half == pytest.approx(full / 2)


def test_cost_verdict():
    eq = _equity({k: 0.6 for k in range(len(LABELS))})
    good = cost_check(_priced_line({k: 0.0 for k in range(len(LABELS))}), CAP, LABELS, eq, 24, 2)
    bad = cost_check(_priced_line({k: 0.5 for k in range(len(LABELS))}), CAP, LABELS, eq, 24, 2)
    assert cost_verdict({"cost_check": bad}, {"cost_check": good}) == "pass"
    assert cost_verdict({"cost_check": good}, {"cost_check": good}) == "pass"
    assert cost_verdict({"cost_check": good}, {"cost_check": bad}) == "FAIL"
    assert cost_verdict({"cost_check": good}, {"cost_check": None}) == "not measured"


def test_a_solve_without_the_169_classes_is_not_priced(tmp_path):
    path = tmp_path / "rung.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"strategy": line(MONOTONE), "args": {"raise_cap": CAP, "stack": 24, "big_blind": 2},
                     "abstraction": _Abstraction(LABELS)}, handle)
    assert report(str(path), 0.2, 0.05, equity_table())["cost_check"] is None


def test_a_real_rung_is_priced_end_to_end():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "cfr", "ladder169", "nolimit_12bb.pkl")
    if not os.path.exists(path) or not os.path.exists(EQUITY_TABLE):
        pytest.skip("the 169-class rung or the all-in table is not on this machine")
    result = report(path, 0.2, 0.05, equity_table())
    priced = result["cost_check"]
    assert priced is not None and priced["nodes"]
    assert all(n["cost"] >= 0 for n in priced["nodes"]) and priced["cost"] == pytest.approx(sum(n["cost"] for n in priced["nodes"]))
    shove = next(n for n in priced["nodes"] if n["history"] == "5")
    assert shove["call_bb"] == 11 and shove["pot_bb"] == 24
