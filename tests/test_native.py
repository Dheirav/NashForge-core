"""
The C++ core against the Python it replaces.

A port that quietly disagrees is worse than no port: it would not fail, it would
shift every equity estimate, every bucket and every strategy trained afterwards.
So the deterministic half is pinned hand-for-hand and the sampled half is
checked for bias.

Skipped when the module is not built, because `native/build.sh` needs cmake,
ninja and nanobind, and the rest of the suite must not depend on them.
"""
import random

import numpy as np
import pytest

from abstraction.equity import FULL_DECK, equity_vs_random
from engine.hand_eval_fast import score_hand_7_fast

native = pytest.importorskip("pokerbot_native",
                             reason="run native/build.sh to build the C++ core")


def test_the_cpp_evaluator_is_identical_on_every_hand_class():
    """
    Exact, not approximate. Restricted decks are included because random
    seven-card hands almost never make quads or straight flushes, and those are
    the branches most likely to be wrong.
    """
    decks = [
        list(range(52)),
        [s * 13 + r for s in range(4) for r in range(6)],     # forces quads, boats
        [s * 13 + r for s in range(2) for r in range(13)],    # forces flushes
    ]
    rng = random.Random(2026)
    classes = set()
    for deck in decks:
        for _ in range(2000):
            cards = rng.sample(deck, 7)
            ranks = np.array([c % 13 for c in cards], dtype=np.int32)
            suits = np.array([c // 13 for c in cards], dtype=np.int32)
            expected = int(score_hand_7_fast(ranks, suits))
            assert native.score_hand_7([c % 13 for c in cards],
                                       [c // 13 for c in cards]) == expected, (
                f"C++ evaluator differs on {cards}")
            classes.add(expected >> 20)
    assert len(classes) >= 8, f"only saw hand classes {sorted(classes)}"


def test_the_cpp_rollout_is_unbiased_against_the_python_one():
    """
    Not identical: the two draw from different generators on purpose. What must
    hold is that neither systematically favours a hand, which a mean difference
    indistinguishable from zero is what shows.
    """
    rng = random.Random(99)
    differences = []
    for _ in range(400):
        board_size = rng.choice([3, 4, 5])
        cards = rng.sample(range(52), 2 + board_size)
        hole, board = cards[:2], cards[2:]
        mine = equity_vs_random([FULL_DECK[c] for c in hole],
                                [FULL_DECK[c] for c in board], 2000,
                                np.random.default_rng(rng.randrange(10 ** 6)))
        theirs = native.equity_vs_random(hole, board, 2000, rng.randrange(10 ** 9))
        differences.append(mine - theirs)

    values = np.array(differences)
    interval = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    assert abs(values.mean()) < interval, (
        f"C++ rollout is biased by {values.mean():+.5f} +/- {interval:.5f}")


def test_the_cpp_betting_game_agrees_over_the_enumerated_tree():
    """
    The same check `test_betting_equivalence.py` makes, against a third
    implementation.

    Betting is written twice in this project and has diverged twice: a
    pot-fraction raise once sized off the pot before the call instead of after,
    making every engine raise about 20% too small, and the traversal game once
    failed to end a hand at an all-in, keying 19.7% of decision nodes
    differently. Neither failed; both returned plausible wrong numbers. A C++
    port makes it three implementations, so it gets the same treatment.

    Compared at every node: pot, both stacks, the history string (which is the
    information-set key's public half), and the legal-action list itself. All
    four raise schedules are covered, including the tapered ones, since a taper
    is exactly the kind of change that splits two implementations apart.
    """
    import itertools

    from abstraction.buckets import CardAbstraction
    from games.nolimit import NoLimitHoldem

    stack, small_blind, big_blind = 200, 1, 2
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2,
                                  samples=20, equity_samples=4)
    abstraction.fit(np.random.default_rng(0))

    def python_replay(actions, schedule):
        game = NoLimitHoldem(abstraction, starting_stack=stack,
                             small_blind=small_blind, big_blind=big_blind,
                             raise_cap=schedule, equity_samples=4)
        state = game.initial_state()._replace(hole=((0, 1), (2, 3)))
        legal_each = []
        for action in actions:
            if game.is_terminal(state):
                return -1, None, None, legal_each
            if game.current_player(state) < 0:
                return -2, None, None, legal_each
            legal = list(game.legal_actions(state))
            legal_each.append(legal)
            if action not in legal:
                return -3, None, None, legal_each
            state = game.next_state(state, action)
        return sum(state.contributions), list(state.stacks), state.history, legal_each

    # The last three name their sizes rather than counting them from the
    # largest end (23 September). A named level is exactly the kind of change
    # that splits two implementations apart, because the C++ has its own copy
    # of "which raises are legal here", so every one of them is driven through
    # the same sequences as the counted schedules.
    from abstraction.betting import ALL_IN, RAISE_HALF, RAISE_POT, RAISE_TWO
    named = ((RAISE_HALF, RAISE_POT, ALL_IN), (RAISE_HALF, RAISE_POT, ALL_IN), (ALL_IN,))
    schedules = [(1, [4], []), (2, [4, 4], []), ((4, 1), [4, 1], []), ((4, 2), [4, 2], []),
                 (named, [], [list(level) for level in named]),
                 (((RAISE_POT, ALL_IN),), [], [[RAISE_POT, ALL_IN]]),
                 (((RAISE_HALF, RAISE_TWO), (ALL_IN,)), [], [[RAISE_HALF, RAISE_TWO], [ALL_IN]])]
    compared = 0
    for python_schedule, cpp_schedule, cpp_sizes in schedules:
        for depth in (1, 2, 3):
            for actions in itertools.product(range(6), repeat=depth):
                expected = python_replay(list(actions), python_schedule)
                got = native.replay(list(actions), stack, small_blind, big_blind,
                                    cpp_schedule, cpp_sizes)
                compared += 1
                assert expected[0] == got[0], (
                    f"schedule={python_schedule} actions={actions}: "
                    f"pot {expected[0]} vs {got[0]}")
                if expected[0] >= 0:
                    assert expected[1] == list(got[1]), (
                        f"schedule={python_schedule} actions={actions}: stacks differ")
                    assert expected[2] == got[2], (
                        f"schedule={python_schedule} actions={actions}: history differs")
                assert expected[3] == [list(step) for step in got[3]], (
                    f"schedule={python_schedule} actions={actions}: legal actions differ")
    assert compared > 1000, f"only compared {compared} sequences"


def test_the_cpp_solver_converges_to_kuhns_analytic_value():
    """
    Kuhn poker's value to the first player is exactly -1/18 under optimal play.

    This is the acceptance test for the ported MCCFR, and it is stronger than
    comparing against our own Python: that would show the translation was
    faithful, while this shows regret matching, external sampling and strategy
    averaging are each correct. It is the same anchor `tests/test_mccfr.py`
    uses for the Python solver.
    """
    import itertools

    def game_value(strategy):
        def probability(key, action):
            entry = strategy.get(key)
            return entry[action] if entry else 0.5

        total = 0.0
        for mine, theirs in itertools.permutations(range(3), 2):
            showdown = 1.0 if mine > theirs else -1.0
            value = probability(f"{mine}|", 0) * (
                probability(f"{theirs}|p", 0) * showdown
                + probability(f"{theirs}|p", 1) * (
                    probability(f"{mine}|pb", 0) * -1.0
                    + probability(f"{mine}|pb", 1) * 2.0 * showdown))
            value += probability(f"{mine}|", 1) * (
                probability(f"{theirs}|b", 0) * 1.0
                + probability(f"{theirs}|b", 1) * 2.0 * showdown)
            total += value / 6.0
        return total

    values = [game_value(native.solve_kuhn(200_000, seed)) for seed in range(3)]
    mean = sum(values) / len(values)
    assert abs(mean - (-1 / 18)) < 0.002, (
        f"converged to {mean:+.5f}, expected {-1/18:+.5f}; seeds {values}")

    # The same anchor with four worker threads sharing the table. A threaded
    # run is not reproducible bit for bit (which worker takes which iteration
    # is up to the scheduler), so this is the test of it: the algorithm still
    # converges, with the per-node lock and the out-of-order discount stamp.
    threaded = [game_value(native.solve_kuhn(200_000, seed, 0, 4)) for seed in range(3)]
    mean = sum(threaded) / len(threaded)
    assert abs(mean - (-1 / 18)) < 0.002, (
        f"four threads converged to {mean:+.5f}, expected {-1/18:+.5f}; seeds {threaded}")


def test_the_cpp_no_limit_solver_produces_a_usable_strategy():
    """
    The solver runs end to end and returns something the Python can play.

    Deliberately small: the acceptance test that matters is the one run by hand
    on 11 September, where C++ and Python solvers at 40,000 iterations were
    scored by the same harness and agreed to 16 and 23 BB/100 against random and
    always-call, well inside the measurement interval. That takes six minutes
    because of the Python arm, so it does not belong in the suite. This checks
    the wiring: keys are well formed, distributions are distributions.
    """
    from abstraction.buckets import CardAbstraction, preflop_key
    from abstraction.equity import FULL_DECK

    abstraction = CardAbstraction(preflop_buckets=4, postflop_buckets=4,
                                  samples=60, equity_samples=8)
    abstraction.fit(np.random.default_rng(0))

    preflop = [0] * (52 * 52)
    for first in range(52):
        for second in range(52):
            if first != second:
                preflop[first * 52 + second] = abstraction._preflop[
                    preflop_key([FULL_DECK[first], FULL_DECK[second]])]

    solver = native.NoLimitSolver(
        preflop,
        list(abstraction._centroid_list["flop"]),
        list(abstraction._centroid_list["turn"]),
        list(abstraction._centroid_list["river"]),
        8, 200, 1, 2, [4], 0)
    solver.train(300)

    strategy = solver.average_strategy()
    assert strategy, "solver produced no information sets"
    assert solver.iterations() == 300

    for key, probabilities in strategy.items():
        assert "|" in key, f"malformed information-set key {key!r}"
        bucket, history = key.split("|", 1)
        assert bucket.isdigit(), f"bucket is not a number in {key!r}"
        assert set(history) <= set("012345/"), f"unexpected history in {key!r}"
        assert 2 <= len(probabilities) <= 6, f"{key!r} has {len(probabilities)} actions"
        assert abs(sum(probabilities) - 1.0) < 1e-9, f"{key!r} is not a distribution"
        assert all(p >= 0.0 for p in probabilities), f"{key!r} has a negative probability"


def test_board_texture_agrees_between_python_and_the_native_core():
    """Every class of board the C++ solver could bucket, read the same way."""
    import numpy as np
    from abstraction.buckets import board_texture
    from abstraction.equity import FULL_DECK
    import pokerbot_native
    rng = np.random.default_rng(13)
    for _ in range(3000):
        n = int(rng.choice([3, 4, 5]))
        picked = [int(i) for i in rng.choice(52, size=n, replace=False)]
        cards = [FULL_DECK[i] for i in picked]
        assert board_texture(cards) == pokerbot_native.board_texture([c.index for c in cards])


def test_the_native_preflop_table_holds_a_lossless_class_index():
    """
    The preflop table was int8_t and the lossless preflop numbers hands to
    168, so aces would have wrapped to a negative bucket and keyed as one.
    """
    solver = native.NoLimitSolver([168] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6],
                                  8, 200, 1, 2, [4], 0)
    solver.train(50)
    preflop_keys = [k for k in solver.average_strategy() if "/" not in k]
    assert preflop_keys and all(k.startswith("168|") for k in preflop_keys)


def test_the_native_solver_takes_an_update_rule_by_name():
    """
    The rule was hardcoded to vanilla, whose unweighted average left a rarely
    reached node at its early near-uniform visits. The four names are the ones
    `cfr/updates.py` defines, so a rule means the same thing on both paths.
    """
    table = [0] * (52 * 52)
    for rule in ("vanilla", "linear", "cfr+", "dcfr"):
        solver = native.NoLimitSolver(table, [0.3, 0.6], [0.3, 0.6], [0.3, 0.6],
                                      8, 200, 1, 2, [4], 0, rule=rule)
        solver.train(20)
        assert solver.iterations() == 20
    with pytest.raises(Exception):
        native.NoLimitSolver(table, [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4], 0, rule="nope")


def test_the_native_solver_reproduces_its_golden_output():
    """
    The reference every engineering change is measured against: the same
    small solve, the same seed, every average-strategy entry identical to the
    last bit. A change that alters this file either has a bug or changes the
    path, and a path change is re-baselined only with `make_golden.py --force`
    after its own convergence tests. See `scripts/cfr/make_golden.py`.
    """
    import json
    from scripts.cfr.make_golden import GOLDEN, solve
    with open(GOLDEN) as handle:
        golden = json.load(handle)
    strategy = solve(golden["spec"])
    assert len(strategy) == golden["entries"]
    assert set(strategy) == set(golden["strategy"])
    mismatched = [k for k, v in golden["strategy"].items() if strategy[k] != v]
    assert not mismatched, f"{len(mismatched)} entries differ, e.g. {mismatched[:3]}"


def _kuhn_value(strategy):
    """Kuhn's game value to the first player under `strategy`, as the anchor test computes it."""
    import itertools

    def probability(key, action):
        entry = strategy.get(key)
        return entry[action] if entry else 0.5

    total = 0.0
    for mine, theirs in itertools.permutations(range(3), 2):
        showdown = 1.0 if mine > theirs else -1.0
        value = probability(f"{mine}|", 0) * (
            probability(f"{theirs}|p", 0) * showdown
            + probability(f"{theirs}|p", 1) * (
                probability(f"{mine}|pb", 0) * -1.0
                + probability(f"{mine}|pb", 1) * 2.0 * showdown))
        value += probability(f"{mine}|", 1) * (
            probability(f"{theirs}|b", 0) * 1.0
            + probability(f"{theirs}|b", 1) * 2.0 * showdown)
        total += value / 6.0
    return total


def test_warm_start_keeps_and_recovers_kuhns_value():
    """
    The warm start (MCCFR::warm_start) in the two cases that matter.

    Seeded from a converged solve and given the same weight in iterations as
    the run itself, the solver must stay at -1/18: the prior is right and the
    schedule must not throw it away. Seeded from a deliberately wrong prior
    (always bet, always call) it must still get there, because a warm start is
    a starting point and the regrets have to be able to overrule it; the run
    is four times the prior's weight so the correction has room. Written
    because the use is a cap-2 rung seeded from the one-raise rung, whose
    strategy is a good guess at the shared nodes and wrong in the details,
    and the two properties together are what make that safe.
    """
    converged = native.solve_kuhn(200_000, 0)
    prior = [(key, list(map(float, probs))) for key, probs in converged.items()]
    kept = [_kuhn_value(native.solve_kuhn(200_000, seed, 0, 1, prior, 200_000, 1.0))
            for seed in range(3)]
    mean = sum(kept) / len(kept)
    assert abs(mean - (-1 / 18)) < 0.002, (
        f"warm start from the converged solve drifted to {mean:+.5f}; seeds {kept}")

    # The wrong prior, under the linear rule the trainer uses, whose discount
    # is what lets a run outgrow its start: the vanilla average carries the
    # iterations spent overruling the prior forever (4k of weight over 1M
    # iterations still read -0.048). Weight times scale is a regret
    # magnitude, and Kuhn's per-visit regret is a fraction of a chip, so
    # 4,000 at a scale of 1 is a prior worth tens of thousands of visits;
    # 40,000 took 2M iterations to get within 0.002.
    wrong = [(key, [0.0, 1.0]) for key in converged]        # bet or call everything
    recovered = [_kuhn_value(native.solve_kuhn(400_000, seed, 0, 1, wrong, 4_000, 1.0, "linear"))
                 for seed in range(3)]
    mean = sum(recovered) / len(recovered)
    assert abs(mean - (-1 / 18)) < 0.002, (
        f"warm start from a wrong prior stayed at {mean:+.5f}; seeds {recovered}")

    # Without entries the solver takes the old path exactly, which the golden
    # test pins on the no-limit game; here the cheap form of the same claim.
    assert native.solve_kuhn(50_000, 3) == native.solve_kuhn(50_000, 3, 0, 1, [], 0, 1.0)


def test_pruning_keeps_kuhns_value_and_off_is_the_old_path():
    """
    Regret-based pruning (MCCFR::set_pruning) must not change where the run
    converges: a pruned action is skipped only while its regret is far below
    zero, and the unpruned iterations find it again if it recovers. Kuhn's
    regrets are a few chips at most, so a threshold of -20 prunes the plainly
    dominated actions after a short warm-up. With pruning off the call must be
    the old path exactly, which the no-limit golden test also pins.
    """
    values = [_kuhn_value(native.solve_kuhn(200_000, seed, 0, 1, [], 0, 1.0, "vanilla", "proportional",
                                            20_000, -20.0))
              for seed in range(3)]
    mean = sum(values) / len(values)
    assert abs(mean - (-1 / 18)) < 0.003, f"with pruning converged to {mean:+.5f}; seeds {values}"
    assert native.solve_kuhn(50_000, 3) == native.solve_kuhn(50_000, 3, 0, 1, [], 0, 1.0, "vanilla", "proportional", -1, 0.0)


def test_skipping_the_early_average_still_reaches_kuhns_value():
    """
    With the average accumulated only from a quarter of the way in, Kuhn's
    value must still be −1/18: the regret bound holds on any suffix window
    (Brown and Sandholm 2019), and the early, uniform visits it drops are the
    ones that kept rarely reached nodes near 50/50.
    """
    import itertools

    def game_value(strategy):
        def probability(key, action):
            entry = strategy.get(key)
            return entry[action] if entry else 0.5
        total = 0.0
        for mine, theirs in itertools.permutations(range(3), 2):
            showdown = 1.0 if mine > theirs else -1.0
            value = probability(f"{mine}|", 0) * (
                probability(f"{theirs}|p", 0) * showdown
                + probability(f"{theirs}|p", 1) * (
                    probability(f"{mine}|pb", 0) * -1.0
                    + probability(f"{mine}|pb", 1) * 2.0 * showdown))
            value += probability(f"{mine}|", 1) * (
                probability(f"{theirs}|b", 0) * 1.0
                + probability(f"{theirs}|b", 1) * 2.0 * showdown)
            total += value / 6.0
        return total

    values = [game_value(native.solve_kuhn(200_000, seed, average_from=50_000)) for seed in range(3)]
    mean = sum(values) / len(values)
    assert abs(mean - (-1 / 18)) < 0.002, f"converged to {mean:+.5f}; seeds {values}"


def test_the_flat_export_matches_the_map_export():
    """Three arrays instead of a tree, a dict and millions of small arrays; the same numbers."""
    import numpy as np
    from cfr.flat import FlatStrategy
    solver = native.NoLimitSolver([0] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], 3)
    solver.train(300)
    as_map = solver.average_strategy()
    key_bytes, offsets, values = solver.average_strategy_flat(32)
    keys = np.asarray(key_bytes).view("S32").reshape(-1)
    flat = FlatStrategy(keys, np.asarray(offsets), np.asarray(values))
    assert len(flat) == len(as_map)
    for key, row in as_map.items():
        assert np.array_equal(flat[key], np.asarray(row))


def test_one_deal_per_iteration_produces_a_usable_strategy_and_is_off_by_default():
    """
    With common random numbers on, every branch of an iteration sees the same
    deal; the strategy must still be well formed and reach the same nodes.
    The golden test covers the default (off) path, so this only pins the
    switch and the shape; the head-to-head is its gate.
    """
    table = [0] * (52 * 52)
    on = native.NoLimitSolver(table, [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], 0)
    on.set_common_random_numbers(True)
    on.train(300)
    strategy = on.average_strategy()
    assert strategy and on.iterations() == 300
    for key, probabilities in strategy.items():
        assert abs(sum(probabilities) - 1.0) < 1e-9 and all(p >= 0.0 for p in probabilities)
    # Every deal must be nine distinct cards: run through many iterations without a
    # duplicate showing up as an impossible board (the solver would score nonsense).
    on.train(2000)
    assert on.information_sets() > len(strategy)


def test_an_exact_all_in_terminal_equals_the_mean_of_sampled_runouts():
    """
    Both all in on the flop: the exact edge over the 990 runouts must be the
    limit of the sampled runouts, and on the turn over the 44. Aces against
    kings on a dry flop, and a flush draw against top pair, as the checks.
    """
    from abstraction.equity import card_index
    from engine.cards import Card
    def idx(*names):
        return [card_index(Card(n[0].upper(), n[1].lower())) for n in names]
    for mine, theirs, board in ((idx("As", "Ah"), idx("Kd", "Kc"), idx("2c", "7d", "9s")),
                                (idx("Jh", "Th"), idx("Ad", "9c"), idx("9h", "4h", "2s")),
                                (idx("Qs", "Qd"), idx("8c", "7c"), idx("6c", "5d", "Kc", "2h"))):
        exact = native.allin_edge(mine, theirs, board, True)
        sampled = native.allin_edge(mine, theirs, board, False, 40000, 9)
        assert abs(exact - sampled) < 0.02, (exact, sampled)
        assert -1.0 <= exact <= 1.0


def test_the_current_strategy_replaces_only_the_empty_averages_when_asked():
    """
    The two exports may differ only at nodes whose average was a uniform, and
    there the flagged export is regret matching's current strategy, which is
    what the node actually plays. Everywhere else they are identical.
    """
    import numpy as np
    table = [0] * (52 * 52)
    solver = native.NoLimitSolver(table, [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], 11)
    solver.train(40)
    plain = solver.average_strategy()
    solver.set_current_when_empty(True)
    flagged = solver.average_strategy()
    assert set(plain) == set(flagged)
    changed = [k for k in plain if not np.allclose(plain[k], flagged[k])]
    assert changed, "a forty-iteration solve should hold nodes that were never averaged"
    for k in changed:
        n = len(plain[k])
        assert np.allclose(plain[k], [1.0 / n] * n), "only a uniform average may be replaced"
        assert abs(sum(flagged[k]) - 1.0) < 1e-9


def test_a_scripted_opponent_sees_its_cards_and_the_copy_matches_the_python_shape():
    # 22 September: the game kept the deal in one field and the scripted
    # opponent read another, so every best-response solve was against a bot
    # holding "2c 2c" (+1,079 BB/100 in the solver's own game, -102 at the
    # table). The betting state must carry the cards at every opponent
    # decision, and the C++ copy of a shape must decide as the Python does on
    # the same situation with the same equity.
    import pokerbot_native as native
    from chipzen.archetypes import PARAMS
    solver = native.NoLimitSolver([168] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6],
                                  20, 200, 1, 2, [4, 2, 1], 0)
    assert solver.opponent_sees_cards(300, 1) == 1.0
    # The copy's rule for an unopened pot: open above open_eq, otherwise limp
    # above limp_eq, else fold: aces open, seven-deuce offsuit limps.
    params = {k: float(v) for k, v in PARAMS["station"].items()}
    open_ = native.archetype_act(params, 2, 2000, 0, [12, 25], [], [1, 2], [1, 2], [199, 198], "", [0, 1, 2, 3, 4, 5], 3)
    limp = native.archetype_act(params, 2, 2000, 0, [0, 18], [], [1, 2], [1, 2], [199, 198], "", [0, 1, 2, 3, 4, 5], 3)
    assert open_ >= 2 and limp == 1


def test_the_cpp_betting_game_agrees_on_a_schedule_per_street():
    """
    A raise schedule per street (26 Sept), driven across streets.

    The test above never leaves the first street, and a per-street schedule is
    only different after the flop is dealt, so this one deals an empty board
    at every chance node on both sides (the betting never reads the cards) and
    goes to the river: every sequence to depth five, then random longer ones.
    Compared at every step as above: pot, stacks, history and the legal list.
    """
    import itertools
    import random

    from abstraction.betting import (ALL_IN, RAISE_HALF as H, RAISE_POT as P, RAISE_TWO as T,
                                     StreetSchedule, normalise_schedule)
    from abstraction.buckets import CardAbstraction
    from games.nolimit import NoLimitHoldem

    stack, small_blind, big_blind = 200, 1, 2
    abstraction = CardAbstraction(preflop_buckets=2, postflop_buckets=2, samples=20, equity_samples=4)
    abstraction.fit(np.random.default_rng(0))
    today = ((H, P, T, ALL_IN), (T, ALL_IN), (ALL_IN,))
    proposed = StreetSchedule((((H, P, T, ALL_IN), (H, T, ALL_IN), (ALL_IN,)), today, today, today))
    odd = StreetSchedule((((P, ALL_IN),), ((H,), (ALL_IN,)), ((H, P, T, ALL_IN), (T, ALL_IN), (ALL_IN,)), ((ALL_IN,),)))

    def python_replay(actions, schedule):
        game = NoLimitHoldem(abstraction, starting_stack=stack, small_blind=small_blind,
                             big_blind=big_blind, raise_cap=schedule, equity_samples=4)
        state = game.initial_state()._replace(hole=((0, 1), (2, 3)))
        rng = np.random.default_rng(1)
        legal_each = []
        for action in actions:
            while not game.is_terminal(state) and game.current_player(state) < 0:
                state = game.next_state(state, game.sample_chance(state, rng))
            if game.is_terminal(state):
                return -1, None, None, legal_each
            legal = list(game.legal_actions(state))
            legal_each.append(legal)
            if action not in legal:
                return -3, None, None, legal_each
            state = game.next_state(state, action)
        return sum(state.contributions), list(state.stacks), state.history, legal_each

    compared = crossed = 0
    for schedule in (proposed, odd):
        per_street = [[list(level) for level in normalise_schedule(spec)] for spec in schedule.streets]
        random.seed(7)
        sequences = [list(a) for depth in range(1, 6) for a in itertools.product(range(6), repeat=depth)]
        # Whole hands, walked: at each step a legal action from the Python game, mostly
        # checks and calls so the streets close, ending where the hand ends. A random
        # string of actions mostly runs past the end of the hand and tests nothing.
        for _ in range(3000):
            game = NoLimitHoldem(abstraction, starting_stack=stack, small_blind=small_blind,
                                 big_blind=big_blind, raise_cap=schedule, equity_samples=4)
            state, walk, rng = game.initial_state()._replace(hole=((0, 1), (2, 3))), [], np.random.default_rng(1)
            while True:
                while not game.is_terminal(state) and game.current_player(state) < 0:
                    state = game.next_state(state, game.sample_chance(state, rng))
                if game.is_terminal(state):
                    break
                legal = list(game.legal_actions(state))
                action = 1 if (1 in legal and random.random() < 0.7) else random.choice(legal)
                walk.append(action)
                state = game.next_state(state, action)
            sequences.append(walk)
        for actions in sequences:
            expected = python_replay(actions, schedule)
            got = native.replay(actions, stack, small_blind, big_blind, [], [], per_street, True)
            compared += 1
            assert expected[0] == got[0], f"{schedule} {actions}: pot {expected[0]} vs {got[0]}"
            assert [list(l) for l in expected[3]] == [list(l) for l in got[3]], f"{schedule} {actions}: legal lists differ"
            if expected[0] >= 0:
                assert expected[1] == list(got[1]) and expected[2] == got[2], f"{schedule} {actions}: state differs"
                crossed += expected[2].count("/") >= 2
    assert crossed > 1000, f"only {crossed} sequences reached the turn; the test is not testing streets"


def _archetype(name):
    from chipzen.archetypes import PARAMS
    return {k: float(v) for k, v in PARAMS[name].items()}


def _small_solver():
    return native.NoLimitSolver([0] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], 3)


def test_a_one_script_mix_trains_exactly_as_one_archetype():
    """set_opponent_mix with one entry makes no second draw, so the single-archetype recipe is unchanged by it."""
    single, mixed = _small_solver(), _small_solver()
    single.set_opponent_archetype(_archetype("station"), 2, 0.25)
    mixed.set_opponent_mix([(_archetype("station"), 0.25)], 2)
    single.train(300)
    mixed.train(300)
    a, b = single.average_strategy(), mixed.average_strategy()
    assert a.keys() == b.keys()
    assert all(list(a[k]) == list(b[k]) for k in a)


def test_the_mix_refuses_what_it_cannot_mean():
    solver = _small_solver()
    with pytest.raises(Exception):
        solver.set_opponent_mix([], 2)
    with pytest.raises(Exception):
        solver.set_opponent_mix([(_archetype("station"), 0.7), (_archetype("maniac"), 0.5)], 2)
    with pytest.raises(Exception):
        solver.set_opponent_mix([({"no_such_parameter": 1.0}, 0.2)], 2)
    with pytest.raises(Exception):
        solver.set_opponent_mix([(_archetype("station"), 0.0)], 2)


def test_a_second_script_changes_what_is_learned():
    station, both = _small_solver(), _small_solver()
    station.set_opponent_mix([(_archetype("station"), 0.25)], 2)
    both.set_opponent_mix([(_archetype("station"), 0.25), (_archetype("maniac"), 0.15)], 2)
    station.train(300)
    both.train(300)
    a, b = station.average_strategy(), both.average_strategy()
    shared = [k for k in a if k in b]
    assert shared and any(list(a[k]) != list(b[k]) for k in shared)
    assert both.has_opponent_policy() if hasattr(both, "has_opponent_policy") else True


def test_a_mix_can_be_evaluated_against():
    solver = _small_solver()
    solver.set_opponent_mix([(_archetype("station"), 0.25), (_archetype("maniac"), 0.15)], 2)
    solver.train(100)
    value = solver.evaluate_against_policy(2000, 5)
    assert value == value      # a number, not NaN; the trainer prints it at the end of a run


def test_a_named_dcfr_setting_trains_exactly_as_the_rule():
    """'dcfr:1.5,0,2' is the paper's setting, so it must reproduce 'dcfr' bit for bit on the single-threaded path."""
    named, spelled = _small_solver_rule("dcfr"), _small_solver_rule("dcfr:1.5,0,2")
    named.train(300)
    spelled.train(300)
    a, b = named.average_strategy(), spelled.average_strategy()
    assert a.keys() == b.keys() and all(list(a[k]) == list(b[k]) for k in a)


def test_a_dcfr_setting_changes_the_result_and_bad_ones_are_refused():
    base, other = _small_solver_rule("dcfr"), _small_solver_rule("dcfr:1.5,0.5,1")
    base.train(300)
    other.train(300)
    a, b = base.average_strategy(), other.average_strategy()
    assert any(list(a[k]) != list(b[k]) for k in a if k in b)
    for bad in ("dcfr:1.5,0", "dcfr:x,y,z", "dcfr:"):
        with pytest.raises(Exception):
            _small_solver_rule(bad)


def _small_solver_rule(rule):
    return native.NoLimitSolver([0] * (52 * 52), [0.3, 0.6], [0.3, 0.6], [0.3, 0.6], 8, 200, 1, 2, [4, 4], 3, rule=rule)


def _edge_by_python(mine, theirs, board, samples=20000, seed=3):
    """P(win) - P(lose) by the Python evaluator: exact on a full board, sampled otherwise."""
    used = set(mine) | set(theirs) | set(board)
    deck = [c for c in range(52) if c not in used]
    rng = random.Random(seed)
    runs = [[]] if len(board) == 5 else [rng.sample(deck, 5 - len(board)) for _ in range(samples)]
    edge = 0
    for extra in runs:
        full = list(board) + extra
        a = score_hand_7_fast(np.array([c % 13 for c in mine + full]), np.array([c // 13 for c in mine + full]))
        b = score_hand_7_fast(np.array([c % 13 for c in theirs + full]), np.array([c // 13 for c in theirs + full]))
        edge += (a > b) - (a < b)
    return edge / len(runs)


def test_allin_edge_on_every_board_a_hand_can_have():
    """
    Preflop and river as well as flop and turn. On 25 September an empty board
    was scored as a river with nothing on it (T7s read 70% against AK) and a
    full one wrote past the end of the hand arrays; the old test above only
    ever asked about the flop and the turn.
    """
    from abstraction.equity import card_index
    from engine.cards import Card

    def idx(*names):
        return [card_index(Card(n[0].upper(), n[1].lower())) for n in names]
    cases = ((idx("Th", "7h"), idx("Kd", "As"), []),                       # preflop
             (idx("Ts", "Td"), idx("9h", "Kd"), []),
             (idx("As", "Ah"), idx("Kd", "Kc"), idx("2c", "7d", "9s")),     # flop
             (idx("Qs", "Qd"), idx("8c", "7c"), idx("6c", "5d", "Kc", "2h")),  # turn
             (idx("Jh", "Th"), idx("Ad", "9c"), idx("9h", "4h", "2s", "Qh", "3c")))  # river
    for mine, theirs, board in cases:
        exact = native.allin_edge(mine, theirs, board, True)
        assert exact == pytest.approx(-native.allin_edge(theirs, mine, board, True), abs=1e-12), board
        reference = _edge_by_python(mine, theirs, board)
        tolerance = 1e-12 if len(board) == 5 else 0.02
        assert abs(exact - reference) <= tolerance, (board, exact, reference)
        sampled = native.allin_edge(mine, theirs, board, False, 40000, 9)
        assert abs(exact - sampled) < 0.02, (board, exact, sampled)


@pytest.mark.parametrize("board", [[30], [30, 31], [30, 31, 32, 33, 34, 35]])
def test_allin_edge_refuses_a_board_no_hand_can_have(board):
    with pytest.raises(ValueError):
        native.allin_edge([0, 1], [2, 3], board, True)


def test_allin_edge_refuses_a_card_dealt_twice():
    with pytest.raises(ValueError):
        native.allin_edge([0, 1], [1, 3], [], True)
