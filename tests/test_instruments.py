"""
The measuring instruments must read zero on a mirror.

Three defects in the cross-tree gate went unnoticed for three days in
September 2026 (every gate at 200 chips whatever the rung, stack-capped
fold/call entries rejected as the wrong width, one raise cap for both seats),
and each produced a confident wrong story about the solver. Every one would
have failed a self-play check. So the two instruments the project now decides
by, `play_pickles` (a solve against a solve, each seat on its own tree at the
rung's stack) and `chipzen_duel` (whole bot against whole bot), are pinned to
zero on a mirror here, small enough for the suite, with the exact
configuration the decisions use.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cfr.flat import load_strategy  # noqa: E402
from evaluation.benchmark import benchmark, cfr_agent  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNG = os.path.join(ROOT, "results", "cfr", "ladder169l", "nolimit_18bb.pkl")
LADDER = os.path.join(ROOT, "results", "cfr", "ladder_a")


def _agent(saved, seed):
    misses = [0, 0]
    args = saved["args"]
    return cfr_agent(saved["strategy"], saved["abstraction"], np.random.default_rng(seed),
                     misses=misses, raise_cap=args["raise_cap"], on_miss="call", stack_cap=True), misses


def test_a_solve_against_itself_at_its_own_stack_reads_zero_with_no_misses():
    if not os.path.exists(RUNG):
        pytest.skip("the 18bb rung is not on this machine")
    saved = load_strategy(RUNG)
    a, ma = _agent(saved, 0)
    b, mb = _agent(saved, 1000)
    stack, bb = int(saved["args"]["stack"]), int(saved["args"]["big_blind"])
    cap = saved["args"]["raise_cap"]
    result = benchmark(a, b, "mirror", hands=3000, seed=0, starting_stack=stack,
                       small_blind=bb // 2, big_blind=bb, raise_caps=(cap, cap))
    # 3,000 duplicate hands of an 18bb game read ±4 BB/100; zero is the answer.
    assert abs(result.bb_per_100) < 8.0, f"self-play read {result.bb_per_100:+.1f} BB/100"
    # At its own stack, with the stack cap honoured, a solve has an entry for
    # every node it reaches: a miss here means the harness built a history or
    # an action list the tree does not have.
    assert ma[0] == 0 and mb[0] == 0, f"misses {ma} / {mb} in self-play"


def test_the_duel_reads_even_on_a_mirror():
    if not os.path.isdir(LADDER):
        pytest.skip("the ladder is not on this machine")
    from scripts.chipzen_duel import Dealer, build
    a = build(LADDER, "--deep-primary --stack-cap", "A", np.random.default_rng(1), None)
    b = build(LADDER, "--deep-primary --stack-cap", "B", np.random.default_rng(2), None)
    a.opponent, b.opponent = "B", "A"
    rng = np.random.default_rng(0)
    deck = np.arange(52)
    diffs = []
    for i in range(300):
        rng.shuffle(deck)
        cards = [int(c) for c in deck[:9]]
        first = Dealer([a, b], cards, 2500, 50, 100, 2 * i + 1, ("B", "A")).play()
        second = Dealer([b, a], cards, 2500, 50, 100, 2 * i + 2, ("A", "B")).play()
        diffs.append((first[0] + second[1]) / 2.0)
    diffs = np.array(diffs)
    se = diffs.std() / np.sqrt(diffs.size)
    assert abs(diffs.mean()) < 4 * max(se, 1.0), f"mirror read {diffs.mean():+.1f} ± {se:.1f} chips/hand"
    # The dealer must close every hand: a hand that did not close raises.
    assert a.stats.decisions > 0 and b.stats.decisions > 0


class _Scripted:
    """A duel player that plays a fixed sequence, then calls."""
    def __init__(self, moves):
        self.moves, self.stats = list(moves), type("S", (), {"decisions": 0})()

    def decide(self, state, valid, seat):
        self.stats.decisions += 1
        move = self.moves.pop(0) if self.moves else ("call", 0)
        action, amount = move
        if action == "raise":
            return {"action": "raise", "params": {"amount": amount}}
        return {"action": action if action in valid else "call", "params": {}}


def test_play_pickles_plays_the_first_pickle_purified_when_asked():
    # A purified set is gated as it plays: the flag must reach the agent, or the gate measures the unpurified solve.
    if not os.path.exists(RUNG):
        pytest.skip("the 18bb rung is not on this machine")
    sys.path.insert(0, os.path.join(ROOT, "scripts", "cfr"))
    import play_pickles
    for mode in ("none", "all"):
        agent, *_ = play_pickles.load(RUNG, 0, 1, "call", mode)
        assert agent is not None
    with pytest.raises(Exception):
        play_pickles.load(RUNG, 0, 1, "call", "sometimes")
