"""
The chance tables must be the real dealer, seen through the abstraction.

A full-width solver will trust these tables instead of dealing cards, so a
table that disagrees with the dealer would put the solver in a different game
from the one the bot plays. Three checks: the tables are internally
consistent (a player cannot lose a flush draw's texture class from one street
to the next), the state marginals agree with fresh deals, and the showdown
rates by river state agree with fresh deals. Kept small so the suite stays
under its budget; the million-deal tables in results/cfr/chance are checked
the same way by hand.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.cfr import chance_tables  # noqa: E402

RUNG = os.path.join(os.path.dirname(__file__), "..", "results", "cfr", "ladder169l", "nolimit_18bb.pkl")


@pytest.fixture(scope="module")
def tables():
    if not os.path.exists(RUNG):
        pytest.skip("the 18bb rung is not on this machine")
    chance_tables._init(RUNG)
    return chance_tables.count((7, 3000, RUNG)), chance_tables.count((11, 3000, RUNG))


def test_texture_never_falls_between_streets(tables):
    (_, _, turn, river, _), _ = tables
    strengths = int(round((turn.shape[0] / 6) ** 0.5))
    per_texture = strengths * strengths
    for table in (turn, river):
        for src in range(table.shape[0]):
            for dst in range(table.shape[1]):
                if table[src, dst] and (dst // per_texture) // 2 < (src // per_texture) // 2:
                    raise AssertionError(f"flush class fell from state {src} to {dst}")


def test_two_samples_agree_on_marginals_and_showdowns(tables):
    (pre_a, _, turn_a, _, show_a), (pre_b, _, turn_b, _, show_b) = tables
    # Preflop bucket marginal for player 0: two independent 3,000-deal samples
    # over 169 buckets; the summed absolute difference of the two frequency
    # vectors is about sqrt(2 * 169 / 3000) in expectation, so 0.4 is generous.
    m_a = pre_a.sum(axis=1) / pre_a.sum()
    m_b = pre_b.sum(axis=1) / pre_b.sum()
    assert np.abs(m_a - m_b).sum() < 0.4
    # Flop state marginal, same idea over 216 states.
    f_a = turn_a.sum(axis=1) / turn_a.sum()
    f_b = turn_b.sum(axis=1) / turn_b.sum()
    assert np.abs(f_a - f_b).sum() < 0.45
    # Showdown: player 0 wins about half of all deals, and by state the two
    # samples agree where both saw the state at least 30 times.
    for show in (show_a, show_b):
        assert abs(show[:, 0].sum() / show.sum() - 0.5) < 0.03
    seen = (show_a.sum(axis=1) >= 30) & (show_b.sum(axis=1) >= 30)
    win_a = show_a[seen, 0] / show_a[seen].sum(axis=1)
    win_b = show_b[seen, 0] / show_b[seen].sum(axis=1)
    assert seen.sum() > 10
    assert np.abs(win_a - win_b).mean() < 0.12
    # And the states are ordered: a higher strength class for player 0 at the
    # same texture and opponent class wins more often, on average over states.
    strengths = int(round((show_a.shape[0] / 6) ** 0.5))
    total = show_a + show_b
    by_strength = np.zeros(strengths)
    weight = np.zeros(strengths)
    for state in range(total.shape[0]):
        s0 = (state // strengths) % strengths
        by_strength[s0] += total[state, 0]
        weight[s0] += total[state].sum()
    rate = by_strength[weight > 0] / weight[weight > 0]
    assert np.all(np.diff(rate) > -0.05), f"win rate by strength class not increasing: {rate}"
