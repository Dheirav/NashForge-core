"""
Milestone 4: card abstraction and the size of the abstract game.

Abstraction is where a solver stops being exact, so the properties worth
asserting are the ones that keep it *honest*: equity must order hands the way
poker does, buckets must be monotone in strength, and the information-set count
must be arithmetic rather than a guess.

Equity is checked against hands whose values are well known — aces beat a random
hand about 85% of the time — because a subtly wrong equity produces buckets that
look plausible and rank hands wrongly, which no downstream test would catch.

    python -m pytest tests/test_abstraction.py -q
"""
import random

import numpy as np
import pytest

from abstraction.betting import (ALL_IN, CHECK_CALL, FOLD, RAISE_ACTIONS,
                                 RAISE_POT, RAISE_TWO, STREETS,
                                 count_decision_points, enumerate_street_sequences,
                                 legal_actions, measure, normalise_schedule,
                                 raise_sizes_at)
from abstraction.buckets import (CardAbstraction, canonical_preflop_hands,
                                 preflop_key, _fit_kmeans_1d)
from abstraction.equity import FULL_DECK, equity_vs_random, remaining_deck
from engine.cards import Card


def C(text):
    return Card(text[0], text[1])


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def test_deck_is_complete_and_distinct():
    assert len(FULL_DECK) == 52
    assert len({(c.rank, c.suit) for c in FULL_DECK}) == 52


def test_remaining_deck_excludes_known_cards():
    known = [C("Ah"), C("Kd"), C("2c")]
    remaining = remaining_deck(known)
    assert remaining.size == 49
    excluded = {FULL_DECK[i] for i in remaining}
    assert all(card not in excluded for card in known)


@pytest.mark.parametrize("hand,low,high", [
    (["Ah", "Ad"], 0.80, 0.90),      # pocket aces: about 85% against a random hand
    (["Kh", "Kd"], 0.75, 0.87),
    (["Ah", "Kh"], 0.60, 0.72),      # suited big cards
    (["7h", "2d"], 0.28, 0.40),      # the worst starting hand
])
def test_preflop_equity_matches_known_values(hand, low, high):
    """
    A wrong equity produces buckets that look entirely reasonable and rank hands
    incorrectly, so it is pinned against values that are independently known.
    """
    cards = [C(text) for text in hand]
    measured = np.mean([
        equity_vs_random(cards, [], 600, np.random.default_rng(seed))
        for seed in range(3)
    ])
    assert low <= measured <= high, f"{hand}: {measured:.3f}"


def test_equity_responds_to_the_board():
    """The same hole cards must be worth different amounts on different boards."""
    hole = [C("9h"), C("9c")]
    rng = lambda: np.random.default_rng(0)
    flopped_set = equity_vs_random(hole, [C("2c"), C("7s"), C("9d")], 600, rng())
    overcards = equity_vs_random(hole, [C("Ac"), C("Ks"), C("Qd")], 600, rng())
    assert flopped_set > overcards + 0.2


def test_equity_is_bounded():
    for hole, board in [(["Ah", "Ad"], []), (["7h", "2d"], ["Ac", "Kc", "Qc"])]:
        value = equity_vs_random([C(t) for t in hole], [C(t) for t in board],
                                 200, np.random.default_rng(0))
        assert 0.0 <= value <= 1.0


def test_equity_is_reproducible_given_a_seed():
    args = ([C("Ah"), C("Kh")], [C("2c"), C("7s"), C("9d")], 200)
    first = equity_vs_random(*args, np.random.default_rng(7))
    second = equity_vs_random(*args, np.random.default_rng(7))
    assert first == second


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

def test_there_are_169_distinct_starting_hands():
    hands = canonical_preflop_hands()
    assert len(hands) == 169
    assert len(set(hands)) == 169
    # 13 pairs, 78 suited, 78 offsuit
    assert sum(1 for _, _, suited in hands if suited) == 78


def test_preflop_key_is_order_and_suit_canonical():
    assert preflop_key([C("Kh"), C("Ah")]) == preflop_key([C("Ah"), C("Kh")])
    assert preflop_key([C("Ah"), C("Kh")]) != preflop_key([C("Ah"), C("Kd")])
    assert preflop_key([C("Ah"), C("Kd")]) == preflop_key([C("As"), C("Kc")])


def test_kmeans_returns_sorted_centroids():
    values = np.array([0.1, 0.12, 0.5, 0.52, 0.9, 0.95])
    centroids = _fit_kmeans_1d(values, 3)
    assert centroids.size == 3
    assert (np.diff(centroids) > 0).all(), "buckets must be ordered by strength"


def test_kmeans_handles_more_buckets_than_values():
    centroids = _fit_kmeans_1d(np.array([0.2, 0.8]), 10)
    assert centroids.size <= 2


@pytest.fixture(scope="module")
def fitted():
    """A small abstraction; sampling is kept low so the suite stays quick."""
    return CardAbstraction(preflop_buckets=5, postflop_buckets=5,
                           samples=150, equity_samples=60).fit(np.random.default_rng(0))


def test_a_pickle_predating_the_lookup_table_still_buckets(fitted):
    """
    An abstraction is pickled so a result outlives the code that produced it,
    which makes every derived field a compatibility hazard. `_centroid_list` was
    added after `results/cfr/nolimit_strategy.pkl` was written, and that file now
    unpickles with it set to None — so every postflop lookup raises rather than
    answering. Restoring must rebuild it, and rebuild it to the *same* clustering
    rather than refitting to a plausible different one.
    """
    aged = fitted.__dict__.copy()
    aged.pop("_centroid_list", None)          # as an older pickle carries it

    revived = CardAbstraction.__new__(CardAbstraction)
    revived.__setstate__(aged)

    hole, board = [C("Ah"), C("Kh")], [C("Qh"), C("7d"), C("2s")]
    assert revived.bucket(hole, board, np.random.default_rng(0)) == \
        fitted.bucket(hole, board, np.random.default_rng(0))


def test_pickling_an_abstraction_round_trips(fitted):
    """The ordinary path must keep working, not just the aged one."""
    import pickle

    revived = pickle.loads(pickle.dumps(fitted))
    hole, board = [C("Ah"), C("Kh")], [C("Qh"), C("7d"), C("2s")]
    assert revived.bucket(hole, board, np.random.default_rng(0)) == \
        fitted.bucket(hole, board, np.random.default_rng(0))
    assert revived.describe() == fitted.describe()


def test_preflop_buckets_order_hands_by_strength(fitted):
    """Aces must not land in a weaker bucket than seven-deuce."""
    aces = fitted.bucket([C("Ah"), C("Ad")], [])
    kings = fitted.bucket([C("Kh"), C("Kd")], [])
    junk = fitted.bucket([C("7h"), C("2d")], [])
    assert aces >= kings > junk


def test_every_starting_hand_gets_a_bucket(fitted):
    for high, low, suited in canonical_preflop_hands():
        hand = [Card(high, "h"), Card(low, "h" if suited else "d")]
        bucket = fitted.bucket(hand, [])
        assert 0 <= bucket < fitted.preflop_buckets


def test_postflop_buckets_order_by_strength(fitted):
    board = [C("2c"), C("7s"), C("9d")]
    rng = lambda: np.random.default_rng(1)
    strong = fitted.bucket([C("9h"), C("9c")], board, rng())   # a set
    weak = fitted.bucket([C("3h"), C("4d")], board, rng())     # nothing
    assert strong > weak


def test_fitting_is_deterministic_given_a_seed():
    """An abstraction is a fixed artifact; it must not vary between builds."""
    def build():
        return CardAbstraction(preflop_buckets=4, postflop_buckets=4,
                               samples=80, equity_samples=40).fit(
                                   np.random.default_rng(3))

    first, second = build(), build()
    for street in ("flop", "turn", "river"):
        assert first._centroids[street].tolist() == second._centroids[street].tolist()


def test_bucket_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        CardAbstraction().bucket([C("Ah"), C("Ad")], [])


# ---------------------------------------------------------------------------
# The betting tree
# ---------------------------------------------------------------------------

def test_folding_is_not_offered_with_nothing_to_call():
    assert FOLD not in legal_actions(raises_so_far=0, facing_bet=False, raise_cap=2)
    assert FOLD in legal_actions(raises_so_far=1, facing_bet=True, raise_cap=2)


def test_raises_stop_at_the_cap():
    assert RAISE_POT in legal_actions(1, True, raise_cap=2)
    assert set(legal_actions(2, True, raise_cap=2)) == {FOLD, CHECK_CALL}


def test_an_all_in_cannot_be_raised():
    """
    There are no chips left to raise with. Treating all-in as an ordinary raise
    inflates the tree with lines that cannot occur — it cut the sequence count
    by a fifth at cap 2.
    """
    assert set(legal_actions(1, True, raise_cap=3, last_action=ALL_IN)) == {FOLD, CHECK_CALL}


def test_every_street_sequence_ends_properly():
    """A street must close on a call, a check-through, or a fold."""
    for sequence, folded in enumerate_street_sequences(2):
        assert sequence, "empty sequence"
        if folded:
            assert sequence[-1] == FOLD
        else:
            assert sequence[-1] == CHECK_CALL
            assert FOLD not in sequence


def test_deeper_caps_give_strictly_larger_trees():
    sizes = [len(enumerate_street_sequences(cap)) for cap in (1, 2, 3)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_decision_points_are_shared_between_players():
    counts = count_decision_points(2)
    assert counts[0] > 0 and counts[1] > 0


# ---------------------------------------------------------------------------
# Sizing arithmetic
# ---------------------------------------------------------------------------

def test_information_sets_scale_linearly_in_buckets():
    small = measure({s: 4 for s in STREETS}, raise_cap=1)
    large = measure({s: 8 for s in STREETS}, raise_cap=1)
    assert large.information_sets == 2 * small.information_sets


def test_information_sets_explode_with_the_raise_cap():
    """
    Betting lines multiply across all four streets, so the cap compounds while
    buckets only scale. This is why the bet abstraction, not the card
    abstraction, decides whether the game fits in memory.
    """
    shallow = measure({s: 8 for s in STREETS}, raise_cap=1)
    deep = measure({s: 8 for s in STREETS}, raise_cap=2)
    assert deep.information_sets > 100 * shallow.information_sets


def test_memory_accounts_for_regret_and_strategy():
    size = measure({s: 8 for s in STREETS}, raise_cap=1, num_actions=6,
                   bytes_per_entry=8)
    assert size.table_bytes == size.information_sets * 6 * 2 * 8


def test_reaching_sequences_compound_by_street():
    size = measure({s: 4 for s in STREETS}, raise_cap=1)
    reaching = size.reaching_sequences
    assert reaching["preflop"] == 1
    assert reaching["flop"] < reaching["turn"] < reaching["river"]


# ---------------------------------------------------------------------------
# The pluggable postflop strength signal
# ---------------------------------------------------------------------------

def test_both_strength_signals_produce_ordered_buckets():
    """
    Whichever signal a bucketing is fitted on, a set must land above nothing.
    made_hand cannot see draws, but it must still rank made hands correctly.
    """
    from abstraction.buckets import STRENGTH_SIGNALS

    board = [C("2c"), C("7s"), C("9d")]
    for signal in STRENGTH_SIGNALS:
        fitted = CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=120,
                                 equity_samples=30, strength=signal).fit(
                                     np.random.default_rng(0))
        strong = fitted.bucket([C("9h"), C("9c")], board, np.random.default_rng(1))
        weak = fitted.bucket([C("3h"), C("4d")], board, np.random.default_rng(1))
        assert strong > weak, signal


def test_fit_and_lookup_share_one_signal():
    """
    A clustering fitted on one signal and queried with another would assign
    every situation to whichever bucket happened to sit near the wrong scale —
    plausible output, meaningless buckets. Both routes go through one method.
    """
    fitted = CardAbstraction(preflop_buckets=4, postflop_buckets=4, samples=80,
                             equity_samples=25, strength="made_hand").fit(
                                 np.random.default_rng(0))
    hole, board = [C("Ah"), C("Kh")], [C("2c"), C("7s"), C("9d")]
    direct = fitted._postflop_strength(hole, board, np.random.default_rng(0))
    from engine.features import made_hand_strength
    assert direct == made_hand_strength(hole, board)


def test_unknown_strength_signal_is_rejected():
    with pytest.raises(ValueError):
        CardAbstraction(strength="vibes")


def test_nearest_centroid_matches_exhaustive_search():
    """The binary search must agree with the argmin it replaced."""
    from abstraction.buckets import _nearest_centroid

    rng = np.random.default_rng(0)
    for _ in range(500):
        centroids = np.sort(rng.random(rng.integers(2, 10)))
        value = float(rng.random())
        expected = int(np.abs(centroids - value).argmin())
        assert _nearest_centroid(centroids.tolist(), value) == expected


def test_an_int_schedule_is_exactly_what_shipped():
    """
    The taper must not move the abstraction every figure was measured in.

    `normalise_schedule(N)` has to reproduce the uniform N-raise tree to the
    information set, or every number this project has published silently refers
    to a different game than the code now builds.
    """
    assert normalise_schedule(1) == (RAISE_ACTIONS,)
    assert normalise_schedule(2) == (RAISE_ACTIONS, RAISE_ACTIONS)
    buckets = {street: 6 for street in STREETS}
    assert measure(buckets, raise_cap=1).information_sets == 49_200
    assert measure(buckets, raise_cap=2).information_sets == 5_040_160 * 6 // 4


def test_tapering_keeps_the_large_sizes_and_buys_depth_cheaply():
    """
    Dropping the small sizes first is the whole point, and it is what makes a
    re-raise affordable: a second raise carrying all four sizes costs 7,560,240
    information sets at six buckets, and carrying only all-in costs 347,136.
    """
    assert raise_sizes_at((4, 1), 1) == (ALL_IN,)
    assert raise_sizes_at((4, 2), 1) == (RAISE_TWO, ALL_IN)
    assert raise_sizes_at((4, 1), 2) == (), "schedule must end after its last rung"

    buckets = {street: 6 for street in STREETS}
    assert measure(buckets, raise_cap=(4, 1)).information_sets == 347_136
    assert measure(buckets, raise_cap=(4, 2)).information_sets == 1_283_568


def test_every_path_narrows_to_the_same_actions_under_a_taper():
    """
    Three modules decide which raises are legal. They agreed by coincidence of
    being written from the same rules, and a tapered schedule is exactly the
    kind of change that splits them. `_constrain` and `legal_mask` mirror
    `legal_actions` rather than calling it, so pin them against it directly.
    """
    import numpy as np
    from evaluation.benchmark import _constrain

    for spec in (1, 2, (4, 1), (4, 2)):
        for depth in range(4):
            for to_call in (0, 4):
                allowed = set(legal_actions(depth, to_call > 0, spec))
                narrowed = _constrain(np.ones(6), to_call, depth, spec)
                got = {a for a in range(6) if narrowed[a]}
                # `_constrain` keeps check/call rather than stranding an actor,
                # so compare only the raises, which is what the taper changes.
                raises_expected = allowed & set(RAISE_ACTIONS)
                raises_got = got & set(RAISE_ACTIONS)
                assert raises_got == raises_expected, (
                    f"spec={spec} depth={depth} to_call={to_call}: "
                    f"benchmark allows {sorted(raises_got)}, "
                    f"tree allows {sorted(raises_expected)}")


def test_the_packed_score_orders_hands_like_the_full_evaluator():
    """
    `score_hand_7` is the hot path's evaluator and it must not have its own
    opinion about poker.

    `equity_vs_random` compares hands with `>` and `==` only, so it does not
    need the five Card objects `evaluate_hand_fast` builds. Skipping them is
    where the solver's 3.4x speedup came from on 11 September, and the entire
    safety of that rests on the two agreeing about order. A disagreement would
    not crash: it would silently shift equities, and therefore buckets, and
    therefore every strategy trained afterwards.

    Decks restricted by rank or by suit are included deliberately, because
    random seven-card hands almost never produce quads or straight flushes and
    a test that never sees them is not testing the branches most likely wrong.
    """
    from engine.cards import Card, RANKS, SUITS
    from engine.hand_eval import RANK_ORDER
    from engine.hand_eval_fast import evaluate_hand_fast, score_hand_7

    suit_index = {suit: i for i, suit in enumerate(SUITS)}

    def packed(cards):
        return score_hand_7(
            np.array([RANK_ORDER[c.rank] for c in cards], dtype=np.int32),
            np.array([suit_index[c.suit] for c in cards], dtype=np.int32))

    decks = [
        [Card(r, s) for s in SUITS for r in RANKS],          # ordinary play
        [Card(r, s) for s in SUITS for r in RANKS[:6]],      # forces quads, boats
        [Card(r, s) for s in SUITS[:2] for r in RANKS],      # forces flushes
    ]
    rng = random.Random(20260911)
    classes = set()
    for deck in decks:
        for _ in range(1500):
            left, right = rng.sample(deck, 7), rng.sample(deck, 7)
            full_left, full_right = evaluate_hand_fast(left), evaluate_hand_fast(right)
            classes.add(full_left.hand_rank)

            assert (packed(left) >> 20) == full_left.hand_rank, (
                f"hand class differs for {[str(c) for c in left]}")

            expected = (full_left > full_right) - (full_left < full_right)
            got = (packed(left) > packed(right)) - (packed(left) < packed(right))
            assert expected == got, (
                f"ordering differs: {[str(c) for c in left]} vs "
                f"{[str(c) for c in right]}")

    assert len(classes) >= 8, f"only saw hand classes {sorted(classes)}"


def test_the_table_evaluator_matches_the_reference_one():
    """
    `score_hand_7_fast` replaces two scans with table reads and must not have
    its own opinion about poker.

    It is what `_rollouts` calls, so a disagreement would not crash: it would
    shift every equity estimate, every bucket, and every strategy trained after
    it. `score_hand_7` stays as the reference the fast one is pinned against,
    the same relationship `score_hand_7` itself has with `evaluate_hand_fast`.

    Restricted decks are included because random seven-card hands almost never
    make quads or straight flushes, and those branches are the ones most likely
    to be wrong.
    """
    from engine.cards import Card, RANKS, SUITS
    from engine.hand_eval import RANK_ORDER
    from engine.hand_eval_fast import score_hand_7, score_hand_7_fast

    suit_index = {suit: i for i, suit in enumerate(SUITS)}
    decks = [
        [Card(r, s) for s in SUITS for r in RANKS],
        [Card(r, s) for s in SUITS for r in RANKS[:6]],
        [Card(r, s) for s in SUITS[:2] for r in RANKS],
    ]
    rng = random.Random(20260911)
    classes = set()
    for deck in decks:
        for _ in range(2000):
            cards = rng.sample(deck, 7)
            ranks = np.array([RANK_ORDER[c.rank] for c in cards], dtype=np.int32)
            suits = np.array([suit_index[c.suit] for c in cards], dtype=np.int32)
            reference = score_hand_7(ranks, suits)
            assert score_hand_7_fast(ranks, suits) == reference, (
                f"table evaluator differs on {[str(c) for c in cards]}")
            classes.add(reference >> 20)
    assert len(classes) >= 8, f"only saw hand classes {sorted(classes)}"


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

from abstraction.buckets import TEXTURE_CLASSES, board_texture  # noqa: E402


def _cards(text):
    from engine.cards import Card
    return [Card(t[0], t[1]) for t in text.split()]


def test_texture_is_zero_on_a_dry_board():
    assert board_texture(_cards("Qc 9s 2d")) == 0
    assert board_texture([]) == 0


def test_three_of_a_suit_and_four_of_a_suit_are_different_classes():
    assert board_texture(_cards("3h Ks 4h 9h")) == 2      # three hearts
    assert board_texture(_cards("3h Ks 4h 9h 5h")) == 4   # four hearts


def test_four_to_a_straight_is_flagged_with_the_ace_both_ways():
    assert board_texture(_cards("Qc 9s Kd Ac Jd")) == 1   # K Q J and A within five ranks... plus 9: A K Q J
    assert board_texture(_cards("Ah 2c 3d 5s")) == 1      # wheel draw, ace low
    assert board_texture(_cards("2c 7d 9h Kd")) == 0


def test_flush_and_straight_combine_into_six_classes():
    assert board_texture(_cards("Qc 9c Kc Jc")) == 5      # four clubs and four to a straight
    assert 0 <= board_texture(_cards("Ts 9s 8s")) < TEXTURE_CLASSES


def test_a_texture_aware_abstraction_keeps_the_strength_part_recoverable():
    from abstraction.buckets import CardAbstraction
    import numpy as np
    plain = CardAbstraction(preflop_buckets=6, postflop_buckets=6, samples=200,
                            equity_samples=40).fit(np.random.default_rng(0))
    aware = CardAbstraction(preflop_buckets=6, postflop_buckets=6, samples=200,
                            equity_samples=40, texture=True).fit(np.random.default_rng(0))
    hole, board = _cards("Kc 9d"), _cards("3h Ks 4h 9h 5h")
    strength = plain.bucket(hole, board, np.random.default_rng(1))
    composite = aware.bucket(hole, board, np.random.default_rng(1))
    assert composite == strength + 6 * 4
    assert aware.strength_of(composite, "river") == strength
    assert aware.num_buckets("river") == 36 and plain.num_buckets("river") == 6


# ----------------------------------------------------------------------
# Lossless preflop
#
# Six Chen classes put KQo, T9s and 77 in one bucket, and on the arena that
# bucket called a three-bet shove 97 percent of the time because it also holds
# TT and AQ. 169 classes cost the tree almost nothing, because the postflop
# key carries only the postflop bucket.


@pytest.fixture(scope="module")
def lossless():
    return CardAbstraction(preflop_buckets=169, postflop_buckets=6,
                           samples=60, equity_samples=8).fit(np.random.default_rng(0))


def test_a_lossless_preflop_keeps_every_starting_hand_apart(lossless):
    classes = {lossless.bucket([Card(h, "h"), Card(l, "h" if s else "d")], [])
               for h, l, s in canonical_preflop_hands()}
    assert len(classes) == 169 and classes == set(range(169))
    assert lossless.num_buckets("preflop") == 169


def test_a_lossless_preflop_still_numbers_weakest_first(lossless):
    aces = lossless.bucket([C("Ah"), C("Ad")], [])
    kings = lossless.bucket([C("Kh"), C("Kd")], [])
    kq = lossless.bucket([C("Kh"), C("Qd")], [])
    junk = lossless.bucket([C("7h"), C("2d")], [])
    assert aces == 168 and aces > kings > kq > junk == 0


def test_a_lossless_preflop_reports_a_coarse_strength_class(lossless):
    """
    The arena player's thresholds are written for six classes, so the hand's
    own index must not leak through `strength_of`: with 169 preflop classes
    every hand would read as the top class and the rules would shove with it.
    """
    for h, l, s in canonical_preflop_hands():
        bucket = lossless.bucket([Card(h, "h"), Card(l, "h" if s else "d")], [])
        assert 0 <= lossless.strength_of(bucket, "preflop") < 6
    top = lossless.strength_of(lossless.bucket([C("Ah"), C("Ad")], []), "preflop")
    bottom = lossless.strength_of(lossless.bucket([C("7h"), C("2d")], []), "preflop")
    assert top == 5 and bottom == 0


def test_a_clustered_preflop_is_its_own_strength_class(fitted):
    bucket = fitted.bucket([C("Kh"), C("Qd")], [])
    assert fitted.strength_of(bucket, "preflop") == bucket


def test_a_lossless_abstraction_survives_pickling_and_an_older_pickle_does_not_break(lossless, fitted):
    import pickle
    revived = pickle.loads(pickle.dumps(lossless))
    assert revived.strength_of(168, "preflop") == 5
    aged = fitted.__dict__.copy()
    aged.pop("_preflop_strength", None)
    old = CardAbstraction.__new__(CardAbstraction)
    old.__setstate__(aged)
    assert old.strength_of(3, "preflop") == 3
