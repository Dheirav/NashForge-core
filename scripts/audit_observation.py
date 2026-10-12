"""
Is the 17-feature observation a good one?

Answered by measurement in three parts.

**A. What is in the vector.** Per-feature standard deviation finds dead slots;
the correlation matrix finds duplicates; PCA gives the effective dimensionality.
The August audit found nine components explaining 90% of a seventeen-slot
vector, and one slot with a standard deviation of exactly zero. Nothing has
re-checked it since the observation was made board-aware.

**B. What is missing.** For every sampled state, the player's true equity is
computed by Monte Carlo and compared against what the vector can recover. The
decisive statistic is not the overall R-squared but the *spread of true equity
among states the agent cannot tell apart*: group by near-identical
`hand_strength` and measure the equity range inside each group. A wide range
means the observation is blind in a way no amount of training repairs — a flush
draw and the same overcards without it are the same input.

**C. Whether the proposed fix helps.** Candidate draw and texture features are
computed and B is re-run with them. If the within-group spread collapses, the
case for rebuilding the feature layer is numerical rather than an appeal to the
audit.

Run for both table sizes, because several slots — position, players_behind,
players_in_hand — are close to constant heads-up, and heads-up is the
configuration being trained first.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/home/dheirav/Code/PokerBot")

import numpy as np

from abstraction.equity import equity_vs_random
from engine import Action, PokerGame, get_feature_names
from engine.features import FeatureCache

STATES = 1500          # decision states sampled per table size
EQUITY_SAMPLES = 400   # Monte Carlo draws per equity estimate
SEED = 11


# ---------------------------------------------------------------------------
# Candidate features the observation does not currently carry
# ---------------------------------------------------------------------------

def draw_features(hole, board):
    """
    Flush draw, open-ended straight draw, paired board, three-to-a-flush board.

    Deliberately cheap: suit counts and rank gaps, no simulation. If these close
    the gap that part B measures, they are worth adding at negligible cost.
    """
    cards = list(hole) + list(board)
    suits = [c.suit for c in cards]
    board_suits = [c.suit for c in board]

    flush_draw = 1.0 if any(suits.count(s) == 4 for s in set(suits)) else 0.0
    three_flush = 1.0 if any(board_suits.count(s) >= 3 for s in set(board_suits)) else 0.0

    order = "23456789TJQKA"
    ranks = sorted({order.index(c.rank) for c in cards})
    open_ended = 0.0
    for i in range(len(ranks) - 3):
        window = ranks[i:i + 4]
        if window[3] - window[0] == 3:          # four to a straight, consecutive
            open_ended = 1.0
            break

    board_ranks = [c.rank for c in board]
    paired_board = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0

    return np.array([flush_draw, open_ended, paired_board, three_flush])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_states(num_players, count, rng):
    """Drive real hands with random legal actions, recording decision states."""
    vectors, equities, extras, streets = [], [], [], []
    seed = int(rng.integers(1 << 30))
    game = PokerGame([1000] * num_players, small_blind=5, big_blind=10, seed=seed)
    caches = {}

    def fresh():
        return PokerGame([1000] * num_players, small_blind=5, big_blind=10,
                         seed=int(rng.integers(1 << 30)))

    while len(vectors) < count:
        # reset_hand() does not restore stacks. Once a player busts, every
        # later hand is instantly over with no player to act, and a loop that
        # only calls reset_hand spins forever producing nothing — silently,
        # with no error. Rebuild the table instead.
        if game.is_hand_over() or min(p.stack for p in game.players) <= 0:
            if min(p.stack for p in game.players) <= 0:
                game = fresh()
            else:
                game.reset_hand()
            caches = {}

        player = game.state.current_player
        if player is None:
            game = fresh()
            caches = {}
            continue

        if player not in caches:
            caches[player] = FeatureCache(game, player)

        vector = np.array(caches[player].get_features(game), dtype=np.float64)
        hole = game.players[player].hole_cards
        board = game.state.community_cards

        vectors.append(vector)
        equities.append(equity_vs_random(hole, board, EQUITY_SAMPLES, rng))
        extras.append(draw_features(hole, board))
        streets.append(game.state.betting_round)

        # Uniform random play ends hands almost immediately — folds and all-ins
        # are a quarter of the action space, so four in five states sampled that
        # way are preflop. Weighting toward continuation buys the postflop
        # states part B needs. This changes which states are looked at, not the
        # relationship between a state's features and its equity.
        legal = game.get_legal_actions(player)
        preference = {"check": 5.0, "call": 5.0, "raise": 3.0,
                      "fold": 0.4, "all-in": 0.2}
        weights = np.array([preference.get(a["type"], 1.0) for a in legal])
        choice = legal[int(rng.choice(len(legal), p=weights / weights.sum()))]
        kind = choice["type"]
        # A raise is offered as a range, not an amount; everything else carries
        # its own. Getting this wrong makes every raise illegal, which shows up
        # as a hand that resets forever rather than as an error.
        if kind == "raise":
            low, high = choice["min"], choice["max"]
            amount = int(rng.integers(low, high + 1)) if high > low else low
        else:
            amount = choice.get("amount", 0)
        try:
            game.apply_action(player, Action(kind, amount))
        except Exception:
            game = fresh()
            caches = {}

    return (np.array(vectors), np.array(equities), np.array(extras),
            np.array(streets))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def r_squared(design, target):
    """R^2 of an ordinary least-squares fit, with an intercept."""
    matrix = np.column_stack([np.ones(len(design)), design])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficients
    total = ((target - target.mean()) ** 2).sum()
    return 1.0 - (residual ** 2).sum() / total if total > 0 else float("nan")


def report(num_players, rng):
    label = "heads-up" if num_players == 2 else f"{num_players}-handed"
    print(f"\n{'=' * 74}\n{label.upper()}  —  {STATES:,} decision states\n{'=' * 74}")

    vectors, equities, extras, streets = sample_states(num_players, STATES, rng)
    names = get_feature_names()

    # --- A. what is in the vector -----------------------------------------
    print("\nA. WHAT IS IN THE VECTOR\n")
    deviations = vectors.std(axis=0)
    dead = [(n, d) for n, d in zip(names, deviations) if d < 0.02]
    print(f"{'feature':<22}{'std':>9}")
    print("-" * 31)
    for name, deviation in sorted(zip(names, deviations), key=lambda x: x[1]):
        flag = "  <- carries nothing" if deviation < 0.02 else ""
        print(f"{name:<22}{deviation:>9.4f}{flag}")

    live = deviations > 1e-9
    correlations = np.corrcoef(vectors[:, live], rowvar=False)
    live_names = [n for n, keep in zip(names, live) if keep]
    duplicates = []
    for i in range(len(live_names)):
        for j in range(i + 1, len(live_names)):
            if abs(correlations[i, j]) > 0.90:
                duplicates.append((live_names[i], live_names[j], correlations[i, j]))

    print(f"\nPairs correlated above |0.90| — one of each carries no new information:")
    if duplicates:
        for a, b, r in sorted(duplicates, key=lambda x: -abs(x[2])):
            print(f"  {a:<20} {b:<20} r = {r:+.3f}")
    else:
        print("  none")

    centred = vectors[:, live] - vectors[:, live].mean(axis=0)
    _, singular, _ = np.linalg.svd(centred, full_matrices=False)
    explained = np.cumsum(singular ** 2) / (singular ** 2).sum()
    needed = int(np.searchsorted(explained, 0.90) + 1)
    print(f"\nEffective dimensionality: {needed} components explain 90% of the "
          f"variance in {len(names)} slots ({int(live.sum())} of them live).")

    # --- B. what is missing ------------------------------------------------
    print("\n\nB. WHAT IS MISSING — can the vector recover true equity?\n")
    strength = vectors[:, names.index("hand_strength")]
    print(f"  R² from all {len(names)} features        {r_squared(vectors[:, live], equities):.3f}")
    print(f"  R² from hand_strength alone       {r_squared(strength.reshape(-1, 1), equities):.3f}")
    others = [i for i, n in enumerate(names) if n != "hand_strength" and live[i]]
    print(f"  R² with hand_strength removed     {r_squared(vectors[:, others], equities):.3f}")

    postflop = streets != "preflop"
    if postflop.sum() > 40:
        print(f"\n  Postflop only ({int(postflop.sum()):,} states) — where a draw exists:")
        print(f"    R² from all features            {r_squared(vectors[postflop][:, live], equities[postflop]):.3f}")
        print(f"    R² from hand_strength alone     {r_squared(strength[postflop].reshape(-1, 1), equities[postflop]):.3f}")

    # Group by every card feature the vector carries, not by hand_strength
    # alone. When hand_strength *was* the whole card signature the two were the
    # same thing; once draw and texture features exist, grouping on strength
    # alone measures a signature the agent no longer has, and understates it.
    card_names = [n for n in ("hand_strength", "flush_draw", "open_ended",
                              "paired_board", "three_flush") if n in names]
    card_columns = [names.index(n) for n in card_names]
    print(f"\n  States the agent CANNOT TELL APART — equity spread within each"
          f"\n  group sharing all {len(card_names)} card features (postflop):\n")
    buckets = defaultdict(list)
    for index in np.flatnonzero(postflop):
        signature = tuple(round(float(vectors[index, c]), 2) for c in card_columns)
        buckets[signature].append(equities[index])
    spreads = [(key, np.ptp(vals), np.std(vals), len(vals))
               for key, vals in buckets.items() if len(vals) >= 15]
    spreads.sort(key=lambda x: -x[1])
    print(f"    {'card signature':>26}{'n':>6}{'range':>9}{'std':>8}")
    print("    " + "-" * 45)
    for key, span, deviation, n in spreads[:6]:
        shown = " ".join(f"{value:g}" for value in key)
        print(f"    {shown:>26}{n:>6}{span:>9.3f}{deviation:>8.3f}")
    if spreads:
        worst = max(s[1] for s in spreads)
        typical = float(np.mean([s[2] for s in spreads]))
        print(f"\n    Worst group spans {worst:.3f} of equity; typical within-group"
              f" std {typical:.3f}.")

    # --- C. does the proposed fix help? ------------------------------------
    print("\n\nC. DOES ADDING DRAW AND TEXTURE FEATURES HELP?\n")
    augmented = np.column_stack([vectors[:, live], extras])
    print(f"  R² all features                   {r_squared(vectors[:, live], equities):.3f}")
    print(f"  R² + draw/texture features        {r_squared(augmented, equities):.3f}")
    if postflop.sum() > 40:
        print(f"  R² postflop                       {r_squared(vectors[postflop][:, live], equities[postflop]):.3f}")
        print(f"  R² postflop + draw/texture        {r_squared(augmented[postflop], equities[postflop]):.3f}")

    if spreads:
        residual_buckets = defaultdict(list)
        design = np.column_stack([np.ones(postflop.sum()), extras[postflop]])
        coefficients, *_ = np.linalg.lstsq(design, equities[postflop], rcond=None)
        corrected = equities[postflop] - design @ coefficients + equities[postflop].mean()
        for value, equity in zip(strength[postflop], corrected):
            residual_buckets[round(float(value), 2)].append(equity)
        after = [np.std(v) for v in residual_buckets.values() if len(v) >= 25]
        if after:
            print(f"\n  Within-group equity std, postflop:"
                  f"  {typical:.3f}  ->  {float(np.mean(after)):.3f}"
                  f"   after accounting for draws and texture")


def main():
    rng = np.random.default_rng(SEED)
    print(f"Feature audit — {STATES:,} states per table size, "
          f"{EQUITY_SAMPLES} equity samples each")
    for num_players in (2, 6):
        report(num_players, rng)
    print()


if __name__ == "__main__":
    main()
