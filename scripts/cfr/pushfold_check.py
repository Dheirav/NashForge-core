"""
Check the short rungs against the exact heads-up push-or-fold equilibrium.

    venv/bin/python scripts/cfr/pushfold_check.py --ladder-dir results/cfr/ladder169l --depths 5 8 12
    venv/bin/python scripts/cfr/pushfold_check.py --ladder-dir results/cfr/ladder200t --depths 5 8

Under blinds that rise every twenty hands, most of a match is played short and
its end is push or fold. That game is small enough to solve exactly: the small
blind shoves or folds, the big blind calls or folds, 169 hands a side. The
equilibrium is found by fictitious play over hand-versus-hand equities (Monte
Carlo, cached), which is how the published Nash push/fold charts are built.

What is compared, per rung: the solver's small-blind shove range at the root
against the equilibrium's, its big-blind calling range against a shove, and
the chips its small-blind strategy gives up against a best-responding big
blind inside the push/fold game (its exploitability there, in bb per hand).
A raise that is not all-in is counted as a shove for the comparison, because
at these depths a raise commits the stack; the report says how often that
happened. The solver also limps, which the push/fold game has no word for.
"""
import argparse
import itertools
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, schedule_from_args  # noqa: E402
from abstraction.buckets import canonical_preflop_hands  # noqa: E402
from abstraction.equity import FULL_DECK  # noqa: E402
from engine.cards import RANKS, Card  # noqa: E402
from engine.hand_eval_fast import score_hand_7_fast  # noqa: E402
from numba import njit  # noqa: E402


@njit(cache=True)
def _pair_equity(ci0, ci1, cj0, cj1, boards):
    """Wins and ties for hand (ci0, ci1) over hand (cj0, cj1) on the given boards."""
    wins = 0
    ties = 0
    r = np.empty(7, dtype=np.int64)
    s = np.empty(7, dtype=np.int64)
    for b in range(boards.shape[0]):
        for k in range(5):
            r[k] = boards[b, k] % 13
            s[k] = boards[b, k] // 13
        r[5] = ci0 % 13; s[5] = ci0 // 13; r[6] = ci1 % 13; s[6] = ci1 // 13
        a = score_hand_7_fast(r, s)
        r[5] = cj0 % 13; s[5] = cj0 // 13; r[6] = cj1 % 13; s[6] = cj1 // 13
        c = score_hand_7_fast(r, s)
        if a > c:
            wins += 1
        elif a == c:
            ties += 1
    return wins, ties
from evaluation.benchmark import _solver_actions  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(ROOT, "results", "cfr", "equity_tables", "preflop_169_vs_169.npz")
SUITS = "shdc"


def combos(hand):
    """Every concrete two-card combo of a canonical hand, as deck indices."""
    high, low, suited = hand
    hi, lo = RANKS.index(high), RANKS.index(low)
    out = []
    for a in range(4):
        for b in range(4):
            if high == low and b <= a:
                continue
            if high != low and suited != (a == b):
                continue
            out.append((a * 13 + hi, b * 13 + lo))
    return out


def equity_matrix(samples=600, seed=0):
    """
    E[i, j]: hand i's equity against hand j, and W[i, j]: how many concrete
    combos of j exist given a combo of i (card removal), averaged over i's combos.
    """
    if os.path.exists(CACHE):
        data = np.load(CACHE)
        if int(data["samples"]) >= samples:
            return data["E"], data["W"]
    hands = canonical_preflop_hands()
    n = len(hands)
    rng = np.random.default_rng(seed)
    E = np.zeros((n, n))
    W = np.zeros((n, n))
    all_combos = [combos(h) for h in hands]
    started = time.perf_counter()
    for i in range(n):
        ci = all_combos[i][0]                       # one representative; suits are symmetric
        for j in range(n):
            compatible = [cj for cj in all_combos[j] if not ({cj[0], cj[1]} & {ci[0], ci[1]})]
            W[i, j] = len(compatible)
            if not compatible:
                continue
            wins = ties = total = 0
            per = max(1, samples // len(compatible))
            for cj in compatible:
                used = {ci[0], ci[1], cj[0], cj[1]}
                deck = np.array([k for k in range(52) if k not in used])
                # `per` boards at once: the first five of a random ordering of the deck.
                order = np.argsort(rng.random((per, deck.size)), axis=1)[:, :5]
                boards = deck[order].astype(np.int64)
                w, t = _pair_equity(ci[0], ci[1], cj[0], cj[1], boards)
                wins += w
                ties += t
                total += per
            E[i, j] = (wins + 0.5 * ties) / total
        if (i + 1) % 20 == 0:
            taken = time.perf_counter() - started
            print(f"  equity rows {i + 1}/{n}  {taken:.0f}s  eta {(n - i - 1) * taken / (i + 1) / 60:.1f} min", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, E=E, W=W, samples=samples)
    return E, W


def solve_pushfold(E, W, stack_bb, iterations=3000):
    """
    Fictitious play on the push/fold game at `stack_bb` effective (blinds 0.5
    and 1 already posted out of it). Returns the small blind's push
    probability and the big blind's call probability per hand, as averages.
    """
    n = E.shape[0]
    freq = W / W.sum(axis=1, keepdims=True)            # P(opponent holds j | we hold i)
    S = stack_bb
    push = np.full(n, 0.5)
    call = np.full(n, 0.5)
    push_avg = np.zeros(n)
    call_avg = np.zeros(n)
    for t in range(1, iterations + 1):
        # Big blind best response to the current push range: call iff equity vs
        # the pushing range makes calling (2S * eq - S) better than folding (-1).
        # Equity against the pushing range, from the big blind's seat: hand j
        # vs pusher i, weighted by the pusher's push frequency and card removal.
        # From the big blind's seat holding j: P(pusher holds i and pushes), and
        # E[j, i] is j's equity against i (E is indexed [holder, opponent]).
        push_range = freq * push[None, :]
        mass = push_range.sum(axis=1)
        eq_vs_push = np.where(mass > 0, (E * push_range).sum(axis=1) / np.where(mass > 0, mass, 1), 0.5)
        best_call = (2 * S * eq_vs_push - S >= -1.0).astype(float)
        # Small blind best response: push EV = P(fold) * 1 + P(call) * (2S * eq - S); fold = -0.5.
        call_range = freq * call[None, :]
        p_call = call_range.sum(axis=1)
        eq_vs_call = np.where(p_call > 0, (E * call_range).sum(axis=1) / np.where(p_call > 0, p_call, 1), 0.5)
        push_ev = (1 - p_call) * 1.0 + p_call * (2 * S * eq_vs_call - S)
        best_push = (push_ev >= -0.5).astype(float)
        push = push + (best_push - push) / t
        call = call + (best_call - call) / t
        push_avg += push
        call_avg += call
    return push_avg / iterations, call_avg / iterations


def sb_value(E, W, push, call, S):
    """Small blind's EV per hand (bb) of a push strategy against a call strategy."""
    freq = W / W.sum(axis=1, keepdims=True)
    call_range = freq * call[None, :]
    p_call = call_range.sum(axis=1)
    eq = np.where(p_call > 0, (E * call_range).sum(axis=1) / np.where(p_call > 0, p_call, 1), 0.5)
    push_ev = (1 - p_call) * 1.0 + p_call * (2 * S * eq - S)
    per_hand = push * push_ev + (1 - push) * (-0.5)
    weights = W.sum(axis=1) / W.sum()                    # how often each hand is dealt
    return float((per_hand * weights).sum()), per_hand


def solver_ranges(path, hands):
    """The solver's shove-or-raise probability from the small blind and its call vs a shove."""
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    strategy, abstraction = saved["strategy"], saved["abstraction"]
    args = saved.get("args") or {}
    cap = args.get("raise_cap", 1)
    schedule = schedule_from_args(cap)
    push = np.zeros(len(hands))
    shove_only = np.zeros(len(hands))
    call = np.zeros(len(hands))
    limp = np.zeros(len(hands))
    for k, (high, low, suited) in enumerate(hands):
        hole = [Card(high, "h"), Card(low, "h" if suited else "d")]
        bucket = abstraction.bucket(hole, [], np.random.default_rng(0))
        root = strategy.get(f"{bucket}|")
        actions = _solver_actions("", 1, schedule)
        if root is not None and len(actions) == root.size:
            dist = dict(zip(actions, root))
            shove_only[k] = dist.get(ALL_IN, 0.0)
            push[k] = sum(p for a, p in dist.items() if a >= 2)
            limp[k] = dist.get(CHECK_CALL, 0.0)
        facing = strategy.get(f"{bucket}|5")
        actions = _solver_actions("5", 1, schedule)
        if facing is not None and len(actions) == facing.size:
            call[k] = dict(zip(actions, facing)).get(CHECK_CALL, 0.0)
    return push, shove_only, call, limp


def name(hand):
    high, low, suited = hand
    return f"{high}{low}{'s' if suited else ('' if high == low else 'o')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder-dir", default=os.path.join(ROOT, "results", "cfr", "ladder169l"))
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 8, 12])
    parser.add_argument("--prefix", default="nolimit", help="rung file prefix: nolimit or cap2")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "cfr", "pushfold_check.md"))
    args = parser.parse_args()
    hands = canonical_preflop_hands()
    print("equity matrix (cached after the first run)...", flush=True)
    E, W = equity_matrix(args.samples)
    lines = [f"# Push-or-fold check: {os.path.relpath(args.ladder_dir, ROOT)}", "",
             "Small blind shoves or folds, big blind calls or folds, 169 hands a side, solved by fictitious "
             "play on Monte Carlo equities. The solver's raises of any size count as shoves here.", ""]
    for depth in args.depths:
        path = os.path.join(args.ladder_dir, f"{args.prefix}_{depth}bb.pkl")
        if not os.path.exists(path):
            lines.append(f"## {depth}bb: no rung at {path}")
            continue
        nash_push, nash_call = solve_pushfold(E, W, depth)
        push, shove_only, call, limp = solver_ranges(path, hands)
        nash_value, _ = sb_value(E, W, nash_push, nash_call, depth)
        # Exploitability of the solver's small blind inside the game: its value
        # against the big blind's best response to it.
        freq = W / W.sum(axis=1, keepdims=True)
        push_range = freq * push[None, :]
        mass = push_range.sum(axis=1)
        eq_vs = np.where(mass > 0, (E * push_range).sum(axis=1) / np.where(mass > 0, mass, 1), 0.5)
        br_call = (2 * depth * eq_vs - depth >= -1.0).astype(float)
        solver_value, per_hand = sb_value(E, W, push, br_call, depth)
        nash_vs_br, _ = sb_value(E, W, nash_push, br_call, depth)
        weights = W.sum(axis=1) / W.sum()
        push_pct_nash = float((nash_push * weights).sum())
        push_pct_solver = float((push * weights).sum())
        call_pct_nash = float((nash_call * weights).sum())
        call_pct_solver = float((call * weights).sum())
        agree_push = float(((nash_push > 0.5) == (push > 0.5)) @ weights)
        agree_call = float(((nash_call > 0.5) == (call > 0.5)) @ weights)
        wrong_push = [name(h) for k, h in enumerate(hands) if nash_push[k] > 0.5 and push[k] < 0.5]
        wrong_fold = [name(h) for k, h in enumerate(hands) if nash_push[k] < 0.5 and push[k] > 0.5]
        wrong_call = [name(h) for k, h in enumerate(hands) if nash_call[k] > 0.5 and call[k] < 0.5]
        wrong_callx = [name(h) for k, h in enumerate(hands) if nash_call[k] < 0.5 and call[k] > 0.5]
        lines += [f"## {depth}bb rung: {os.path.basename(path)}", "",
                  f"| | equilibrium | solver |", "|---|---|---|",
                  f"| small blind shoves (share of hands) | {push_pct_nash:.0%} | {push_pct_solver:.0%} (all-in only {float((shove_only * weights).sum()):.0%}, limps {float((limp * weights).sum()):.0%}) |",
                  f"| big blind calls a shove | {call_pct_nash:.0%} | {call_pct_solver:.0%} |",
                  f"| hands where the two agree, weighted | shove {agree_push:.0%}, call {agree_call:.0%} | |",
                  f"| small blind value vs a best-responding big blind (bb/hand) | {nash_vs_br:+.3f} (equilibrium {nash_value:+.3f}) | {solver_value:+.3f} |",
                  "",
                  f"Solver folds where the equilibrium shoves ({len(wrong_push)}): {' '.join(wrong_push[:40])}",
                  f"Solver shoves where the equilibrium folds ({len(wrong_fold)}): {' '.join(wrong_fold[:40])}",
                  f"Solver folds to a shove where the equilibrium calls ({len(wrong_call)}): {' '.join(wrong_call[:40])}",
                  f"Solver calls a shove where the equilibrium folds ({len(wrong_callx)}): {' '.join(wrong_callx[:40])}", ""]
        print("\n".join(lines[-12:]), flush=True)
    text = "\n".join(lines)
    with open(args.out, "w") as handle:
        handle.write(text + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
