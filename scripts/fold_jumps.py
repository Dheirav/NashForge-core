"""
Leg (d) of the stopping rule: where a hand faces an all-in preflop, what do the
solve's calls and folds cost, and does a stronger hand fold more than the hand
one rank below it?

    venv/bin/python scripts/fold_jumps.py results/cfr/ladder_a/cap2_70bb.pkl
    venv/bin/python scripts/fold_jumps.py rung_10m.pkl rung_20m.pkl      # T, then 2T: the verdict
    venv/bin/python scripts/fold_jumps.py results/cfr/ladder_a   # every rung of a ladder

The stopping rule's other legs average over play. The gate and the head to
head score tens of thousands of hands and LBR scores a best response, so none
of them sees one hand class that is wrong at one node. One such
node in a deep rung once put an off-pattern share of a weak suited hand on an
all-in, and a later polish deleted another. Both were our own shoves, which `scripts/strategy_sweep.py` lists. This is the
other side of the same spot: what we call an all-in with.

A jump in itself is not a defect. At an all-in the right answer is close to a
threshold, call above it and fold below, so two adjacent hands can sit a whole
1.0 apart and both be right. What cannot be right is the order going the wrong
way: a hand that beats its neighbour card for card (AQs against AJs, KK against
QQ, KJo against QJo) folding clearly more often. So the check compares each
class with the class one rank below it in either card, same suitedness, and
flags the pairs where the stronger one folds at least `--threshold` more.

Two things would fill the list with noise and are left out:

* **Classes that do not reach the node.** A class that never makes the raise
  before the all-in has an average strategy there that nothing ever trained,
  and it never plays. Each class's own reach (the product of its own action
  probabilities along the history) must be at least `--min-reach`.
* **Entries still exactly uniform.** These were never visited at all; they are
  counted and reported, not compared.

Own reach is not enough by itself. At `5` (the small blind open-shoves) the big
blind has not acted, so every class has an own reach of 1, and yet the node is
almost never trained, because the solver's small blind almost never shoves
there. On the 70bb (4,2,1) rung at 20M that node and the three where the small
blind's raise is re-raised all-in by a line it rarely meets held 59 of the 70
flagged pairs, at a line probability of 0.002 or less. So each flag is weighted
by how often the line is played: the opponent's line probability (its range's
own reach, by combinations) times the stronger class's own reach.

With two solves of the same tree, at T and 2T, the leg passes when that
weighted excess is no larger at 2T than at T: more iterations must not make the
order worse where the order is played. The raw count is printed beside it.

**The verdict is on chips, not on the order (6 October).** The rank check was
the first version, and pricing its flags showed why it cannot be the verdict.
At 12bb, after a limp and a shove, Q7s folding 97% against Q6s calling 63% was
the most flagged pair on the rung, and it was Q7s that was right: both hands sit
under the 45.8% a call needs, so the stronger one folding is correct and the
weaker one calling is the (tiny) mistake. Near the calling line the rank order
is not even the equity order, because a shove range full of AJ, KJ and QJ
dominates a jack more than a ten (T8s has more equity than J8s there). So the
check now prices each node:

* **The price** comes from the trainer's own chip rules. The history is replayed
  through `games.nolimit.NoLimitHoldem` (the native solver sizes raises by the
  same formula), which gives what the player to act must call and the final pot
  if it does, with an all-in larger than our stack returning its excess.
* **The equity** of each class against the opponent's range at the node, where
  the range is each class's combinations times the opponent's own reach to the
  node, from the 169 by 169 preflop all-in table
  (`results/cfr/chance/nolimit_12bb_preflop_allin.npy`, P(win) - P(lose) over
  sampled runouts, so each cell carries a few hundredths of noise). Preflop
  all-in equity does not depend on the stacks, so the 12bb table holds at every
  depth. Card removal between the two hands is ignored.
* **The cost** of a class is what its fold share loses against the best reply
  to that range: the call's value times the fold share when calling wins, the
  call's loss times the call share when it loses. It is weighted by how often
  the class is dealt and reaches the node and by the opponent's line
  probability, and summed in big blinds per 100 hands dealt at that depth.

This is the solve measured against the best reply to its own ranges at each
all-in, not exploitability: a range that is itself wrong is not caught here.
The T against 2T verdict passes when the total cost is no larger at 2T. The rank
pairs are still printed, as a list to read, not a verdict.

Limits. Raising one card by a rank is nearly always worth equity against an
all-in range, but not always by much: connectors against one-gappers (87s and
97s) are close, so a rank flag there may be noise around a real tie. Entries
still uniform are priced as they would play (half fold, half call), since the
bot would play them that way. Preflop only, because only there are the classes
hands; a postflop bucket mixes strength with texture.
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.betting import ALL_IN, FOLD, legal_actions, schedule_from_args  # noqa: E402

RANKS = "23456789TJQKA"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EQUITY_TABLE = os.path.join(ROOT, "results", "cfr", "chance", "nolimit_12bb_preflop_allin.npy")
#: The lossless preflop abstraction: one class per starting hand, numbered by a fixed
#: sort (abstraction/buckets.py), so every 169-class solve indexes the all-in table alike.
CLASSES = 169


def parse_label(label: str) -> Tuple[int, int, Optional[bool]]:
    """`AKs` to (12, 11, True), `TT` to (8, 8, None)."""
    high, low = RANKS.index(label[0]), RANKS.index(label[1])
    if high == low:
        return high, low, None
    return high, low, label[2] == "s"


def make_label(high: int, low: int, suited: Optional[bool]) -> str:
    if high == low:
        return RANKS[high] * 2
    return f"{RANKS[high]}{RANKS[low]}{'s' if suited else 'o'}"


def neighbour_pairs(labels: List[str]) -> List[Tuple[int, int]]:
    """
    (stronger, weaker) index pairs one rank apart in one card, same suitedness.

    Raising one card by one rank, the other card and the suits fixed, almost
    always gains all-in equity, which is what lets the check flag the order
    without an equity table (the exceptions are close, see the module notes).
    Pairs compare with pairs only.
    """
    index = {label: i for i, label in enumerate(labels) if label}
    pairs = []
    for label, weak in index.items():
        high, low, suited = parse_label(label)
        if suited is None:
            candidates = [(high + 1, low + 1)]
        else:
            candidates = [(high + 1, low), (high, low + 1)]
        for up_high, up_low in candidates:
            if up_high >= len(RANKS) or up_low >= up_high and suited is not None:
                continue
            strong = index.get(make_label(up_high, up_low, suited))
            if strong is not None:
                pairs.append((strong, weak))
    return pairs


def actions_at(history: str, cap) -> Tuple[int, ...]:
    """The legal preflop actions at a history, as the trainer defined them."""
    raises = sum(1 for ch in history if ch >= "2")
    facing = not history or history[-1] >= "2"     # the small blind opens facing the big blind
    last = int(history[-1]) if history else None
    return legal_actions(raises, facing, cap, last, street=0)


def own_reach(strategy, cap, klass: int, history: str, player: Optional[int] = None) -> float:
    """
    The product of a class's own action probabilities along the history, for the
    player to act there unless `player` (0 the small blind, 1 the big) says otherwise.
    """
    actor = len(history) % 2 if player is None else player
    reach = 1.0
    for i in range(actor, len(history), 2):
        prefix = history[:i]
        probabilities = strategy.get(f"{klass}|{prefix}")
        if probabilities is None:
            return 0.0
        actions = actions_at(prefix, cap)
        action = int(history[i])
        if action not in actions or len(probabilities) != len(actions):
            return 0.0
        reach *= float(probabilities[actions.index(action)])
        if reach == 0.0:
            return 0.0
    return reach


def line_probability(strategy, cap, history: str, weights: np.ndarray) -> float:
    """
    How often the opponent's whole range takes this line: its classes' own reach
    weighted by their combinations (card removal ignored). Under self-play this
    is what trains the node, so a node the opponent almost never reaches has an
    average strategy that nothing has trained, whatever its own reach says.
    """
    opponent = 1 - len(history) % 2
    return float(sum(w * own_reach(strategy, cap, k, history, opponent) for k, w in enumerate(weights) if w > 0))


def combo_weights(labels: List[str]) -> np.ndarray:
    """Each class's share of the 1,326 starting hands: 6 a pair, 4 suited, 12 offsuit."""
    counts = np.array([0 if not l else 6 if len(l) == 2 else 4 if l.endswith("s") else 12 for l in labels], float)
    return counts / counts.sum() if counts.sum() else counts


def all_in_histories(strategy) -> List[str]:
    """Preflop histories whose player to act faces an all-in."""
    seen = set()
    for key in strategy:
        _, _, history = key.partition("|")
        if "/" not in history and history.endswith(str(ALL_IN)):
            seen.add(history)
    return sorted(seen, key=lambda h: (len(h), h))


def check(strategy, cap, labels: List[str], threshold: float = 0.2, min_reach: float = 0.05) -> dict:
    """Every stronger-folds-more pair at every preflop all-in node of one solve."""
    pairs = neighbour_pairs(labels)
    weights = combo_weights(labels)
    flagged, compared, uniform, per_node = [], 0, 0, []
    for history in all_in_histories(strategy):
        actions = actions_at(history, cap)
        if FOLD not in actions:
            continue
        slot = actions.index(FOLD)
        line = line_probability(strategy, cap, history, weights)
        node_flags = 0
        fold: Dict[int, float] = {}
        reach: Dict[int, float] = {}
        for klass in range(len(labels)):
            probabilities = strategy.get(f"{klass}|{history}")
            if probabilities is None or len(probabilities) != len(actions):
                continue
            probabilities = np.asarray(probabilities, dtype=float)
            if np.allclose(probabilities, 1.0 / len(probabilities), atol=1e-9):
                uniform += 1
                continue
            r = own_reach(strategy, cap, klass, history)
            if r < min_reach:
                continue
            fold[klass], reach[klass] = float(probabilities[slot]), r
        for strong, weak in pairs:
            if strong not in fold or weak not in fold:
                continue
            compared += 1
            excess = fold[strong] - fold[weak]
            if excess >= threshold:
                node_flags += 1
                flagged.append({"history": history, "stronger": labels[strong], "weaker": labels[weak],
                                "fold_stronger": fold[strong], "fold_weaker": fold[weak],
                                "excess": excess, "reach_stronger": reach[strong], "line": line,
                                "weighted": excess * line * reach[strong]})
        per_node.append({"history": history, "line": line, "flagged": node_flags})
    flagged.sort(key=lambda row: -row["weighted"])
    return {"nodes": len(per_node), "per_node": per_node, "pairs_compared": compared, "uniform_entries": uniform,
            "flagged": flagged, "excess": float(sum(row["excess"] for row in flagged)),
            "weighted_excess": float(sum(row["weighted"] for row in flagged)),
            "threshold": threshold, "min_reach": min_reach}


def verdict(at_t: dict, at_2t: dict) -> str:
    """
    Leg (d): pass when doubling the iterations did not make the order worse, read
    on the excess weighted by how often the line is played (the opponent's line
    probability times the stronger class's own reach), so that a node the solver
    never trains cannot decide the verdict either way.
    """
    if at_2t["pairs_compared"] == 0:
        return "not measured"
    return "pass" if at_2t["weighted_excess"] <= at_t["weighted_excess"] + 1e-12 else "FAIL"


class _NoCards:
    """The game needs an abstraction only to bucket cards, and pricing deals none."""


def price(history: str, stack: int, big_blind: int, cap) -> Tuple[int, int, int]:
    """
    (call, final pot if called, player to act) in chips at a preflop node facing
    an all-in, by replaying the history through the trainer's own game, so the
    sizes are the tree's: half pot, pot and twice pot from the pot after calling,
    at least a big blind, capped at the stack. A history the game would not allow
    raises ValueError rather than being priced as something else.
    """
    from games.nolimit import NoLimitHoldem
    game = NoLimitHoldem(_NoCards(), starting_stack=stack, small_blind=big_blind // 2,
                         big_blind=big_blind, raise_cap=cap)
    state = game.next_state(game.initial_state(), (((0, 1), (2, 3)), ()))
    for ch in history:
        action = int(ch)
        if action not in game.legal_actions(state):
            raise ValueError(f"'{history}': {action} is not legal after '{state.history}'")
        state = game.next_state(state, action)
    player = game.current_player(state)
    if player not in (0, 1) or "/" in state.history:
        raise ValueError(f"'{history}': not a preflop decision")
    opponent = 1 - player
    to_call = state.committed[opponent] - state.committed[player]
    call = min(to_call, state.stacks[player])
    # An all-in that covers more than we hold returns its excess uncalled.
    excess = max(0, state.contributions[opponent] - (state.contributions[player] + call))
    return call, sum(state.contributions) + call - excess, player


def equity_table(path: str = EQUITY_TABLE) -> np.ndarray:
    """Seat-blind preflop all-in equity, [mine, theirs], from the P(win) - P(lose) table."""
    return (1.0 + np.load(path)) / 2.0


def cost_check(strategy, cap, labels: List[str], equity: np.ndarray, stack: int, big_blind: int,
               top: int = 5) -> dict:
    """What the calls and folds at every preflop all-in node cost, in bb per 100 hands dealt."""
    n = len(labels)
    weights = combo_weights(labels)
    nodes, total = [], 0.0
    for history in all_in_histories(strategy):
        actions = actions_at(history, cap)
        if FOLD not in actions:
            continue
        call, pot, player = price(history, stack, big_blind, cap)
        opponent_reach = np.array([own_reach(strategy, cap, k, history, 1 - player) for k in range(n)])
        rng = weights * opponent_reach
        line = float(rng.sum())
        node = {"history": history, "line": line, "call_bb": call / big_blind, "pot_bb": pot / big_blind,
                "needs": call / pot if pot else 0.0, "cost": 0.0, "top": []}
        if line > 0:
            eq = equity[:n, :n] @ rng / line
            ev = (eq * pot - call) / big_blind                  # bb, calling against folding
            costs = np.zeros(n)
            for k in range(n):
                probabilities = strategy.get(f"{k}|{history}")
                if probabilities is None or len(probabilities) != len(actions):
                    continue
                fold = float(probabilities[actions.index(FOLD)])
                reach = own_reach(strategy, cap, k, history, player)
                loss = ev[k] * fold if ev[k] > 0 else -ev[k] * (1.0 - fold)
                costs[k] = weights[k] * reach * line * loss * 100.0
                if costs[k] > 0:
                    node["top"].append({"class": labels[k], "equity": float(eq[k]), "ev_call_bb": float(ev[k]),
                                        "fold": fold, "own_reach": reach, "cost": float(costs[k])})
            node["cost"] = float(costs.sum())
            node["top"] = sorted(node["top"], key=lambda r: -r["cost"])[:top]
        total += node["cost"]
        nodes.append(node)
    nodes.sort(key=lambda r: -r["cost"])
    return {"cost": total, "nodes": nodes}


def cost_verdict(at_t: dict, at_2t: dict) -> str:
    """Leg (d) on chips: pass when doubling the iterations did not make the calls and folds cost more."""
    if at_t.get("cost_check") is None or at_2t.get("cost_check") is None:
        return "not measured"
    return "pass" if at_2t["cost_check"]["cost"] <= at_t["cost_check"]["cost"] + 1e-12 else "FAIL"


def labels_of(saved: dict) -> List[str]:
    """The 169 class labels in the solve's own preflop class numbering."""
    preflop = saved["abstraction"]._preflop
    labels = [""] * (max(preflop.values()) + 1)
    for (high, low, suited), index in preflop.items():
        labels[index] = f"{high}{high}" if high == low else f"{high}{low}{'s' if suited else 'o'}"
    return labels


def report(path: str, threshold: float, min_reach: float, equity: Optional[np.ndarray] = None) -> dict:
    from cfr.flat import load_strategy
    saved = load_strategy(path)
    cap = schedule_from_args(saved["args"]["raise_cap"])
    labels = labels_of(saved)
    result = check(saved["strategy"], cap, labels, threshold, min_reach)
    result["path"] = path
    result["depth_bb"] = saved["args"]["stack"] / saved["args"]["big_blind"]
    # The table is indexed by the lossless classes; a coarser preflop abstraction has no rows in it.
    result["cost_check"] = None
    if equity is not None and len(labels) == CLASSES and all(labels):
        result["cost_check"] = cost_check(saved["strategy"], cap, labels, equity,
                                          int(saved["args"]["stack"]), int(saved["args"]["big_blind"]))
    return result


def show(result: dict, top: int) -> None:
    print(f"\n=== {os.path.basename(result['path'])}  ({result['depth_bb']:.0f}bb)")
    priced = result.get("cost_check")
    if priced is None:
        print("  cost: not priced (the all-in table needs the 169 lossless preflop classes)")
    else:
        print(f"  cost of the calls and folds against the best reply to each node's own range: "
              f"{priced['cost']:.3f} bb/100 hands at this depth")
        for node in priced["nodes"][:3]:
            print(f"    '{node['history']}'  {node['cost']:.4f} bb/100  (line {node['line']:.3f}, "
                  f"call {node['call_bb']:g}bb into {node['pot_bb']:g}bb, needs {100 * node['needs']:.1f}%)")
            for row in node["top"][:3]:
                print(f"        {row['class']:4s} equity {100 * row['equity']:.1f}%  EV(call) {row['ev_call_bb']:+.2f}bb  "
                      f"fold {100 * row['fold']:.0f}%  cost {row['cost']:.4f}")
    print(f"  rank order (a list to read, not the verdict): "
          f"{result['nodes']} preflop all-in nodes, {result['pairs_compared']:,} neighbour pairs compared, "
          f"{result['uniform_entries']:,} entries still uniform")
    print(f"  stronger hand folding ≥{result['threshold']:.2f} more than its neighbour: "
          f"{len(result['flagged'])} pairs, summed excess {result['excess']:.2f}, "
          f"weighted by how often the line is played {result['weighted_excess']:.4f}")
    nodes = sorted(result["per_node"], key=lambda n: -n["line"])
    print("  nodes (opponent's line probability, pairs flagged): " +
          ", ".join(f"'{n['history']}' {n['line']:.3f} {n['flagged']}" for n in nodes))
    for row in result["flagged"][:top]:
        print(f"    '{row['history']}'  {row['stronger']} folds {100 * row['fold_stronger']:.0f}% against "
              f"{row['weaker']} {100 * row['fold_weaker']:.0f}%  (own reach {row['reach_stronger']:.2f}, "
              f"line {row['line']:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+",
                        help="one solve, a ladder directory, or two solves of one tree at T and 2T")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="flag a pair when the stronger hand folds at least this much more")
    parser.add_argument("--min-reach", type=float, default=0.05,
                        help="leave out a class whose own reach to the node is below this")
    parser.add_argument("--top", type=int, default=8, help="flagged pairs to print per solve")
    parser.add_argument("--json", help="write every result here")
    parser.add_argument("--equity-table", default=EQUITY_TABLE, help="169x169 preflop all-in P(win) - P(lose)")
    args = parser.parse_args()
    equity = equity_table(args.equity_table)

    if len(args.targets) == 1 and os.path.isdir(args.targets[0]):
        paths = sorted(p for p in glob.glob(os.path.join(args.targets[0], "*.pkl")) if ".flat." not in p)
    else:
        paths = args.targets
    if len(paths) > 2 and not os.path.isdir(args.targets[0]):
        parser.error("give one solve, a ladder directory, or exactly two solves (T, then 2T)")

    results = []
    for path in paths:
        try:
            results.append(report(path, args.threshold, args.min_reach, equity))
        except (KeyError, AttributeError) as error:         # a rung saved without its abstraction or args
            print(f"{os.path.basename(path)}: skipped ({error!r})")
            continue
        show(results[-1], args.top)

    out = {"results": results}
    if len(args.targets) == 2 and len(results) == 2:
        out["verdict"] = cost_verdict(results[0], results[1])
        out["rank_verdict"] = verdict(results[0], results[1])
        costs = [r["cost_check"]["cost"] if r.get("cost_check") else float("nan") for r in results]
        print(f"\n(d) {out['verdict']}: the calls and folds cost {costs[0]:.3f} bb/100 at T and {costs[1]:.3f} at 2T. "
              f"Rank order, for reading: weighted excess {results[0]['weighted_excess']:.4f} and "
              f"{results[1]['weighted_excess']:.4f} ({out['rank_verdict']})")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
