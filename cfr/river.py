"""
River endgame solving on the exact hand.

The blueprint plays the river through six equity buckets and a betting tree
fitted before the cards were known. Here, once the river is dealt, the river is
solved afresh: over the 1,081 exact hole-card pairs the board allows, with each
player's range taken from the blueprint's own reach along the public history
(unsafe re-solving, the choice every practical bot since 2017 has made), with
the pot, the outstanding bet and the stacks as they actually are, and with the
opponent's actual bet size at the root rather than a translated one.

Ganzfried and Sandholm (AAMAS 2015) measured +29 to +87 mbb/hand on river hands
from a base that already had thousands of buckets; ours has six, so the
headroom is larger. It also removes the river's off-tree misses, because the
tree is built at decision time from the live chips.

The solver is vector-form CFR+ with linear averaging: regrets and strategy sums
are arrays over hands, terminal utilities come from one sort of the hands by
rank with per-card prefix sums for the blocker correction, and all showdown
terminals are evaluated in one batch. About 160 nodes and 80 terminals on a
two-raise river; a few milliseconds an iteration in numpy.

Behind a flag everywhere it is reachable (`ArenaPlayer(river=...)`,
`chipzen_run.py --river-solve`). Nothing measured in this project has gone
through it yet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, raise_sizes_at
from abstraction.equity import FULL_DECK
from engine.cards import Card
from engine.hand_eval_fast import score_hand_7_fast
from evaluation.benchmark import _solver_actions
from slumbot.bridge import RAISE_FRACTIONS

#: Iterations and the wall-clock cap. CFR+ on a river of this size is close to
#: converged by a few hundred; the cap is what the arena's clock allows.
ITERATIONS = 400
TIME_BUDGET_S = 8.0
#: What the native core runs by default. Measured 22 September on a two-raise
#: river against a 6,000-iteration solve: at 400 iterations 311 of 1,081 hands
#: had a root probability more than 0.1 off, at 2,000 (1.4 s native) 65, at
#: 3,000 (2.1 s) 24. The Python's 400 was the clock's limit, not convergence.
NATIVE_ITERATIONS = 2000


# ---------------------------------------------------------------------------
# The hands the board allows
# ---------------------------------------------------------------------------

#: Hand sets by board, and per-board bucket arrays by (abstraction, board):
#: the range walk bucketed 1,081 hands per street on every river decision,
#: 0.29 s of a 1.5 s solve, and a board is seen again on the same river.
_HAND_SETS: Dict[tuple, "HandSet"] = {}
_BUCKETS: Dict[tuple, np.ndarray] = {}


@dataclass
class HandSet:
    """The exact hole-card pairs a five-card board leaves, ranked."""
    pairs: np.ndarray              # (H, 2) deck indices, a < b
    ranks: np.ndarray              # (H,) packed 7-card score, higher is better
    index_of: Dict[Tuple[int, int], int]
    order: np.ndarray              # hands sorted by rank
    sorted_ranks: np.ndarray
    lo: np.ndarray                 # per hand: how many sorted hands rank strictly below
    hi: np.ndarray                 # per hand: how many rank at or below (so H - hi are above)
    members: List[np.ndarray]      # per card: hand indices containing it, sorted by rank
    member_lo: List[np.ndarray]    # per card: strictly-lower position within `members`
    member_hi: List[np.ndarray]

    @classmethod
    def build(cls, board: Sequence[Card]) -> "HandSet":
        """Memoised on the board: 0.39 s cold, 4 ms warm, and a board repeats within a hand."""
        key = tuple(sorted(c.index for c in board))
        found = _HAND_SETS.get(key)
        if found is not None:
            return found
        built = cls._build(board)
        if len(_HAND_SETS) >= 64:
            _HAND_SETS.clear()
        _HAND_SETS[key] = built
        return built

    @classmethod
    def _build(cls, board: Sequence[Card]) -> "HandSet":
        used = {c.index for c in board}
        deck = [i for i in range(52) if i not in used]
        board_ranks = np.array([c.index % 13 for c in board], dtype=np.int64)
        board_suits = np.array([c.index // 13 for c in board], dtype=np.int64)
        pairs, ranks = [], []
        for x in range(len(deck)):
            for y in range(x + 1, len(deck)):
                a, b = deck[x], deck[y]
                r = np.concatenate((board_ranks, [a % 13, b % 13])).astype(np.int64)
                s = np.concatenate((board_suits, [a // 13, b // 13])).astype(np.int64)
                pairs.append((a, b))
                ranks.append(int(score_hand_7_fast(r, s)))
        pairs_arr = np.array(pairs, dtype=np.int64)
        ranks_arr = np.array(ranks, dtype=np.int64)
        order = np.argsort(ranks_arr, kind="stable")
        sorted_ranks = ranks_arr[order]
        lo = np.searchsorted(sorted_ranks, ranks_arr, side="left")
        hi = np.searchsorted(sorted_ranks, ranks_arr, side="right")
        members, member_lo, member_hi = [], [], []
        for card in range(52):
            idx = np.flatnonzero((pairs_arr[:, 0] == card) | (pairs_arr[:, 1] == card))
            idx = idx[np.argsort(ranks_arr[idx], kind="stable")]
            r = ranks_arr[idx]
            members.append(idx)
            member_lo.append(np.searchsorted(r, r, side="left"))
            member_hi.append(np.searchsorted(r, r, side="right"))
        return cls(pairs_arr, ranks_arr, {p: i for i, p in enumerate(pairs)}, order,
                   sorted_ranks, lo, hi, members, member_lo, member_hi)

    @property
    def size(self) -> int:
        return int(self.pairs.shape[0])

    def compatible_mass(self, reach: np.ndarray) -> np.ndarray:
        """
        For each hand, the opponent's reach over hands sharing no card with it.

        (T, H) in, (T, H) out. Total, less the two cards' shares, plus the hand
        itself which both shares removed.
        """
        total = reach.sum(axis=1, keepdims=True)
        per_card = np.zeros((reach.shape[0], 52))
        for card in range(52):
            if len(self.members[card]):
                per_card[:, card] = reach[:, self.members[card]].sum(axis=1)
        return total - per_card[:, self.pairs[:, 0]] - per_card[:, self.pairs[:, 1]] + reach

    def showdown(self, reach: np.ndarray) -> np.ndarray:
        """
        For each hand, opponent reach mass it beats minus mass that beats it,
        among compatible hands. (T, H) in, (T, H) out.
        """
        T = reach.shape[0]
        zero = np.zeros((T, 1))
        cum = np.concatenate((zero, np.cumsum(reach[:, self.order], axis=1)), axis=1)
        total = cum[:, -1:]
        lower = cum[:, self.lo]
        higher = total - cum[:, self.hi]
        for card in range(52):
            idx = self.members[card]
            if not len(idx):
                continue
            c = np.concatenate((zero, np.cumsum(reach[:, idx], axis=1)), axis=1)
            lower[:, idx] -= c[:, self.member_lo[card]]
            higher[:, idx] -= c[:, -1:] - c[:, self.member_hi[card]]
        return lower - higher


# ---------------------------------------------------------------------------
# The river betting tree, in live chips
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str                       # "decision", "fold", "showdown"
    player: int = -1                # who acts, for a decision
    actions: List[int] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)
    contrib: Tuple[int, int] = (0, 0)    # chips each player has put in, in total
    folder: int = -1
    index: int = -1                 # decision index, for the regret tables


def build_tree(pot_before: int, to_call: int, stacks: Tuple[int, int],
               raises_so_far: int, schedule=2, first_player: int = 0,
               sizes: Sequence[float] = RAISE_FRACTIONS) -> Tuple[Node, List[Node]]:
    """
    The river from here. Player `first_player` acts, facing `to_call` (0 for a
    check). `pot_before` excludes the outstanding bet. Bets are sized off the
    pot after the call, as the abstraction defines them, and capped by the
    shorter stack; a raise that would take a player past their stack is a
    shove and is not duplicated.
    """
    decisions: List[Node] = []
    half = pot_before // 2
    # Contributions so far: the pot before the bet split evenly, plus the bet
    # by the player who made it (the one not to act).
    contrib0 = [half, pot_before - half]
    bettor = 1 - first_player
    contrib0[bettor] += to_call
    stack = list(stacks)

    def make(player: int, contrib: List[int], remaining: List[int], raises: int,
             facing: int, checked: bool) -> Node:
        other = 1 - player
        node = Node("decision", player=player, contrib=(contrib[0], contrib[1]), index=len(decisions))
        decisions.append(node)
        # Fold, or check/call.
        if facing > 0:
            node.actions.append(FOLD)
            node.children.append(Node("fold", contrib=(contrib[0], contrib[1]), folder=player))
        call = min(facing, remaining[player])
        after_call = list(contrib)
        after_call[player] += call
        rem_call = list(remaining)
        rem_call[player] -= call
        node.actions.append(CHECK_CALL)
        if facing > 0 or checked:
            node.children.append(Node("showdown", contrib=(after_call[0], after_call[1])))
        else:
            node.children.append(make(other, after_call, rem_call, raises, 0, True))
        # Raises, if either stack allows and the schedule has depth left.
        if rem_call[player] <= 0 or remaining[other] <= 0:
            return node
        allowed = raise_sizes_at(schedule, raises, 3)       # the river, always
        pot_after_call = after_call[0] + after_call[1]
        seen = set()
        for code in allowed:
            if code == ALL_IN:
                amount = rem_call[player]
            else:
                amount = int(round(pot_after_call * sizes[code - 2]))
                amount = max(amount, 1)
            # Capped by our stack, and by theirs: chips they cannot call are
            # returned, so a raise past either stack is the same shove.
            amount = min(amount, rem_call[player], remaining[other])
            if amount in seen:
                continue
            seen.add(amount)
            c = list(after_call)
            c[player] += amount
            r = list(rem_call)
            r[player] -= amount
            node.actions.append(code)
            node.children.append(make(other, c, r, raises + 1, amount, False))
        return node

    root = make(first_player, contrib0, stack, raises_so_far, to_call, False)
    return root, decisions


# ---------------------------------------------------------------------------
# CFR+ over the tree
# ---------------------------------------------------------------------------

class RiverSolver:
    """Solve one river from live chips and two ranges."""

    def __init__(self, hands: HandSet, root: Node, decisions: List[Node],
                 ranges: Tuple[np.ndarray, np.ndarray]):
        self.hands, self.root, self.decisions = hands, root, decisions
        self.ranges = ranges
        H = hands.size
        self.regrets = [np.zeros((len(n.actions), H)) for n in decisions]
        self.strategy_sum = [np.zeros((len(n.actions), H)) for n in decisions]
        self.iterations = 0
        # Terminal bookkeeping, filled per pass.
        self._terminals: List[Node] = []
        self._collect(root)

    def _collect(self, node: Node) -> None:
        if node.kind != "decision":
            self._terminals.append(node)
        for child in node.children:
            self._collect(child)

    def _strategy(self, node: Node) -> np.ndarray:
        regret = self.regrets[node.index]
        positive = np.maximum(regret, 0.0)
        total = positive.sum(axis=0, keepdims=True)
        uniform = np.full_like(positive, 1.0 / positive.shape[0])
        return np.where(total > 0, positive / np.where(total > 0, total, 1.0), uniform)

    def iterate(self) -> None:
        self.iterations += 1
        for traverser in (0, 1):
            self._pass(traverser)

    def _pass(self, traverser: int) -> None:
        H = self.hands.size
        opp = 1 - traverser
        # Forward: reach of each player at every node.
        reach: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        strategies: Dict[int, np.ndarray] = {}
        showdown_nodes, showdown_reach = [], []
        fold_nodes, fold_reach = [], []

        def forward(node: Node, r0: np.ndarray, r1: np.ndarray) -> None:
            if node.kind == "showdown":
                showdown_nodes.append(node)
                showdown_reach.append(r1 if traverser == 0 else r0)
                return
            if node.kind == "fold":
                fold_nodes.append(node)
                fold_reach.append(r1 if traverser == 0 else r0)
                return
            sigma = self._strategy(node)
            strategies[node.index] = sigma
            reach[node.index] = (r0, r1)
            for a, child in enumerate(node.children):
                if node.player == 0:
                    forward(child, r0 * sigma[a], r1)
                else:
                    forward(child, r0, r1 * sigma[a])

        forward(self.root, self.ranges[0], self.ranges[1])

        # Terminal utilities for the traverser, batched.
        values: Dict[int, np.ndarray] = {}
        if showdown_nodes:
            R = np.stack(showdown_reach)
            diff = self.hands.showdown(R)
            for k, node in enumerate(showdown_nodes):
                values[id(node)] = node.contrib[opp] * diff[k]
        if fold_nodes:
            R = np.stack(fold_reach)
            mass = self.hands.compatible_mass(R)
            for k, node in enumerate(fold_nodes):
                if node.folder == traverser:
                    values[id(node)] = -node.contrib[traverser] * mass[k]
                else:
                    values[id(node)] = node.contrib[opp] * mass[k]

        weight = float(self.iterations)

        def backward(node: Node) -> np.ndarray:
            if node.kind != "decision":
                return values[id(node)]
            child_values = np.stack([backward(c) for c in node.children])
            sigma = strategies[node.index]
            if node.player == traverser:
                value = (sigma * child_values).sum(axis=0)
                regret = self.regrets[node.index]
                regret += child_values - value
                np.maximum(regret, 0.0, out=regret)          # regret matching+
                own_reach = reach[node.index][traverser]
                self.strategy_sum[node.index] += weight * own_reach * sigma
                return value
            return child_values.sum(axis=0)

        backward(self.root)

    def solve(self, iterations: int = ITERATIONS, budget_s: float = TIME_BUDGET_S) -> int:
        started = time.perf_counter()
        for _ in range(iterations):
            self.iterate()
            if time.perf_counter() - started > budget_s:
                break
        return self.iterations

    def root_strategy(self, hand_index: int) -> Dict[int, float]:
        """The average strategy at the root for one of our hands, by action code."""
        total = self.strategy_sum[self.root.index][:, hand_index]
        if total.sum() <= 0:
            probabilities = np.full(len(total), 1.0 / len(total))
        else:
            probabilities = total / total.sum()
        return {code: float(p) for code, p in zip(self.root.actions, probabilities)}


class NativeRiverSolver:
    """
    The same solve in C++ (native/src/river.hpp), fifty-odd times faster.

    The Python solver above stays as the reference it is pinned against: the
    tree is built here, flattened in creation order (every child after its
    parent, which is what lets the native passes be loops), and handed over
    with the hand set's pairs and ranks and the two ranges.
    """

    def __init__(self, hands: HandSet, root: Node, decisions: List[Node],
                 ranges: Tuple[np.ndarray, np.ndarray]):
        import pokerbot_native as native
        nodes: List[Node] = []

        def collect(node: Node) -> None:
            nodes.append(node)
            for child in node.children:
                collect(child)

        collect(root)
        index = {id(n): i for i, n in enumerate(nodes)}
        kind = [0 if n.kind == "decision" else (1 if n.kind == "fold" else 2) for n in nodes]
        self.root_actions = list(root.actions)
        self._native = native.RiverSolver(
            [int(p) for p in hands.pairs[:, 0]], [int(p) for p in hands.pairs[:, 1]],
            [int(r) for r in hands.ranks], kind, [n.player for n in nodes],
            [n.contrib[0] for n in nodes], [n.contrib[1] for n in nodes], [n.folder for n in nodes],
            [[index[id(c)] for c in n.children] for n in nodes], [list(n.actions) for n in nodes],
            [float(x) for x in ranges[0]], [float(x) for x in ranges[1]])
        self.iterations = 0

    def solve(self, iterations: int = ITERATIONS, budget_s: float = TIME_BUDGET_S) -> int:
        self.iterations = self._native.solve(iterations, budget_s)
        return self.iterations

    def root_strategy(self, hand_index: int) -> Dict[int, float]:
        probabilities = self._native.root_strategy(hand_index)
        return {code: float(p) for code, p in zip(self.root_actions, probabilities)}


def make_solver(hands: HandSet, root: Node, decisions: List[Node],
                ranges: Tuple[np.ndarray, np.ndarray], native: Optional[bool] = None):
    """The native solver when it is importable (or asked for), else the Python reference."""
    if native is False:
        return RiverSolver(hands, root, decisions, ranges)
    try:
        return NativeRiverSolver(hands, root, decisions, ranges)
    except ImportError:
        if native:
            raise
        return RiverSolver(hands, root, decisions, ranges)


# ---------------------------------------------------------------------------
# Ranges from the blueprint
# ---------------------------------------------------------------------------

def _hole(pair: Tuple[int, int]) -> List[Card]:
    return [FULL_DECK[pair[0]], FULL_DECK[pair[1]]]


def blueprint_ranges(hands: HandSet, strategy: dict, abstraction, schedule,
                     history: str, board: Sequence[Card], we_are_small_blind: bool
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each player's reach over the exact hands, walked along the public history
    under the blueprint. The opponent is assumed to play the blueprint too,
    which is what makes this unsafe re-solving; where the blueprint has no
    entry, the action is given the uniform share it would have played.
    """
    H = hands.size
    reach = [np.ones(H), np.ones(H)]
    streets = history.split("/")
    board_sizes = (0, 3, 4, 5)
    for street_index, street in enumerate(streets):
        if street_index >= 4:
            break
        prefix_board = list(board[:board_sizes[street_index]])
        # Buckets for every hand on this board prefix, seeded as the agent
        # seeds; memoised per (abstraction, board prefix, hand set).
        memo_key = (id(abstraction), tuple(c.index for c in prefix_board), id(hands))
        buckets = _BUCKETS.get(memo_key)
        if buckets is None:
            buckets = np.empty(H, dtype=np.int64)
            for i in range(H):
                hole = _hole(tuple(hands.pairs[i]))
                key = (tuple(c.index for c in hole), tuple(c.index for c in prefix_board))
                buckets[i] = abstraction.bucket(hole, prefix_board, np.random.default_rng(hash(key) % (2 ** 32)))
            if len(_BUCKETS) >= 256:
                _BUCKETS.clear()
            _BUCKETS[memo_key] = buckets
        # Who acts first on this street: the small blind preflop, the big blind after.
        actor_is_us = we_are_small_blind if street_index == 0 else not we_are_small_blind
        prefix = "/".join(streets[:street_index]) + ("/" if street_index else "")
        # Preflop the first to act faces the big blind, so the root's legal
        # list has a fold in it; after the flop a street opens unbet.
        facing = 1 if street_index == 0 else 0
        for code_char in street:
            code = int(code_char)
            actions = _solver_actions(prefix, facing, schedule)
            player = 0 if actor_is_us else 1
            if code in actions:
                slot = actions.index(code)
                for b in np.unique(buckets):
                    probabilities = strategy.get(f"{b}|{prefix}")
                    mask = buckets == b
                    if probabilities is not None and facing and probabilities.size == 2:
                        # Stack-capped in the tree: the entry is fold/call, and
                        # a raise there has no mass (see `cfr_agent`).
                        reach[player][mask] *= float(probabilities[code]) if code <= CHECK_CALL else 0.0
                    elif probabilities is None or probabilities.size != len(actions):
                        reach[player][mask] *= 1.0 / len(actions)
                    else:
                        reach[player][mask] *= float(probabilities[slot])
            prefix += code_char
            facing = 1 if code >= 2 else 0
            actor_is_us = not actor_is_us
    return reach[0], reach[1]


# ---------------------------------------------------------------------------
# The arena entry point
# ---------------------------------------------------------------------------

@dataclass
class RiverDecision:
    choice: int
    distribution: Dict[int, float]
    iterations: int
    ms: float
    hands: int
    #: Where the opponent's range came from: "blueprint", "oracle", or
    #: "blueprint (oracle empty)" when the oracle gave every hand zero.
    range_source: str = "blueprint"


def decide_river(state: dict, hole: Sequence[Card], board: Sequence[Card], history: str,
                 strategy: dict, abstraction, schedule, we_are_small_blind: bool,
                 rng: np.random.Generator, legal: Optional[np.ndarray] = None,
                 iterations: int = ITERATIONS, budget_s: float = TIME_BUDGET_S,
                 purify: bool = False,
                 opponent_range: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 blend: float = 0.0) -> RiverDecision:
    """
    Solve this river and choose. `state` is the arena's turn state: pot (with
    the outstanding bet in it), to_call, your_stack, opponent_stacks.

    `opponent_range`, given the hand set's hole pairs, replaces the
    blueprint's reach for the opponent (`cfr/river_oracle.py`). A range that
    is zero everywhere means the model could not have played this hand the
    way it was played, and the blueprint's is kept rather than solving
    against nothing.

    `blend` mixes the opponent's blueprint range with every hand the board
    allows, each normalised to one: 0 is the blueprint's reach, 1 is "any two
    cards". The blueprint's range assumes the opponent plays as we do, which is
    wrong for a bot that shoves a pair of fives (an arena opponent: a
    ten-high flush folded to three queens); "any two" is wrong for one whose big
    bets are all value. The blend is the knob between them.
    """
    started = time.perf_counter()
    hands = HandSet.build(board)
    to_call = int(state.get("to_call") or 0)
    pot = int(state.get("pot") or 0)
    ours = int(state.get("your_stack") or 0)
    theirs = int((state.get("opponent_stacks") or [0])[0])
    river = history.split("/")[-1] if history.count("/") >= 3 else ""
    raises_so_far = sum(1 for ch in river if ch in "2345")
    root, decisions = build_tree(pot - to_call, to_call, (ours, theirs), raises_so_far,
                                 schedule if isinstance(schedule, int) else schedule, 0)
    ranges = blueprint_ranges(hands, strategy, abstraction, schedule, history, board, we_are_small_blind)
    range_source = "blueprint"
    if opponent_range is not None:
        theirs_reach = np.asarray(opponent_range(hands.pairs), dtype=float)
        if theirs_reach.sum() > 0:
            ranges, range_source = (ranges[0], theirs_reach), "oracle"
        else:
            range_source = "blueprint (oracle empty)"
    if blend > 0 and range_source == "blueprint":
        theirs = np.asarray(ranges[1], dtype=float)
        total = theirs.sum()
        uniform = np.full_like(theirs, 1.0 / len(theirs))
        mixed = (1.0 - blend) * (theirs / total if total > 0 else uniform) + blend * uniform
        ranges, range_source = (ranges[0], mixed), f"blend {blend:g}"
    solver = make_solver(hands, root, decisions, ranges)
    if isinstance(solver, NativeRiverSolver) and iterations == ITERATIONS:
        iterations = NATIVE_ITERATIONS
    remaining = budget_s - (time.perf_counter() - started)
    done = solver.solve(iterations, max(0.5, remaining))
    ours_index = hands.index_of[tuple(sorted(c.index for c in hole))]
    distribution = solver.root_strategy(ours_index)
    if legal is not None:
        distribution = {a: p for a, p in distribution.items() if legal[a]}
        total = sum(distribution.values())
        distribution = {a: p / total for a, p in distribution.items()} if total > 0 else \
            {a: 1.0 / len(distribution) for a in distribution}
    codes = list(distribution)
    probabilities = np.array([distribution[c] for c in codes])
    if purify:
        choice = codes[int(probabilities.argmax())]
    else:
        choice = codes[int(rng.choice(len(codes), p=probabilities / probabilities.sum()))]
    return RiverDecision(int(choice), distribution, done,
                         (time.perf_counter() - started) * 1000.0, hands.size, range_source)
