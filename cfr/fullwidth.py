"""
Full-width CFR+ over the explicit abstract game.

The sampled solver deals real cards every iteration and touches only the nodes
that deal reaches; its error shrinks with the square root of the visits, and
on 19 September a cap-2 rung at 100M iterations, warm-started and pruned, was
still 8 BB/100 behind the one-raise rung it should dominate. This solver is
the research answer to that: chance is written down as bucket-transition
tables (`scripts/cfr/chance_tables.py`), the game becomes a finite tree, and
every iteration computes exact counterfactual values over every branch, so
regrets move in the right direction every time and the known convergence is
O(1/T) in practice rather than O(1/sqrt(visits)).

The tree is walked in "vector" form: a public node is (betting history,
board-texture path) and each player's private state is a vector over their
current bucket, 169 preflop and six strength classes postflop. Values are
matrices over the two players' private states, chance transitions are matrix
products, and one node's regret update covers every information set that
shares its history at once. Chance is factorised, which is the one
approximation on top of the abstraction itself: preflop buckets are drawn
independently, the board texture independently of the hole cards, and each
player's strength class given their own bucket and the texture; card removal
between the two hands is lost. The showdown table keeps the joint outcome.

Information sets are keyed exactly as the pickles are, `bucket|history`, with
imperfect recall (the current bucket only), so the export plays in the arena
and the gates unchanged.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from abstraction.betting import FOLD
from games.nolimit import NoLimitHoldem, NoLimitState, STREET_BOARD_SIZE

TEXTURES = 6


class ChanceTables:
    """The factorised chance player, from the counts `chance_tables.py` wrote."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        pre, flop, turn, river, showdown = (d["preflop"].astype(float), d["flop"].astype(float),
                                            d["turn"].astype(float), d["river"].astype(float),
                                            d["showdown"].astype(float))
        self.n_pre = pre.shape[0]
        self.S = flop.shape[3]
        S, T = self.S, TEXTURES
        # Preflop bucket marginal, both seats pooled.
        m = pre.sum(axis=1) + pre.sum(axis=0)
        self.p_pre = m / m.sum()
        # Flop: texture marginal, and P(strength | bucket, texture) pooled over seats.
        by_t = flop.sum(axis=(0, 1, 3, 4))
        self.p_tex_flop = by_t / by_t.sum()
        c = flop.sum(axis=(1, 4)) + flop.sum(axis=(0, 3))       # [b, t, s]
        self.m_flop = self._rows(c.reshape(self.n_pre, T, S))    # P(s | b, t), [T][b, s]
        # Turn and river: state = (t, s0, s1) flattened as (t*S + s0)*S + s1.
        self.p_tex_turn, self.m_turn = self._street(turn)
        self.p_tex_river, self.m_river = self._street(river)
        # Showdown: expected outcome for player 0 at a river state, [t, s0, s1].
        sd = showdown.reshape(T, S, S, 3)
        n = sd.sum(axis=3)
        self.showdown = np.where(n > 0, (sd[..., 0] - sd[..., 2]) / np.maximum(n, 1), 0.0)

    def _rows(self, c):
        """Row-normalise P(s | b, t) into a list over t of [b, s] matrices."""
        out = []
        for t in range(TEXTURES):
            rows = c[:, t, :]
            total = rows.sum(axis=1, keepdims=True)
            out.append(np.where(total > 0, rows / np.maximum(total, 1e-12), 1.0 / self.S))
        return out

    def _street(self, counts):
        """P(t' | t) and, for one player, P(s' | t, s, t') from a state-to-state table."""
        S, T = self.S, TEXTURES
        c = counts.reshape(T, S, S, T, S, S)                      # [t, s0, s1, t', s0', s1']
        tex = c.sum(axis=(1, 2, 4, 5))                            # [t, t']
        p_tex = np.where(tex.sum(axis=1, keepdims=True) > 0,
                         tex / np.maximum(tex.sum(axis=1, keepdims=True), 1e-12), 0.0)
        # Player 0's transition pooled with player 1's (the seats are symmetric).
        m = c.sum(axis=(2, 5)) + c.sum(axis=(1, 4)).transpose(0, 1, 2, 3)   # [t, s, t', s']
        mats = [[None] * T for _ in range(T)]
        for t in range(T):
            for t2 in range(T):
                rows = m[t, :, t2, :]
                total = rows.sum(axis=1, keepdims=True)
                mats[t][t2] = np.where(total > 0, rows / np.maximum(total, 1e-12), 1.0 / S)
        return p_tex, mats


class FullWidthSolver:
    """CFR+ (regret floor, linear averaging) over the abstract no-limit game."""

    def __init__(self, tables: ChanceTables, abstraction, stack: int, small_blind: int,
                 big_blind: int, raise_cap, preflop_allin: np.ndarray = None):
        self.t = tables
        self.game = NoLimitHoldem(abstraction, stack, small_blind, big_blind, raise_cap)
        #: [169, 169] P(win) - P(lose) between preflop classes, seat 0's view
        #: (`scripts/cfr/preflop_allin.py`). A preflop all-in is valued from it
        #: directly instead of through the factorised board: the factorisation
        #: cannot see that AA against 72 is 82/18, only that a strong class
        #: meets a weak one on an average board, and at short stacks the
        #: preflop all-in is most of the game. None: the factorised runout.
        self.preflop_allin = preflop_allin
        self.regret: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}
        self.actions_at: Dict[str, Tuple[int, ...]] = {}
        self.iteration = 0
        self.nodes_touched = 0

    # ---- information sets --------------------------------------------------

    def _keys(self, state: NoLimitState, tex: int, n: int) -> List[str]:
        if state.street == 0:
            return [f"{b}|{state.history}" for b in range(n)]
        return [f"{s + self.t.S * tex}|{state.history}" for s in range(n)]

    def _policy(self, keys: List[str], actions: Tuple[int, ...]) -> np.ndarray:
        """Regret matching at every private state of the node, as an [n, |A|] matrix."""
        a = len(actions)
        out = np.empty((len(keys), a))
        for i, k in enumerate(keys):
            r = self.regret.get(k)
            if r is None:
                r = np.zeros(a)
                self.regret[k] = r
                self.strategy_sum[k] = np.zeros(a)
                self.actions_at[k] = actions
            pos = np.maximum(r, 0.0)
            total = pos.sum()
            out[i] = pos / total if total > 0 else 1.0 / a
        return out

    # ---- the walk ---------------------------------------------------------------

    def _terminal_value(self, state: NoLimitState, tex: int, n0: int, n1: int) -> np.ndarray:
        if str(FOLD) in state.history:
            return np.full((n0, n1), self.game.utility(state, 0))
        at_risk = float(min(state.contributions))
        return at_risk * self.t.showdown[tex]

    def _walk(self, state: NoLimitState, tex: int, r0: np.ndarray, r1: np.ndarray,
              pc: float) -> np.ndarray:
        """
        Values to player 0 over (private0, private1), and the regret updates on the way.

        `r0`, `r1` are each player's reach over their private states (own actions
        and own private chance); `pc` is the public chance weight of this texture path.
        """
        self.nodes_touched += 1
        game = self.game
        n0, n1 = r0.size, r1.size
        if game.is_terminal(state):
            return self._terminal_value(state, tex, n0, n1)

        if game.current_player(state) < 0:
            if state.street == 0 and self.preflop_allin is not None and game._all_in(state):
                return float(min(state.contributions)) * self.preflop_allin
            # A street ends: the texture may change and each player's strength
            # class moves by its own transition. Preflop to flop maps 169
            # buckets onto six classes.
            street = state.street + 1
            child = state._replace(board=(0,) * STREET_BOARD_SIZE[street],
                                   history=state.history + "/", committed=(0, 0), street=street)
            if state.street == 0:
                p_tex, mats = self.t.p_tex_flop, self.t.m_flop
                value = np.zeros((n0, n1))
                for t2 in range(TEXTURES):
                    if p_tex[t2] <= 0:
                        continue
                    m = mats[t2]                                   # [169, S]
                    v = self._walk(child, t2, r0 @ m, r1 @ m, pc * p_tex[t2])
                    value += p_tex[t2] * (m @ v @ m.T)
                return value
            p_tex, mats = (self.t.p_tex_turn, self.t.m_turn) if state.street == 1 \
                else (self.t.p_tex_river, self.t.m_river)
            value = np.zeros((n0, n1))
            for t2 in range(TEXTURES):
                p = p_tex[tex, t2]
                if p <= 0:
                    continue
                m = mats[tex][t2]                                  # [S, S]
                v = self._walk(child, t2, r0 @ m, r1 @ m, pc * p)
                value += p * (m @ v @ m.T)
            return value

        player = game.current_player(state)
        actions = tuple(game.legal_actions(state))
        keys = self._keys(state, tex, n0 if player == 0 else n1)
        sigma = self._policy(keys, actions)                        # [n_p, |A|]
        values = []
        for j, a in enumerate(actions):
            child = game.next_state(state, a)
            if player == 0:
                values.append(self._walk(child, tex, r0 * sigma[:, j], r1, pc))
            else:
                values.append(self._walk(child, tex, r0, r1 * sigma[:, j], pc))
        if player == 0:
            node = sum(sigma[:, j][:, None] * values[j] for j in range(len(actions)))
            # Counterfactual value of each action at each private state: the
            # opponent's reach and the public chance weight, not our own.
            u = np.stack([pc * (values[j] @ r1) for j in range(len(actions))], axis=1)   # [n0, |A|]
            reach_own = r0
        else:
            node = sum(sigma[:, j][None, :] * values[j] for j in range(len(actions)))
            u = np.stack([-pc * (r0 @ values[j]) for j in range(len(actions))], axis=1)  # [n1, |A|]
            reach_own = r1
        expected = (sigma * u).sum(axis=1, keepdims=True)
        regret = u - expected
        weight = float(self.iteration)                            # linear averaging (CFR+)
        for i, k in enumerate(keys):
            r = self.regret[k]
            r += regret[i]
            np.maximum(r, 0.0, out=r)                             # the CFR+ floor
            self.strategy_sum[k] += weight * reach_own[i] * sigma[i]
        return node

    def train(self, iterations: int, on_progress=None) -> float:
        started = time.perf_counter()
        root = self.game.initial_state()._replace(hole=((0, 1), (2, 3)), board=())
        for _ in range(iterations):
            self.iteration += 1
            self._walk(root, 0, self.t.p_pre.copy(), self.t.p_pre.copy(), 1.0)
            if on_progress is not None:
                on_progress(self.iteration, iterations, time.perf_counter() - started)
        return time.perf_counter() - started

    def average_strategy(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, s in self.strategy_sum.items():
            total = s.sum()
            out[k] = (s / total) if total > 0 else np.full(s.size, 1.0 / s.size)
        return out
