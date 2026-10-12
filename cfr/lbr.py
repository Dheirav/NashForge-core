"""
Local Best Response — a computable lower bound on exploitability.

Exploitability is the measure that matters: how much a best-responding opponent
wins. On Kuhn and Leduc it is computed exactly by traversing the tree. No-limit
Hold'em cannot be traversed, so the only alternative has been head-to-head chip
counts — and beating a passive baseline says nothing about distance from
equilibrium.

LBR (Lisý & Bowling, 2017) closes that gap. Instead of a true best response it
plays a *greedy* one: track what the opponent's actions reveal about their hand,
then at each decision pick whichever action looks best under a one-step
lookahead. Because that is only a subset of the strategies a real best response
could play, whatever LBR wins is a **lower bound** on exploitability. A large
LBR value proves a strategy is exploitable; a small one proves nothing, which is
the direction of the guarantee and worth stating whenever the number is quoted.

Two design choices specific to this implementation:

* **The opponent's range is a weighted set of candidate hands**, resampled once
  per hand and reweighted by Bayes as the opponent acts. The obvious cheaper
  alternative — carrying a distribution over bucket indices — does not work
  here, because :class:`~abstraction.buckets.CardAbstraction` fits a *separate*
  clustering per street. Bucket 3 on the flop and bucket 3 on the turn are
  unrelated categories, so a belief accumulated over one street's indices cannot
  be carried into the next. Tracking hands and re-bucketing them against
  whatever board is actually out keeps every lookup on the street it belongs to.

* **After the modelled action the hand is rolled out to showdown**, with no
  further betting. This is the standard LBR simplification and it is why the
  result is a bound rather than the true value: LBR never plans a second barrel.

LBR sees only what a real opponent would: its own cards, the board, and the
betting. It never reads ``state.hole`` for the player it is exploiting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, raise_sizes_at
from abstraction.equity import FULL_DECK
from abstraction.translation import translate, translation_distribution
from engine.hand_eval_fast import evaluate_hand_fast
from games.nolimit import RAISE_FRACTION

#: Raise sizes LBR may choose from, as fractions of the pot. Deliberately finer
#: than the abstraction's 0.5 / 1.0 / 2.0, and extending past both ends: a bet
#: the solver never planned against is the entire point, and the sizes between
#: its own are where a strategy trained on three sizes is least prepared.
DEFAULT_BET_SIZES = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.75, 3.5)


@dataclass(frozen=True)
class Move:
    """
    A chosen action: either one the abstraction contains, or a raw bet size.

    Exactly one field is set. ``fraction`` carries a raise the abstraction does
    not have, which must be translated before an opponent can be asked about it.

    ``probe`` marks a raise made where the tree's menu is all-in only. It is
    translated the way the live bridge translates it rather than the way LBR
    translates its other sizes, because that mapping is the thing being probed.
    """
    action: Optional[int]
    fraction: Optional[float]
    probe: bool = False


@dataclass(frozen=True)
class LBRResult:
    """What LBR won, and how sure we are of it."""
    hands: int
    mean: float
    stderr: float
    #: Per-hand results, in hand order, so two runs on the same paired seed can
    #: be differenced hand by hand. Not compared: two results are the same
    #: measurement when their summaries agree.
    values: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def ci95(self) -> tuple:
        return (self.mean - 1.96 * self.stderr, self.mean + 1.96 * self.stderr)

    @property
    def proves_exploitable(self) -> bool:
        """True when the interval clears zero, so the bound is meaningful."""
        return self.mean - 1.96 * self.stderr > 0.0

    def summary(self) -> str:
        low, high = self.ci95
        verdict = ("exploitable by at least this much"
                   if self.proves_exploitable else "no exploitability proven")
        return (f"{self.mean:+.3f} +/- {self.stderr:.3f} chips/hand "
                f"[{low:+.3f}, {high:+.3f}] — {verdict}")


class Range:
    """
    What LBR believes about the opponent's hand: candidate hands and weights.

    The buckets are cached against the board they were computed for, so a new
    street invalidates them and they are recomputed lazily — a hand that ends on
    the flop never pays for turn and river bucketing.
    """

    __slots__ = ("hands", "weights", "buckets", "board")

    def __init__(self, hands: List[Tuple[int, int]], weights: np.ndarray):
        self.hands = hands
        self.weights = weights
        self.buckets: List[int] = []
        self.board: Optional[Tuple[int, ...]] = None

    def live(self) -> np.ndarray:
        """Indices of candidates still carrying weight."""
        return np.flatnonzero(self.weights > 0.0)


class LocalBestResponse:
    """
    A greedy exploiter of a fixed strategy in an abstracted no-limit game.

    Args:
        game: The :class:`~games.nolimit.NoLimitHoldem` being played.
        strategy: The strategy under test, keyed as the game keys its
            information sets.
        rollout_samples: Showdowns sampled per decision to estimate the chance of
            winning.
        candidates: Opponent hands carried as the range. More is a finer read on
            what the betting revealed; the cost is one bucket lookup each per
            street actually reached.
    """

    def __init__(self, game, strategy: Dict[Hashable, np.ndarray],
                 rollout_samples: int = 60, candidates: int = 32,
                 bet_sizes: Sequence[float] = DEFAULT_BET_SIZES,
                 trace: Optional[list] = None,
                 offtree_third_raise: Sequence[float] = (),
                 bridge_translation: bool = False):
        self.game = game
        self.strategy = strategy
        self.rollout_samples = rollout_samples
        self.candidates = candidates
        self.bet_sizes = tuple(bet_sizes)
        #: When a list, one record per sized bet LBR makes: street, size, its
        #: equity against the range, how the strategy answered and the hand's
        #: result. On 26 Sept every single-size menu lost while a menu of
        #: sizes won 31 to 57 BB/100, so the leak is in which size LBR picks
        #: for which hand, and only a record of the picks can show it. The
        #: strength estimate draws from its own generator, so tracing does not
        #: change the hands played.
        self.trace = trace
        #: Pot fractions LBR may raise where the tree offers only all-in. Our
        #: (4, 2, 1) rungs have that at the third raise of a street, and the
        #: live bridge reads any raise there as all-in, so a 0.4 pot raise
        #: buys the fold equity of a shove at a fraction of the risk. LBR's own
        #: sizes cannot reach it: they are translated onto the tree's sized
        #: raises, and at that depth there are none. Empty leaves every hand
        #: exactly as it was.
        self.offtree_third_raise = tuple(float(f) for f in offtree_third_raise)
        #: Read LBR's own off-tree sizes the way ``chipzen.bridge._as_abstract``
        #: reads them, instead of through :func:`translate` onto the legal sized
        #: raises. The two disagree above the largest size: translation clamps a
        #: 3.5 pot bet to 2x pot, while the bridge reads anything at 1.5 times the
        #: largest size or more (and anything that costs the bettor's stack) as
        #: all-in, and re-reads the street if the hand goes on. So without this
        #: LBR scores the 2x-pot node's answer to an overbet the bot in fact
        #: answers at the all-in node. Off by default so every earlier LBR number
        #: reproduces.
        self.bridge_translation = bool(bridge_translation)
        #: The bridge's view of the hand in progress (a ``chipzen.bridge.Hand``)
        #: while probes are on, so its pseudo all-in record and street re-reads
        #: are the bridge's own and not a copy of them.
        self._bridge = None
        #: Counts for the report: how often a probe was made and answered, and
        #: how the strategy was asked after a collapsed street.
        #: ``folded`` and ``called`` count answers to any pseudo all-in, a probe
        #: or an overbet read as one. ``overbet_allins`` counts LBR's own sizes the
        #: bridge read as a pseudo all-in (chips kept), ``capped_allins`` those
        #: that cost the whole stack and so are a real one.
        self.probe_stats = {"probes": 0, "folded": 0, "called": 0,
                            "collapsed_streets": 0, "alt_hits": 0, "alt_misses": 0,
                            "overbet_allins": 0, "capped_allins": 0}

    # ------------------------------------------------------------------
    # The live bridge's reading of a probe, through its own functions.

    def _probe_node(self, state, me: int) -> bool:
        """
        True where LBR may probe: the tree's menu at this depth is all-in only,
        and LBR has chips for a raise that is not one.
        """
        if not self.offtree_third_raise:
            return False
        street_actions = state.history.split("/")[-1]
        depth = sum(1 for a in street_actions if a in "2345")
        if raise_sizes_at(self.game.raise_cap, depth, state.street) != (ALL_IN,):
            return False
        return ALL_IN in self.game.legal_actions(state)

    def _probe_cost(self, state, me: int, fraction: float) -> Optional[int]:
        """Chips a probe costs, or None when it would be a real all-in anyway."""
        cost = self.game._raise_cost(state, me, fraction)
        return None if cost >= state.stacks[me] else cost

    def _bridge_hand(self, me: int):
        from chipzen.bridge import Hand
        from slumbot.bridge import Node
        start = [self.game.starting_stack, self.game.starting_stack]
        return Hand(node=Node(), seat=1 - me, big_blind=self.game.big_blind,
                    start_stacks=start)

    def _bridge_view(self, state, me: int, fraction: float) -> Tuple[float, int, int]:
        """
        The (fraction, level, ceiling) the bridge would compute for this raise.

        The bridge sees chips, not a fraction, so the fraction it is handed is
        recomputed from the real cost the way ``replay`` computes it from the
        arena's levels; the ceiling is the stack the bettor had when the street
        began, which is what the bridge's ``start - prior`` comes to.
        """
        opponent = 1 - me
        cost = self.game._raise_cost(state, me, fraction)
        to_call = max(state.committed[opponent] - state.committed[me], 0)
        pot_after_call = sum(state.contributions) + to_call
        seen = (cost - to_call) / pot_after_call if pot_after_call else 0.0
        return seen, state.committed[me] + cost, state.committed[me] + state.stacks[me]

    def _bridge_perceive(self, state, me: int, fraction: float, hand,
                         rng: np.random.Generator) -> int:
        """What the live bridge reads this raise as, from ``chipzen.bridge._as_abstract``."""
        from chipzen.bridge import _as_abstract
        seen, level, ceiling = self._bridge_view(state, me, fraction)
        node = hand.node
        node.history = state.history
        node.raises_this_street = sum(1 for a in state.history.split("/")[-1] if a in "2345")
        return _as_abstract(seen, level, ceiling, node, self.game.raise_cap, hand, rng)

    def _bridge_distribution(self, state, me: int,
                             fraction: float) -> Tuple[List[int], np.ndarray]:
        """
        Every action the bridge may read this raise as, with its probability.

        Pricing wants the expectation, while ``_as_abstract`` draws one reading.
        Its all-in readings are deterministic, so they are taken from it directly
        on a scratch hand; otherwise it draws from the pseudo-harmonic pair over
        the schedule's sized raises at this depth, and that pair is laid out here
        with the same inputs. A test samples ``_as_abstract`` against this.
        """
        from chipzen.bridge import RAISE_FRACTIONS
        read = self._bridge_perceive(state, me, fraction, self._bridge_hand(me),
                                     np.random.default_rng(0))
        if read == ALL_IN:
            return [ALL_IN], np.ones(1)
        depth = sum(1 for a in state.history.split("/")[-1] if a in "2345")
        sized = [a for a in raise_sizes_at(self.game.raise_cap, depth, state.street) if a != ALL_IN]
        fractions = [RAISE_FRACTIONS[a - 2] for a in sized]
        seen, _, _ = self._bridge_view(state, me, fraction)
        return sized, translation_distribution(fractions, max(seen, fractions[0]))

    def _street_closes(self, state) -> None:
        """A street is about to be dealt: let the bridge re-read it (``_close_street``)."""
        if self._bridge is None:
            return
        from chipzen.bridge import _close_street
        self._bridge.node.history = state.history
        before = self._bridge.collapsed
        _close_street(self._bridge, self.game.raise_cap)
        self.probe_stats["collapsed_streets"] += self._bridge.collapsed - before

    def _alt_history(self, history: str) -> Optional[str]:
        """
        ``history`` with the bridge's collapsed streets re-read, or None.

        The same splice ``replay`` applies to its own history once it has
        walked the hand; the edits sit before the current street, so they
        apply to any continuation of the history they were made on.
        """
        if self._bridge is None or not self._bridge._edits:
            return None
        alt = history
        for position, replacement in sorted(self._bridge._edits, reverse=True):
            alt = alt[:position] + replacement + alt[position + 2:]
        return alt

    def _policy(self, bucket, history: str, num_actions: int) -> Optional[np.ndarray]:
        """
        The strategy's answer as the live player would look it up.

        The true history first. After a collapsed street it is a line the tree
        treats as ended, so the player asks again on the re-read history
        (``ChipzenPlayer.decide``, the "collapsed" stage). The companion solver
        it tries in between is not modelled: LBR measures one strategy.
        """
        probabilities = self.strategy.get(f"{bucket}|{history}")
        if probabilities is not None and probabilities.size == num_actions:
            return probabilities
        alt = self._alt_history(history)
        if alt is None:
            return None
        probabilities = self.strategy.get(f"{bucket}|{alt}")
        if probabilities is None or probabilities.size != num_actions:
            return None
        return probabilities

    def _deal_range(self, state, me: int, rng: np.random.Generator) -> Range:
        """
        A fresh uniform range over hands the opponent could hold.

        Sampled from the deck less LBR's own cards. The board is *not* excluded
        here — it is empty at this point — so candidates that later collide with
        a board card are dropped as it comes out, which is the same conditioning
        applied at the right time.
        """
        blocked = set(int(c) for c in state.hole[me])
        available = [c for c in range(len(FULL_DECK)) if c not in blocked]

        hands: List[Tuple[int, int]] = []
        for _ in range(self.candidates):
            picked = rng.choice(len(available), size=2, replace=False)
            hands.append((available[int(picked[0])], available[int(picked[1])]))

        return Range(hands, np.full(len(hands), 1.0 / len(hands)))

    def _sync(self, candidate_range: Range, board: Tuple[int, ...]) -> None:
        """
        Bring the range up to date with the board.

        Two things happen when a street lands: candidates holding a card that
        just appeared on the board become impossible and lose their weight, and
        every survivor must be re-bucketed against the new board, because the
        bucket it had on the previous street was drawn from a different
        clustering entirely.
        """
        board = tuple(int(c) for c in board)
        if candidate_range.board == board:
            return

        on_board = set(board)
        possible = np.array([0.0 if (a in on_board or b in on_board) else 1.0
                             for a, b in candidate_range.hands])

        weights = candidate_range.weights * possible
        total = weights.sum()
        if total > 0.0:
            candidate_range.weights = weights / total
        elif possible.sum() > 0.0:
            # Everything that survived the board had already been ruled out by
            # the betting. Keep the survivors, drop the read.
            candidate_range.weights = possible / possible.sum()
        else:
            # Every candidate collides with the board. Vanishingly rare, but it
            # must not produce NaNs: fall back to a flat read over all of them.
            candidate_range.weights = np.full(len(candidate_range.hands),
                                              1.0 / len(candidate_range.hands))

        candidate_range.buckets = [
            self.game.bucket_for(hand, board) if weight > 0.0 else -1
            for hand, weight in zip(candidate_range.hands, candidate_range.weights)
        ]
        candidate_range.board = board

    # ------------------------------------------------------------------

    def _action_probabilities(self, candidate_range: Range, index: int,
                              state, num_actions: int) -> Optional[np.ndarray]:
        """How a given candidate hand would act here, or None if unmodelled."""
        return self._policy(candidate_range.buckets[index], state.history, num_actions)

    def _update_belief(self, candidate_range: Range, state,
                       action_index: int) -> None:
        """
        Bayes: reweight each candidate by how likely it was to act this way.

        A hand that would never have played like this drops out, which is the
        entire source of LBR's leverage — it is exploiting the information the
        betting gives away. Hands the strategy has no entry for are left alone
        rather than guessed at, so an unmodelled line neither confirms nor
        eliminates anything.
        """
        self._sync(candidate_range, state.board)
        num_actions = len(self.game.legal_actions(state))
        uniform = 1.0 / num_actions

        updated = candidate_range.weights.copy()
        for index in candidate_range.live():
            probabilities = self._action_probabilities(candidate_range, index,
                                                       state, num_actions)
            likelihood = (uniform if probabilities is None
                          else probabilities[action_index])
            updated[index] *= likelihood

        total = updated.sum()
        if total <= 0.0:
            # The observed action was impossible for every candidate, so the
            # strategy is not the one being modelled here. Keep the previous
            # read rather than inventing a new one.
            return
        candidate_range.weights = updated / total

    def _showdown_samples(self, state, me: int, candidate_range: Range,
                          rng: np.random.Generator):
        """
        Sample showdowns once, so every action can be priced against them.

        Each sample records *which* candidate hand it came from alongside the
        result, which is what lets the same draws be re-weighted per action
        rather than resampled. That matters because the range LBR is up against
        is not the same for every action: an opponent who calls a large bet is
        stronger than the same opponent's range as a whole. Pricing each action
        against its own conditioned range would otherwise cost one full set of
        rollouts per candidate bet.

        Candidates are drawn uniformly over the live ones rather than by belief,
        so the samples carry no weighting of their own and any weighting can be
        applied afterwards.
        """
        self._sync(candidate_range, state.board)
        live = candidate_range.live()
        if live.size == 0:
            return None

        known = set(int(c) for c in state.board) | set(int(c) for c in state.hole[me])
        runout = 5 - len(state.board)
        mine = [FULL_DECK[c] for c in state.hole[me]]
        board = [FULL_DECK[c] for c in state.board]

        drawn_from = np.empty(self.rollout_samples, dtype=np.int64)
        outcomes = np.empty(self.rollout_samples, dtype=np.float64)

        for sample in range(self.rollout_samples):
            index = int(live[rng.integers(live.size)])
            first, second = candidate_range.hands[index]

            blocked = known | {first, second}
            available = [c for c in range(len(FULL_DECK)) if c not in blocked]
            picked = (rng.choice(len(available), size=runout, replace=False)
                      if runout else ())
            completed = board + [FULL_DECK[available[int(c)]] for c in picked]

            ours = evaluate_hand_fast(mine + completed)
            theirs = evaluate_hand_fast(
                [FULL_DECK[first], FULL_DECK[second]] + completed)

            drawn_from[sample] = index
            outcomes[sample] = (1.0 if ours > theirs
                                else 0.5 if ours == theirs else 0.0)

        return drawn_from, outcomes

    @staticmethod
    def _win_against(samples, weights: np.ndarray) -> float:
        """Chance of winning against a range described by ``weights``."""
        if samples is None:
            return 0.5
        drawn_from, outcomes = samples
        applied = weights[drawn_from]
        total = applied.sum()
        if total <= 0.0:
            return 0.5
        return float((applied * outcomes).sum() / total)

    def _win_probability(self, state, me: int, candidate_range: Range,
                         rng: np.random.Generator) -> float:
        """Chance of winning a showdown against the believed range as it stands."""
        samples = self._showdown_samples(state, me, candidate_range, rng)
        return self._win_against(samples, candidate_range.weights)

    def _fold_probabilities(self, state, candidate_range: Range) -> np.ndarray:
        """
        How often *each* candidate hand folds to the action just taken.

        Per hand rather than pooled, because the pooled figure cannot answer the
        question that matters for pricing a bet: who is left when they do not
        fold. A hand the strategy has no entry for is treated as never folding,
        which keeps it in the calling range rather than quietly assuming it goes
        away.
        """
        folds = np.zeros(len(candidate_range.hands))
        actions = self.game.legal_actions(state)
        if FOLD not in actions:
            return folds
        index = list(actions).index(FOLD)

        self._sync(candidate_range, state.board)
        for candidate in candidate_range.live():
            probabilities = self._action_probabilities(candidate_range, candidate,
                                                       state, len(actions))
            if probabilities is not None:
                folds[candidate] = probabilities[index]
        return folds

    def _fold_probability(self, state, candidate_range: Range) -> float:
        """Chance the opponent folds to the action just taken, over the range."""
        folds = self._fold_probabilities(state, candidate_range)
        return float(np.clip((candidate_range.weights * folds).sum(), 0.0, 1.0))

    def _abstract_raises(self, actions: Sequence[int]) -> Tuple[List[int], List[float]]:
        """The legal sized raises and their pot fractions, smallest first."""
        sized = sorted((RAISE_FRACTION[a], a) for a in actions if a in RAISE_FRACTION)
        return [action for _, action in sized], [fraction for fraction, _ in sized]

    def _fold_probabilities_off_tree(self, state, fraction: float,
                                     perceived: Sequence[int],
                                     sizes: Sequence[float],
                                     candidate_range: Range) -> np.ndarray:
        """
        How often each candidate folds to a bet the abstraction does not contain.

        The bet is perceived as one of the neighbouring abstract sizes, so the
        answer is an average over both perceptions weighted by how likely each
        is — not the answer to whichever one happens to be nearer. Per hand
        rather than pooled, because the caller needs to know who is left when
        they do not fold.
        """
        weights = translation_distribution(sizes, fraction)
        folds = np.zeros(len(candidate_range.hands))
        for weight, action in zip(weights, perceived):
            if weight <= 0.0:
                continue
            after = self.game.raise_by_fraction(state, fraction, action)
            folds += weight * self._fold_probabilities(after, candidate_range)
        return folds

    def _choose(self, state, me: int, candidate_range: Range,
                rng: np.random.Generator) -> "Move":
        """The move with the highest one-step value; see :meth:`_candidates`."""
        moves, values = self._candidates(state, me, candidate_range, rng)
        return moves[int(np.argmax(values))]

    def _candidates(self, state, me: int, candidate_range: Range,
                    rng: np.random.Generator) -> Tuple[List["Move"], List[float]]:
        """
        Every move LBR considers here, with its one-step value.

        Folding is worth losing what we already put in. Any other move is valued
        by rolling the hand out to showdown, plus — for a raise — the chance the
        opponent simply folds.

        **A raise is priced against the range that would call it**, which is not
        the range as a whole. Folding removes the weak hands, so whoever calls is
        stronger than average, and more so the larger the bet. Valuing the
        called branch against the unconditioned range therefore overstates LBR's
        equity by an amount that grows with bet size, which makes shoving look
        free: measured directly, that mistake had LBR aggressive on 52% of
        decisions against a converged strategy and losing 4.9 chips a hand.

        Its predecessor erred the other way — counting no call at all, so every
        chip bet was pure downside and the cheapest legal bet won by
        construction. Both are approximations of the same correct treatment:
        condition on the response, then evaluate.

        The showdowns are sampled once and re-weighted per candidate, so
        conditioning each bet on its own calling range costs almost nothing.

        The raises considered are **not** restricted to the abstraction's. A
        best response is not confined to its opponent's action set.
        """
        actions = list(self.game.legal_actions(state))
        opponent = 1 - me

        samples = self._showdown_samples(state, me, candidate_range, rng)
        belief = candidate_range.weights

        mine_in = state.contributions[me]
        theirs_in = state.contributions[opponent]

        def priced(after, folds_each: np.ndarray) -> float:
            """Value a raise, conditioning the showdown on being called."""
            folds = float(np.clip((belief * folds_each).sum(), 0.0, 1.0))

            calling = belief * (1.0 - folds_each)
            win = self._win_against(samples, calling)

            to_call = max(after.committed[me] - after.committed[opponent], 0)
            called = min(to_call, after.stacks[opponent])
            showdown = (win * (after.contributions[opponent] + called)
                        - (1.0 - win) * after.contributions[me])
            return folds * theirs_in + (1.0 - folds) * showdown

        moves: List[Move] = []
        values: List[float] = []

        for action in actions:
            if action == FOLD:
                moves.append(Move(action, None))
                values.append(-float(mine_in))
            elif action == CHECK_CALL:
                after = self.game.next_state(state, action)
                win = self._win_against(samples, belief)
                moves.append(Move(action, None))
                values.append(win * after.contributions[opponent]
                              - (1.0 - win) * after.contributions[me])
            elif action == ALL_IN:
                after = self.game.next_state(state, action)
                moves.append(Move(action, None))
                values.append(priced(after, self._fold_probabilities(
                    after, candidate_range)))

        perceived, sizes = self._abstract_raises(actions)
        if perceived and self.bridge_translation:
            for fraction in self.bet_sizes:
                readings, weights = self._bridge_distribution(state, me, fraction)
                after = self.game.raise_by_fraction(state, fraction, readings[0])
                folds_each = np.zeros(len(candidate_range.hands))
                for weight, reading in zip(weights, readings):
                    if weight > 0.0:
                        folds_each += weight * self._fold_probabilities(
                            self.game.raise_by_fraction(state, fraction, reading), candidate_range)
                moves.append(Move(None, fraction))
                values.append(priced(after, folds_each))
        elif perceived:
            for fraction in self.bet_sizes:
                after = self.game.raise_by_fraction(state, fraction, perceived[0])
                folds_each = self._fold_probabilities_off_tree(
                    state, fraction, perceived, sizes, candidate_range)
                moves.append(Move(None, fraction))
                values.append(priced(after, folds_each))

        if self._probe_node(state, me):
            # Priced against a scratch copy of the bridge's hand, so weighing a
            # probe records nothing; only the one played is recorded.
            for fraction in self.offtree_third_raise:
                if self._probe_cost(state, me, fraction) is None:
                    continue
                perceived = self._bridge_perceive(state, me, fraction,
                                                  self._bridge_hand(me), rng)
                after = self.game.raise_by_fraction(state, fraction, perceived)
                moves.append(Move(None, fraction, probe=True))
                values.append(priced(after, self._fold_probabilities(after, candidate_range)))

        return moves, values

    def _apply_move(self, state, move: "Move", rng: np.random.Generator):
        """
        Play a chosen move, translating it if the abstraction lacks it.

        The perception is *sampled* rather than rounded. A deterministic mapping
        has a boundary, and a boundary is a thing an exploiter sits just inside
        of — which would inflate the bound with an artifact of the mapping
        rather than a fact about the strategy.
        """
        if move.fraction is None:
            return self.game.next_state(state, move.action)

        if move.probe:
            me = self.game.current_player(state)
            if self._bridge is None:
                self._bridge = self._bridge_hand(me)
            perceived = self._bridge_perceive(state, me, move.fraction, self._bridge, rng)
            self.probe_stats["probes"] += 1
            return self.game.raise_by_fraction(state, move.fraction, perceived)

        if self.bridge_translation:
            # The hand's own bridge record, so a pseudo all-in that is called
            # is re-read by ``_close_street`` when the next street is dealt.
            me = self.game.current_player(state)
            if self._bridge is None:
                self._bridge = self._bridge_hand(me)
            before = len(self._bridge.pseudo_allins)
            perceived = self._bridge_perceive(state, me, move.fraction, self._bridge, rng)
            if len(self._bridge.pseudo_allins) > before:
                self.probe_stats["overbet_allins"] += 1
            elif perceived == ALL_IN:
                self.probe_stats["capped_allins"] += 1
            return self.game.raise_by_fraction(state, move.fraction, perceived)

        perceived, sizes = self._abstract_raises(self.game.legal_actions(state))
        index = translate(sizes, move.fraction, rng)
        return self.game.raise_by_fraction(state, move.fraction, perceived[index])

    # ------------------------------------------------------------------

    def play(self, hands: int, rng: Optional[np.random.Generator] = None,
             alternate_seats: bool = True,
             paired_seed: Optional[int] = None) -> LBRResult:
        """
        Play ``hands`` hands as the exploiter and report the average won.

        With ``paired_seed`` each hand draws from its own generators, one for
        the cards and one for every decision, both seeded by the hand's index.
        Two runs that differ only in a menu then play the same cards, and the
        same decisions until the first one that differs, so their per-hand
        difference is zero wherever the change was never used. Without it a
        single generator is shared, as it always was, and a change anywhere
        reshuffles every later deal.
        """
        rng = rng if rng is not None else np.random.default_rng()
        outcomes = []
        for hand in range(hands):
            if paired_seed is not None:
                def one(seat):
                    return self._one(seat, np.random.default_rng([paired_seed, hand, seat, 1]),
                                     np.random.default_rng([paired_seed, hand, seat, 0]))
            else:
                def one(seat):
                    return self._one(seat, rng)
            if alternate_seats:
                outcomes.append((one(0) + one(1)) / 2.0)
            else:
                outcomes.append(one(0))

        values = np.asarray(outcomes, dtype=np.float64)
        stderr = (values.std(ddof=1) / np.sqrt(values.size)
                  if values.size > 1 else float("nan"))
        return LBRResult(values.size, float(values.mean()), float(stderr), values)

    def _one(self, me: int, rng: np.random.Generator,
             deal_rng: Optional[np.random.Generator] = None) -> float:
        """One hand with LBR in seat ``me``; returns chips won by LBR."""
        game = self.game
        deal_rng = deal_rng if deal_rng is not None else rng
        state = game.initial_state()
        candidate_range = None
        guard = 0
        records, awaiting = [], None
        self._bridge = None

        while not game.is_terminal(state):
            guard += 1
            if guard > 200:
                raise RuntimeError(f"hand did not terminate: {state.history!r}")

            if game.is_chance(state):
                if state.hole:
                    self._street_closes(state)
                state = game.next_state(state, game.sample_chance(state, deal_rng))
                if candidate_range is None and state.hole:
                    candidate_range = self._deal_range(state, me, rng)
                continue

            player = game.current_player(state)
            actions = list(game.legal_actions(state))

            if player == me:
                move = self._choose(state, me, candidate_range, rng)
                record = None
                if self.trace is not None and move.fraction is not None:
                    own = np.random.default_rng(7919 * (len(self.trace) + len(records)) + 1)
                    record = {"street": int(state.street), "size": float(move.fraction),
                              "raise": bool(state.committed[1 - me] > state.committed[me]),
                              "pot_bb": sum(state.contributions) / game.big_blind,
                              "equity": float(self._win_probability(state, me, candidate_range, own))}
                    if move.probe:
                        record["probe"] = True
                        record["depth"] = sum(1 for a in state.history.split("/")[-1] if a in "2345")
                state = self._apply_move(state, move, rng)
                if record is not None:
                    record["perceived"] = int(state.history[-1])       # the size the strategy saw
                    records.append(record)
                    awaiting = record
                continue

            if self._bridge is None:
                key = game.information_set(state, player)
                probabilities = self.strategy.get(key)
            else:
                probabilities = self._policy(game._bucket(state, player), state.history, len(actions))
                if self._alt_history(state.history) is not None:
                    self.probe_stats["alt_hits" if probabilities is not None else "alt_misses"] += 1
            if probabilities is None or probabilities.size != len(actions):
                probabilities = np.full(len(actions), 1.0 / len(actions))
            index = int(rng.choice(len(actions), p=probabilities))
            if self._bridge is not None and state.history.endswith(str(ALL_IN)) \
                    and self._bridge.pseudo_allins \
                    and self._bridge.pseudo_allins[-1][0] == len(state.history) - 1:
                self.probe_stats["folded" if actions[index] == FOLD else "called"] += 1
            if awaiting is not None:
                answer = actions[index]
                awaiting["answer"] = "fold" if answer == FOLD else ("call" if answer == CHECK_CALL else "raise")
                awaiting = None
            self._update_belief(candidate_range, state, index)
            state = game.next_state(state, actions[index])

        result = game.utility(state, me)
        if records:
            for record in records:
                record["result_bb"] = result / game.big_blind
            self.trace.extend(records)
        return result


def lbr_value(game, strategy: Dict[Hashable, np.ndarray], hands: int = 2000,
              rng: Optional[np.random.Generator] = None,
              rollout_samples: int = 60, candidates: int = 32,
              bet_sizes: Sequence[float] = DEFAULT_BET_SIZES,
              trace: Optional[list] = None,
              offtree_third_raise: Sequence[float] = (),
              paired_seed: Optional[int] = None,
              bridge_translation: bool = False) -> LBRResult:
    """
    Lower bound on the exploitability of ``strategy``, in chips per hand.

    Positive and clear of zero means the strategy is provably exploitable by at
    least this much. Near zero means only that LBR failed to exploit it — not
    that it is close to equilibrium. A negative value is not negative
    exploitability, which cannot exist: it means this particular greedy
    exploiter lost money, so the bound is slack and says nothing at all.
    """
    return LocalBestResponse(game, strategy, rollout_samples, candidates,
                             bet_sizes, trace, offtree_third_raise,
                             bridge_translation).play(
                                 hands, rng, paired_seed=paired_seed)
