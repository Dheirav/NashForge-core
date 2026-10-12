"""
One panel every agent is measured against, whatever produced it.

The failure this exists to prevent is the one the audit found: agents evaluated
against other agents from the same lineage, so a population could improve
against itself for 89 runs while getting worse at poker, and nothing in the
numbers said so. A benchmark is only worth having if it does not share the
lineage of what it measures.

The panel is `random`, `always-call`, and **the CFR agent** — the last of which
was validated against Kuhn poker's analytic value of −1/18 and exact
exploitability on Leduc, and shares no ancestry with an evolutionary or
policy-gradient run.

Both games, one match
---------------------
The solver lives in :mod:`games.nolimit`, a traversable reimplementation; the
networks train against :class:`engine.PokerGame`. Matches are played in the
*engine*, so no agent is evaluated somewhere other than where it was trained —
a train/evaluate split is precisely how the previous feature bugs hid.

That works because both sides already share the same six abstract actions
(fold, check/call, half pot, pot, two pot, all-in). Since this harness drives
the loop, it knows which abstract action each agent chose and maintains the
solver's ``bucket|history`` key itself, so the CFR agent's lookup is exact
rather than translated. Two constraints are applied to make the engine's legal
set agree with the tree the solver was trained on:

* **A raise cap**, matching the solver's. The engine mask permits unlimited
  raises per street; the solver has never seen those histories.
* **No folding with nothing to call.** The engine offers it, the solver's tree
  excludes it as dominated. An agent that takes it is throwing away a free
  card, so removing it costs nothing and keeps the histories aligned.

Where the lookup misses anyway — an unreached information set — the CFR agent
plays uniformly over legal actions, and :attr:`BenchmarkResult.lookup_miss_rate`
reports how often. A benchmark quietly degrading to a random opponent is worse
than no benchmark, so the rate is measured rather than assumed small.

Duplicate play
--------------
Each hand is dealt twice from the same seed with the seats swapped, so both
agents hold the same cards against the same opposing cards and card luck
cancels rather than being averaged over. The engine reseeds per hand, so this
is exact rather than best-effort.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np

from abstraction.betting import ALL_IN, CHECK_CALL, FOLD, raise_sizes_at
from abstraction.equity import card_index
from abstraction.betting import legal_actions as solver_legal_actions
from engine import Action, PokerGame, get_abstract_action_mask

#: An agent is called with the game, its seat, the legality mask over the six
#: abstract actions, and the solver-format betting history so far. It returns an
#: index into the mask. Anything that does not need the history ignores it.
Agent = Callable[[PokerGame, int, np.ndarray, str], int]

NUM_ACTIONS = 6
RAISE_ACTIONS = (2, 3, 4, 5)


@dataclass
class BenchmarkResult:
    """One matchup, with the uncertainty attached rather than implied."""
    opponent: str
    hands: int
    chips_per_hand: float
    stderr: float
    big_blind: int
    lookup_miss_rate: float = 0.0

    @property
    def ci95(self) -> tuple:
        return (self.chips_per_hand - 1.96 * self.stderr,
                self.chips_per_hand + 1.96 * self.stderr)

    @property
    def bb_per_100(self) -> float:
        return self.chips_per_hand / self.big_blind * 100

    @property
    def separated_from_zero(self) -> bool:
        return abs(self.chips_per_hand) > 1.96 * self.stderr

    def summary(self) -> str:
        low, high = self.ci95
        verdict = "beats" if self.chips_per_hand > 0 else "loses to"
        if not self.separated_from_zero:
            verdict = "not separated from"
        return (f"vs {self.opponent:<14} {self.chips_per_hand:+7.3f} "
                f"+/- {self.stderr:.3f} chips/hand  [{low:+.3f}, {high:+.3f}]  "
                f"= {self.bb_per_100:+7.1f} BB/100   {verdict} it")


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def random_agent(rng: np.random.Generator) -> Agent:
    """Uniform over whatever is legal. The floor any agent must clear."""
    def act(game, player_id, mask, history):
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal))
    return act


def always_call_agent() -> Agent:
    """
    Never folds, never raises. A weak opponent, but a *specific* one — beating
    it says the agent is not simply passive, and losing to it is diagnostic.
    """
    def act(game, player_id, mask, history):
        return CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])
    return act


def _solver_actions(history: str, to_call: int, raise_cap: int):
    """
    The action list the solver used at this node, in its own order.

    The solver stores a probability per *legal* action, not one per abstract
    action: mostly arrays of 2 or 5, and only rarely 6. Reading such an array as
    though slot i meant abstract action i is meaningless, and rejecting it for
    being the wrong length throws away a perfectly good entry. Either way the
    entry has to be placed against the actions it was fitted for, which are
    reconstructible from the betting so far.
    """
    street = history.split("/")[-1]
    raises_so_far = sum(1 for c in street if int(c) in RAISE_ACTIONS)
    last = int(street[-1]) if street else None
    return list(solver_legal_actions(raises_so_far, to_call > 0, raise_cap, last,
                                     street=history.count("/")))


#: Where purification applies: nowhere (the average strategy as stored),
#: postflop only (Baby Tartanian8's choice, preflop left mixed), or everywhere.
PURIFY_MODES = ("none", "postflop", "all", "allx")


ON_MISS_MODES = ("random", "call")


def cfr_agent(strategy: Dict[Hashable, np.ndarray], abstraction,
              rng: np.random.Generator, misses: Optional[List[int]] = None,
              raise_cap: int = 1, probe: Optional[List] = None,
              purify: str = "none", on_miss: str = "random",
              stack_cap: bool = False) -> Agent:
    """
    A solved strategy, playing in the engine.

    ``purify`` plays the most probable action rather than sampling. Its stated
    purpose in the ACPC record (Ganzfried and Sandholm 2012; Tartanian7, 2015:
    +17 to +25 mbb/h against Slumbot-class bots) is to compensate for an
    unconverged average strategy, which is where this project's blueprints are.
    Off by default, because it is a change to the instrument every panel figure
    was measured with, and it raises worst-case exploitability against an
    opponent who adapts.

    The key is rebuilt here rather than read off the engine: the bucket comes
    from the cards, which are the same objects in both games, and the history is
    the one this harness has been maintaining.

    The stored probabilities are then spread onto the six abstract actions
    through the solver's own legal-action list for that node. Treating the array
    as already six-wide was wrong twice over — it discarded 77.5% of successful
    lookups as "missing", and had the lengths happened to match it would have
    read the numbers against the wrong actions silently.

    ``misses`` is a two-slot counter, [missed, consulted], so a benchmark that
    has quietly become a second random opponent shows up as a number.

    ``on_miss`` is what a lookup miss plays: ``random`` (the panel's uniform
    draw over legal actions, the default every panel figure was measured with)
    or ``call``, check/call where legal else fold. The second exists for the
    cross-tree gate: a cap-2 solve against a one-raise one misses on lines the
    bigger tree never visited, and a random shove there measured the fallback
    rather than the solve (25bb read −29.6 random, −17.5 call, 17 September).

    ``stack_cap`` mirrors the trees' rule that a player who cannot cover the
    bet has only fold and call: both solvers store a two-wide entry there,
    and without the mirror this harness reconstructed the full list, rejected
    the entry as the wrong width, and played the miss policy at exactly the
    "facing a big bet" nodes. 34% of a cap-2 rung's keys are such nodes (19
    September); a one-raise tree's two-wide nodes happen to match anyway,
    which is why only cap-2 solves suffered. Off by default because the arena
    player's chip scale is the bridge's, not this engine's; on in the gates.

    ``probe`` is the same idea for the viewer: it receives the distribution
    actually sampled from, or ``None`` where no entry existed and the choice
    was a guess. The GUI shows the policy, and re-deriving it alongside the
    agent is how the shown numbers and the played ones drift apart.
    """
    # Buckets memoised on the cards, and seeded from them.
    #
    # Two problems, one fix. The lookup was recomputing a Monte Carlo equity
    # estimate on every decision of a street for cards that had not changed, and
    # it passed the agent's own advancing generator, so the SAME hole and board
    # could bucket differently from one decision to the next. Training seeds each
    # situation from its own key and therefore gives it one fixed bucket, so the
    # agent being scored was bucketing inconsistently with the way the strategy
    # it plays was fitted, and adding noise to every measurement.
    from cfr.purify import valid as _purify_valid, row as _purify_row
    if purify not in PURIFY_MODES and not _purify_valid(purify):
        raise ValueError(f"purify must be one of {PURIFY_MODES} or tNN (a threshold), got {purify!r}")
    if on_miss not in ON_MISS_MODES:
        raise ValueError(f"on_miss must be one of {ON_MISS_MODES}, got {on_miss!r}")
    buckets: dict = {}

    def bucket_for(hole, board):
        key = (tuple(c.index for c in hole), tuple(c.index for c in board))
        found = buckets.get(key)
        if found is None:
            found = abstraction.bucket(
                hole, board, np.random.default_rng(hash(key) % (2 ** 32)))
            buckets[key] = found
        return found

    def act(game, player_id, mask, history):
        hole = game.players[player_id].hole_cards
        board = game.state.community_cards
        bucket = bucket_for(hole, board)

        probabilities = strategy.get(f"{bucket}|{history}")
        legal = np.flatnonzero(mask)
        if misses is not None:
            misses[1] += 1

        def guess():
            if misses is not None:
                misses[0] += 1
            if probe is not None:
                probe[:] = [None]        # no entry: the choice is not a policy
            if on_miss == "call":
                return CHECK_CALL if mask[CHECK_CALL] else (FOLD if mask[FOLD] else int(legal[0]))
            return int(rng.choice(legal))

        if probabilities is None:
            return guess()

        to_call = game.current_bet - game.players[player_id].bet
        actions = _solver_actions(history, to_call, raise_cap)
        if stack_cap and to_call > 0 and probabilities.size == 2:
            # The tree drops the raise tail when the actor's stack cannot
            # cover the call, and a facing node's list always begins with
            # fold and call, so a two-wide entry there is fold/call whatever
            # the reason. Deciding by the stored width rather than by the
            # engine's stack matters in the arena, where the real stack is
            # not the tree's: with real bet sizes the two boundaries differ,
            # and a stack test turned tree hits into rule decisions and left
            # capped entries unreachable.
            actions = [FOLD, CHECK_CALL]
        if len(actions) != probabilities.size:
            # The reconstruction disagrees with the stored width, so the entry
            # cannot be placed reliably. Rare, and counted rather than guessed at
            # quietly.
            return guess()

        # Plain lists rather than numpy: six elements, where numpy's per-call
        # overhead dwarfs the arithmetic. `_nearest_centroid` uses bisect for
        # the same reason and says so.
        weights = [0.0] * NUM_ACTIONS
        for action, probability in zip(actions, probabilities):
            weights[action] = float(probability) * float(mask[action])

        total = 0.0
        for value in weights:
            total += value
        if total <= 0.0:
            return guess()

        if probe is not None:
            probe[:] = [np.asarray(weights, dtype=np.float64) / total]
        if purify.startswith("t"):
            # Thresholding (cfr/purify.py): drop actions under the share, renormalise, keep sampling.
            weights = list(_purify_row(weights, purify, bool(board)))
            total = sum(weights)
        # "allx": purified everywhere except facing an all-in, where the mix is sampled. Purification turns a
        # mixed fold-or-call row against a very large river shove into an always-one, so the all-in rows keep the mix.
        facing_all_in = to_call > 0 and history.endswith(str(ALL_IN))
        if purify == "all" or (purify == "postflop" and board) or (purify == "allx" and not facing_all_in):
            # The most probable action, first index on a tie: purification
            # rather than sampling, see the docstring.
            best, best_weight = 0, -1.0
            for action, value in enumerate(weights):
                if value > best_weight:
                    best, best_weight = action, value
            return best
        # Inverse-CDF sampling from one uniform draw. `rng.choice(n, p=...)`
        # validates and normalises the distribution on every call, which is most
        # of its cost at this size.
        draw = rng.random() * total
        cumulative = 0.0
        for action, value in enumerate(weights):
            cumulative += value
            if draw < cumulative:
                return action
        return NUM_ACTIONS - 1
    return act


# ---------------------------------------------------------------------------
# Playing a hand
# ---------------------------------------------------------------------------

def _constrain(mask: np.ndarray, to_call: int, raises_this_street: int,
               raise_cap: int, street: int | None = None) -> np.ndarray:
    """
    Narrow the engine's legal set to the tree the solver was trained on.

    Both changes remove options rather than add them, so no agent is ever asked
    for something the engine would reject.
    """
    mask = mask.copy()
    if to_call <= 0:
        mask[FOLD] = 0.0                       # dominated, and off-tree
    # Which raise sizes survive depends on how deep the betting already is, not
    # merely on whether a cap has been hit: a tapered schedule keeps the large
    # sizes and drops the small ones as depth grows. `raise_sizes_at` is the
    # same function `abstraction.betting.legal_actions` uses, so this narrowing
    # cannot drift from the tree the solver was trained on.
    allowed = raise_sizes_at(raise_cap, raises_this_street, street)
    for action in RAISE_ACTIONS:
        if action not in allowed:
            mask[action] = 0.0
    if not mask.any():                         # never strand the actor
        mask[CHECK_CALL] = 1.0
    return mask


def _play_hand(agents: Sequence[Agent], seed: int, starting_stack: int,
               small_blind: int, big_blind: int, raise_cap, raise_caps=None) -> float:
    """
    One hand. Returns chips won by ``agents[0]``, who sits in seat 0.

    The seed fixes the deal, so calling this twice with the seats swapped gives
    both agents the same cards — duplicate play, exactly rather than
    approximately.
    """
    # Imported here: training pulls in torch and multiprocessing, which the arena player never needs.
    from training.fitness import abstract_action_to_engine_action, finish_hand
    game = PokerGame([starting_stack] * 2, small_blind=small_blind,
                     big_blind=big_blind, seed=seed, enable_history=False)

    history = ""
    street = game.state.betting_round
    raises_this_street = 0
    guard = 0

    while not game.is_hand_over():
        guard += 1
        if guard > 200:
            raise RuntimeError(f"hand did not terminate: {history!r}")

        player = game.state.current_player
        if player is None:
            break

        if game.state.betting_round != street:
            street = game.state.betting_round
            history += "/"
            raises_this_street = 0

        actor = game.players[player]
        to_call = game.current_bet - actor.bet
        # Each seat is narrowed to its own tree. With one cap for both, a
        # cap-2 solve could never make the re-raise it was solved with, and
        # every cross-tree gate before 19 September was a cap-2 strategy
        # playing fold/call frequencies meant for a game with a re-raise in it.
        cap = raise_caps[player] if raise_caps is not None else raise_cap
        mask = _constrain(get_abstract_action_mask(game, player), to_call,
                          raises_this_street, cap, street=history.count("/"))

        choice = agents[player](game, player, mask, history)
        if not mask[choice]:
            choice = int(np.flatnonzero(mask)[0])

        history += str(choice)
        if choice in RAISE_ACTIONS:
            raises_this_street += 1

        game.apply_action(player, abstract_action_to_engine_action(choice, game, player))

    # is_hand_over() means the betting is finished, not that the pot has been
    # paid: at showdown the chips are still in the middle. Reading stacks here
    # without settling scores every hand as a loss of whatever was contributed,
    # which looks entirely stable — every deal returned exactly -2 — and is
    # entirely wrong. resolve_showdown also handles a lone survivor, so it must
    # run on fold-outs too.
    finish_hand(game)
    return game.players[0].stack - starting_stack


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

def benchmark(agent: Agent, opponent: Agent, name: str, hands: int = 2000,
              seed: int = 0, starting_stack: int = 200, small_blind: int = 1,
              big_blind: int = 2, raise_cap: int = 1,
              misses: Optional[List[int]] = None, raise_caps=None) -> BenchmarkResult:
    """
    Play ``agent`` against ``opponent`` and report chips per hand to ``agent``.

    Every hand is played twice from one seed with the seats swapped, and the
    two results differenced. Position cancels because each agent sits in both
    seats; card luck cancels because both hold the same cards.

    ``raise_caps`` gives ``(agent's, opponent's)`` raise schedules when the two
    were solved on different trees; each is narrowed to its own, and an agent
    facing a raise its tree lacks takes its miss policy, as it would in play
    without a companion. ``raise_cap`` alone applies one tree to both.
    """
    outcomes = np.empty(hands, dtype=np.float64)
    if misses is None:
        misses = [0, 0]

    for index in range(hands):
        deal = seed * 1_000_003 + index
        caps = None if raise_caps is None else tuple(raise_caps)
        ours = _play_hand([agent, opponent], deal, starting_stack,
                          small_blind, big_blind, raise_cap, caps)
        theirs = _play_hand([opponent, agent], deal, starting_stack,
                            small_blind, big_blind, raise_cap,
                            None if caps is None else (caps[1], caps[0]))
        outcomes[index] = (ours - theirs) / 2.0

    stderr = (float(outcomes.std(ddof=1) / math.sqrt(hands)) if hands > 1
              else float("nan"))
    consulted = max(1, misses[1])
    return BenchmarkResult(opponent=name, hands=hands,
                           chips_per_hand=float(outcomes.mean()), stderr=stderr,
                           big_blind=big_blind,
                           lookup_miss_rate=misses[0] / consulted)


def default_panel(rng: np.random.Generator,
                  cfr_strategy: Optional[Dict[Hashable, np.ndarray]] = None,
                  abstraction=None,
                  misses: Optional[List[int]] = None) -> List[tuple]:
    """
    The opponents every agent is measured against, weakest first.

    The solver is optional only so the panel still works before one has been
    trained. A run without it is measuring against two baselines that any
    competent agent beats, which is a floor check rather than a benchmark.
    """
    panel = [("random", random_agent(rng)), ("always-call", always_call_agent())]
    if cfr_strategy is not None and abstraction is not None:
        panel.append(("cfr", cfr_agent(cfr_strategy, abstraction, rng, misses)))
    return panel
