"""
Playing strategies against each other, for games too large to evaluate exactly.

On Kuhn and Leduc a strategy is measured by traversing the whole tree, which is
exact. No-limit Hold'em cannot be traversed, so the only available measure is to
play hands and count chips — and a chip count over a few thousand hands of poker
is dominated by variance, not by skill.

Two things follow, and both are applied here.

**Every result carries a standard error.** A number without one cannot be
compared to another number. This project has a history of ranking configurations
on differences far inside their own noise; see CODEBASE_AUDIT.md.

**Seats alternate.** In heads-up play position is worth a great deal, so a
strategy measured only in the small blind is measured against a confound. Each
hand is played from both seats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np

from games.base import Game

#: A policy is called with the game, the state, the player to act and the
#: number of legal actions, and returns action probabilities. Returning None
#: means "no opinion", and uniform is used.
#:
#: It receives the STATE rather than just the information-set key so that two
#: agents trained on different abstractions can play each other: each must be
#: able to compute the key its own abstraction would produce. Passing only the
#: game's key would silently feed one agent another's bucketing, and it would
#: look like it was playing badly rather than being asked the wrong question.
Policy = Callable[[Any, Any, int, int], Optional[np.ndarray]]


def strategy_policy(strategy: Dict[Hashable, np.ndarray],
                    abstraction: Any = None) -> Policy:
    """
    Wrap a solved strategy table as a policy.

    With ``abstraction``, keys are computed through that abstraction rather than
    the game's — which is what lets an agent play in a game whose abstraction is
    not its own.
    """
    def policy(game, state, player: int, num_actions: int) -> Optional[np.ndarray]:
        if abstraction is None:
            key = game.information_set(state, player)
        else:
            key = game.information_set_with(state, player, abstraction)
        probabilities = strategy.get(key)
        if probabilities is None or probabilities.size != num_actions:
            return None
        return probabilities
    return policy


def uniform_policy() -> Policy:
    """Plays every legal action equally often."""
    return lambda game, state, player, num_actions: None


def always_call_policy(call_action: int = 1) -> Policy:
    """
    Never folds, never raises — the classic passive baseline.

    A strategy that cannot beat this is not playing poker. Note it also cannot
    be bluffed, so it does not test semi-bluffing or draw play at all.
    """
    def policy(game, state, player: int, num_actions: int) -> Optional[np.ndarray]:
        # The returned array is indexed by *position in the legal list*, not by
        # abstract action. Writing 1.0 at index `call_action` therefore picked
        # whatever happened to sit second: with nothing to call the legal list
        # is [1, 2, 3, 4, 5] -- fold is dropped as dominated -- so index 1 is
        # action 2, raise half pot. This "calling station" raised every time
        # checking was free, which is most of postflop.
        #
        # Found 8 September: it is why `play_hands` and `evaluation.benchmark`
        # gave opposite verdicts on the same two strategies. The benchmark's
        # `always_call_agent` reads the mask by abstract action and was always
        # right; every "vs always call" figure from `train_nolimit.py` was
        # measured against a semi-aggressive opponent wearing the name.
        legal = list(game.legal_actions(state))
        probabilities = np.zeros(num_actions)
        probabilities[legal.index(call_action) if call_action in legal else 0] = 1.0
        return probabilities
    return policy


@dataclass(frozen=True)
class MatchResult:
    """Chips per hand to the first strategy, with its uncertainty."""
    hands: int
    mean: float
    stderr: float

    @property
    def ci95(self) -> tuple:
        return (self.mean - 1.96 * self.stderr, self.mean + 1.96 * self.stderr)

    @property
    def separated_from_zero(self) -> bool:
        """Whether this many hands can tell the result apart from break-even."""
        low, high = self.ci95
        return low > 0.0 or high < 0.0

    def summary(self, per_hand: str = "chips/hand") -> str:
        low, high = self.ci95
        verdict = "" if self.separated_from_zero else "  (not separated from zero)"
        return (f"{self.mean:+.3f} +/- {self.stderr:.3f} {per_hand}"
                f"  95% CI [{low:+.3f}, {high:+.3f}]{verdict}")


def play_hands(game: Game, policies: Sequence[Policy], hands: int,
               rng: Optional[np.random.Generator] = None,
               alternate_seats: bool = True,
               duplicate: bool = True) -> MatchResult:
    """
    Play ``hands`` hands and report the result to ``policies[0]``.

    With ``alternate_seats``, each hand is played twice — once from each seat —
    and the two results averaged, so position cancels rather than being absorbed
    into the estimate.

    With ``duplicate``, the replay is dealt the *same cards*, which is the
    substance of the technique rather than a refinement of it. Swapping seats on
    a fresh deal cancels position and leaves card luck untouched; replaying the
    identical deal with the seats swapped means each policy holds both hands, so
    whoever was dealt the better cards no longer shows up in the difference at
    all. A poker result over a few thousand hands is dominated by that luck, not
    by skill — the audit measured +/-258 BB/100 over 600 hands — and this
    removes far more of it than playing more hands does.

    The runout is shared only as far as both replays reach it: if the first hand
    ends preflop there is no flop to reuse, and the second draws its own. The
    hole cards, which carry most of the variance, are always shared.

    ``duplicate`` changes the variance of the estimate, never its expectation.
    Results measured before it existed are unbiased, only noisier.
    """
    rng = rng if rng is not None else np.random.default_rng()
    outcomes = []

    for _ in range(hands):
        if not alternate_seats:
            outcomes.append(_play_one(game, policies, rng))
            continue

        deal: List[Any] = [] if duplicate else None
        first = _play_one(game, [policies[0], policies[1]], rng, record=deal)
        second = _play_one(game, [policies[1], policies[0]], rng, script=deal)
        outcomes.append((first - second) / 2.0)

    values = np.asarray(outcomes, dtype=np.float64)
    stderr = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else float("nan")
    return MatchResult(hands=values.size, mean=float(values.mean()), stderr=float(stderr))


def _play_one(game: Game, policies: Sequence[Policy],
              rng: np.random.Generator,
              record: Optional[List[Any]] = None,
              script: Optional[Sequence[Any]] = None) -> float:
    """
    One hand; returns chips to the policy seated at index 0.

    ``record`` collects each chance outcome as it is drawn, and ``script``
    replays them in the same order — which is how the duplicate replay is dealt
    the same cards. A script shorter than the hand needs is not an error: the
    replay simply went further than the original did, and draws the rest.
    """
    state = game.initial_state()
    guard = 0
    dealt = 0

    while not game.is_terminal(state):
        guard += 1
        if guard > 200:
            raise RuntimeError(f"hand did not terminate: {state}")

        if game.is_chance(state):
            if script is not None and dealt < len(script):
                outcome = script[dealt]
            else:
                outcome = game.sample_chance(state, rng)
                if record is not None:
                    record.append(outcome)
            dealt += 1
            state = game.next_state(state, outcome)
            continue

        player = game.current_player(state)
        actions = game.legal_actions(state)

        probabilities = policies[player](game, state, player, len(actions))
        if probabilities is None:
            probabilities = np.full(len(actions), 1.0 / len(actions))

        index = int(rng.choice(len(actions), p=probabilities))
        state = game.next_state(state, actions[index])

    return game.utility(state, 0)
