"""
The abstract betting tree, and the information-set arithmetic that follows.

Card abstraction bounds how many hands the solver distinguishes; bet abstraction
bounds how many betting lines it distinguishes. Together they decide whether the
game fits in memory, which is the whole feasibility question for running CFR on
a laptop rather than a cluster.

The action space is the project's existing six — fold, check/call, three raise
sizes, all-in — reused as the bet abstraction, with raises capped per street.
That reuse is deliberate but not free: the six actions were designed as a policy
network's output layer, not as a CFR bet abstraction, and **action abstraction
upper-bounds achievable exploitability regardless of how well the solver runs.**
A floor in the exploitability curve may be the action space rather than the
solver, which is why this is stated in the proposal's threats to validity.

Counting information sets exactly needs no engine and no cards: the betting tree
is finite and small, so it is enumerated here, and the total is that count
multiplied by the buckets available on each street.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

#: The existing six-action abstraction, in the project's established order.
FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT, RAISE_TWO, ALL_IN = range(6)
ACTION_NAMES = ("fold", "check/call", "raise-half", "raise-pot", "raise-2x", "all-in")
RAISE_ACTIONS = (RAISE_HALF, RAISE_POT, RAISE_TWO, ALL_IN)

STREETS = ("preflop", "flop", "turn", "river")


@lru_cache(maxsize=None)
def normalise_schedule(spec: int | Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    A raise schedule: which raise sizes are available at each raise depth.

    An **int** N is the uniform schedule this project has always used, all four
    sizes available for N raises, so ``normalise_schedule(1)`` reproduces the
    shipped abstraction exactly.

    A **tuple** tapers, which is how a deeper betting tree becomes affordable.
    ``(4, 1)`` means four sizes for the opening bet and one for the raise, and
    the one kept is the *largest*, because :data:`RAISE_ACTIONS` runs from
    half-pot up to all-in and the deep options are the ones worth keeping. This
    is the shape Slumbot uses: eleven sizes for an initial bet, eight for a
    raise, three for a three-bet, one thereafter, with no cap on depth. Tapering
    is what buys the depth. Measured on this abstraction at six buckets, a
    second raise costs 7,560,240 information sets if it carries all four sizes
    and 347,136 if it carries only all-in, a 22-fold difference for a tree that
    contains a re-raise either way.

    A level may also be a tuple of action ids, which names the sizes rather
    than counting them: ``((RAISE_HALF, RAISE_POT, ALL_IN), (ALL_IN,))`` is a
    two-level schedule whose opening menu skips the 2x-pot raise. Mixed forms
    are allowed, and the ids are sorted into :data:`RAISE_ACTIONS` order so the
    tree's action indices never depend on how the schedule was written.

    Cached because this sits on the hot path of every traversal.
    """
    if isinstance(spec, int):
        return tuple(RAISE_ACTIONS for _ in range(spec))
    width = len(RAISE_ACTIONS)
    levels = []
    for entry in spec:
        if isinstance(entry, (tuple, list)):
            # A level that names its sizes. Counting from the largest end is a
            # choice, not a rule, and it is the wrong one where the menu is
            # short: over a 2bb open the kept pair is "10bb or all-in", and the
            # 6bb three-bet every solver makes does not exist. Naming the sizes
            # buys it back at no cost, since the tree's size depends on how many
            # are kept and not on which: (4,2,1), (4,3) and (3,3,1) are all
            # 3,438,484 information sets.
            chosen = tuple(action for action in RAISE_ACTIONS if action in tuple(entry))
            levels.append(chosen)
        else:
            levels.append(RAISE_ACTIONS[width - max(0, min(int(entry), width)):])
    return tuple(levels)


@dataclass(frozen=True)
class StreetSchedule:
    """
    A raise schedule per street, preflop to river: each entry is any schedule
    :func:`normalise_schedule` takes.

    LBR's traced bets can put a large share of a 70bb rung's leak in two
    preflop sizes the tree lacks, an overbet and a small re-raise where the
    taper offers only 2x and all-in. Added to every street they multiply the
    tree by 4 to 12; added preflop only, by 1.4 to 1.9. The action codes are
    the same on every street, so the history key does not change.

    Every lookup must say which street it is on: :func:`resolve` refuses a
    StreetSchedule without one, so a caller that was never taught the street
    fails loudly instead of quietly using one street's menu for all four.
    """
    streets: Tuple[object, object, object, object]

    def __post_init__(self):
        if len(self.streets) != len(STREETS):
            raise ValueError(f"a street schedule has {len(STREETS)} entries, preflop to river")

    def for_saving(self) -> dict:
        """The form `train_nolimit.py` writes into a solve's args."""
        return {name: [list(level) for level in normalise_schedule(spec)]
                for name, spec in zip(STREETS, self.streets)}


def resolve(spec, street: int | None):
    """The schedule that applies on ``street``: ``spec`` itself unless it is a StreetSchedule."""
    if isinstance(spec, StreetSchedule):
        if street is None:
            raise ValueError("a per-street raise schedule was asked for its sizes without a street")
        return spec.streets[street]
    return spec


def schedule_from_args(value):
    """A solve's saved ``raise_cap`` back to a schedule: int, tuple, or a per-street dict."""
    if isinstance(value, dict):
        return StreetSchedule(tuple(tuple(tuple(level) for level in value[name]) for name in STREETS))
    if isinstance(value, (list, tuple)):
        return tuple(tuple(level) if isinstance(level, (list, tuple)) else int(level) for level in value)
    return value


def raise_sizes_at(spec, depth: int, street: int | None = None) -> Tuple[int, ...]:
    """Raise sizes legal at ``depth`` raises in, empty once the schedule ends."""
    schedule = normalise_schedule(resolve(spec, street))
    return schedule[depth] if depth < len(schedule) else ()


def max_raises(spec, street: int | None = None) -> int:
    """How deep the schedule goes. Equals ``spec`` when ``spec`` is an int."""
    return len(normalise_schedule(resolve(spec, street)))


def legal_actions(raises_so_far: int, facing_bet: bool,
                  raise_cap: int | Tuple[int, ...],
                  last_action: int | None = None,
                  street: int | None = None) -> Tuple[int, ...]:
    """
    Actions available given the state of the current street's betting.

    Folding with nothing to call is legal in poker but strictly dominated, so it
    is left out of the tree — it would double the branching factor to no
    purpose. Once the schedule is exhausted only folding or calling remains.

    An all-in cannot be raised: a player facing one may only fold or call, since
    there are no chips left to raise with. Treating all-in as an ordinary raise
    inflates the tree with lines that cannot occur.

    ``raise_cap`` is an int for the uniform schedule or a tuple to taper; see
    :func:`normalise_schedule`. Every other module reaches the betting rules
    through this function, so a schedule given here reaches the traversal game,
    the solver's action lists and the tree enumeration without further change.
    The two mask builders that mirror these rules rather than calling them,
    ``evaluation.benchmark._constrain`` and ``slumbot.bridge.legal_mask``, use
    :func:`raise_sizes_at` for the same reason.
    """
    if last_action == ALL_IN:
        return (FOLD, CHECK_CALL)

    actions: List[int] = []
    if facing_bet:
        actions.append(FOLD)
    actions.append(CHECK_CALL)
    actions.extend(raise_sizes_at(raise_cap, raises_so_far, street))
    return tuple(actions)


def enumerate_street_sequences(raise_cap: int) -> List[Tuple[Tuple[int, ...], bool]]:
    """
    Every betting sequence for one street.

    Returns ``(sequence, ended_in_fold)`` pairs. A street closes when a bet is
    called or when both players check; a fold ends the hand outright.
    """
    completed: List[Tuple[Tuple[int, ...], bool]] = []

    def walk(sequence: Tuple[int, ...], raises: int, facing: bool, actor: int) -> None:
        last = sequence[-1] if sequence else None
        for action in legal_actions(raises, facing, raise_cap, last):
            extended = sequence + (action,)
            if action == FOLD:
                completed.append((extended, True))
            elif action == CHECK_CALL:
                if facing:                       # called a bet: street closes
                    completed.append((extended, False))
                elif sequence and sequence[-1] == CHECK_CALL:
                    completed.append((extended, False))   # checked through
                else:
                    walk(extended, raises, False, 1 - actor)
            else:                                # a bet or raise
                walk(extended, raises + 1, True, 1 - actor)

    walk((), 0, False, 0)
    return completed


def count_decision_points(raise_cap: int) -> Dict[int, int]:
    """
    Decision points per player within one street, keyed by player index.

    A decision point is a distinct public betting prefix at which that player is
    to act. Multiplied by the number of card buckets, this gives the information
    sets that street contributes.
    """
    counts = {0: 0, 1: 0}

    def walk(sequence: Tuple[int, ...], raises: int, facing: bool, actor: int) -> None:
        counts[actor] += 1
        last = sequence[-1] if sequence else None
        for action in legal_actions(raises, facing, raise_cap, last):
            if action == FOLD:
                continue
            if action == CHECK_CALL:
                if facing or (sequence and sequence[-1] == CHECK_CALL):
                    continue
                walk(sequence + (action,), raises, False, 1 - actor)
            else:
                walk(sequence + (action,), raises + 1, True, 1 - actor)

    walk((), 0, False, 0)
    return counts


@dataclass(frozen=True)
class AbstractionSize:
    """Measured size of an abstract game."""
    buckets: Dict[str, int]
    raise_cap: int
    decision_points_per_street: int
    reaching_sequences: Dict[str, int]
    information_sets: int
    table_bytes: int

    def summary(self) -> str:
        buckets = " ".join(f"{s}={self.buckets[s]}" for s in STREETS)
        return (f"cap={self.raise_cap}  {buckets}  "
                f"infosets={self.information_sets:,}  "
                f"table={self.table_bytes / 1e6:.1f} MB")


def measure(buckets: Dict[str, int], raise_cap: int = 2,
            num_actions: int = 6, bytes_per_entry: int = 8) -> AbstractionSize:
    """
    Information sets and table memory for an abstraction, counted exactly.

    A street is reached once for each way an earlier street could close without
    a fold, so the betting lines multiply across streets. Memory assumes one
    regret and one strategy accumulator per action, which is what CFR stores.
    """
    reaching: Dict[str, int] = {}
    lines = 1
    total = 0
    decisions = 0
    for index, street in enumerate(STREETS):
        # Each street counted with its own schedule: with a StreetSchedule the
        # decision points and the lines that survive differ street by street.
        cap = resolve(raise_cap, index)
        per_street = count_decision_points(cap)
        here = per_street[0] + per_street[1]
        decisions = decisions or here            # reported: the preflop street's
        reaching[street] = lines
        total += lines * here * buckets[street]
        lines *= len([seq for seq, folded in enumerate_street_sequences(cap) if not folded])

    entries = total * num_actions * 2          # regret and strategy sums
    return AbstractionSize(
        buckets=dict(buckets),
        raise_cap=raise_cap,
        decision_points_per_street=decisions,
        reaching_sequences=reaching,
        information_sets=total,
        table_bytes=entries * bytes_per_entry,
    )


#: Names for the raise sizes, for schedules written on a command line.
SIZE_NAMES_TO_ACTION = {"half": RAISE_HALF, "pot": RAISE_POT, "2x": RAISE_TWO,
                        "two": RAISE_TWO, "jam": ALL_IN, "allin": ALL_IN, "all-in": ALL_IN}


def parse_level(text: str):
    """
    One level of a schedule as written for `--raise-cap`: a count, or names.

    ``"3"`` keeps the largest three sizes, as it always has. ``"half,pot,jam"``
    names them, which is the only way to keep a small size and drop a large one.
    """
    text = str(text).strip()
    if text.isdigit():
        return int(text)
    chosen = []
    for name in text.split(","):
        key = name.strip().lower()
        if key not in SIZE_NAMES_TO_ACTION:
            raise ValueError(f"unknown raise size {name!r}; use a count or "
                             f"{', '.join(sorted(set(SIZE_NAMES_TO_ACTION)))}")
        chosen.append(SIZE_NAMES_TO_ACTION[key])
    if not chosen:
        raise ValueError("a named level needs at least one size")
    return tuple(action for action in RAISE_ACTIONS if action in tuple(chosen))
