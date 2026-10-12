"""
When a long run should save itself.

The convention for this project is **every 10% of the run**, whatever its unit —
generations, solver iterations, measurements. Ten saves per run, so the cost of
an interruption is bounded by a proportion rather than by an absolute count that
means something different in every experiment.

Two failures this replaces, both real. A 59,050-iteration solver was
checkpointed every 2,500 iterations, which sounds careful until the same
constant is reused on a run a hundred times longer and writes four hundred
files. And the deep exploitability run wrote nothing until the end, so when it
was killed at ninety-three minutes it left nothing at all.

The trade-off is deliberate and worth stating: an interruption now costs up to a
tenth of the run rather than a single unit. On a two-hour job that is about
twelve minutes, against the cost of writing ten times as often on a long one.
Where a single unit is genuinely expensive to redo, pass a smaller fraction
rather than reaching for a bare constant.
"""
from __future__ import annotations


def checkpoint_every(total_units: int, fraction: float = 0.10) -> int:
    """
    Units between checkpoints, for a run of ``total_units``.

    Always at least 1, so a short run checkpoints every unit rather than never —
    a run of five generations saving "every 10%" must not round down to zero and
    silently keep nothing.

    Args:
        total_units: Length of the run, in whatever unit it counts.
        fraction: Proportion of the run between saves. Defaults to 10%.

    Returns:
        Units between checkpoints, at least 1 and never more than the run.
    """
    if total_units <= 0:
        return 1
    interval = int(round(total_units * fraction))
    return max(1, min(interval, total_units))


def checkpoint_points(total_units: int, fraction: float = 0.10) -> list:
    """
    The unit indices at which a run should save, for logging a plan up front.

    Useful when a run wants to say what it will do before it does it, so an
    interrupted run can be compared against what was expected rather than
    guessed at.
    """
    step = checkpoint_every(total_units, fraction)
    return list(range(step, total_units + 1, step))
