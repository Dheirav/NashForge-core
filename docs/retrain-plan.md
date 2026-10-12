# The retrain, after the C++ port

Written 11 September 2026. The native solver is 32.5x faster and **changes
answers**, so making it the default invalidates work. This says exactly what,
exactly what it costs, and what it buys, so the decision is made on paper rather
than discovered halfway through.

This would be the **third** time this project has invalidated its own panel. The
first was the audit of 12 August, the second the discovery that the panel was a
4,000-iteration solver. Both times the cost was paid because the alternative was
carrying numbers that were quietly wrong. That is not the situation here: the
current numbers are right. They just will not be comparable to anything trained
afterwards.

## Why it changes answers

The betting is identical, checked over 6,216 enumerated sequences. What differs
is bucketing. The Python seeds each equity rollout from Python's own tuple hash,
which is not reproducible in C++ and not worth reproducing for a sampler. So a
hand near a bucket boundary can land either side of it.

That is not a defect and not an improvement. It is a different draw from the same
distribution, and the two solvers were measured playing indistinguishably: 16.2
and 23.1 BB/100 apart against random and always-call, both inside the
measurement interval.

**The consequence is narrow and total.** Any two numbers produced by different
solvers are no longer a comparison. Everything in `results/` was produced by the
Python.

## What is invalidated

**The solvers.** All six in `results/cfr/`. Cheap to replace: a 250,000-iteration
solver was 3h30m this morning and is about **70 seconds** now.

**Everything measured against the panel**, because the panel *is* a solver:

| | where | what it costs to redo |
|---|---|---|
| Phase 4 comparison | `results/comparison/phase4_v2panel.json` | the expensive one, see below |
| PPO endpoint tests | `results/ppo/phase3_endpoint*.json`, 12 files | same measurement |
| Evolution endpoint | `results/evolution/` | same |
| The intransitivity withdrawal | `docs/engine-comparison.md`, `NEXT.md` | re-measure one edge |
| Slumbot | `results/slumbot/m1_200bb_250k.json` | 7 hours of API |
| The bucket sweep | `results/cfr/bucket_sweep_long.json` | see "opportunities" |

**The abstraction crossover is the awkward one.** `+0.916 ± 0.118 chips/hand at
7.8σ` is this project's headline result, and it compares two abstractions at
matched wall-clock budgets. Wall-clock budgets do not survive a 32.5x speedup:
the same 2560 seconds now buys 32 times the iterations, which is a different
point on the curve entirely. The finding is not wrong, but it is now a statement
about a budget nobody will ever spend again.

## What is NOT invalidated

**The agents themselves.** PPO checkpoints and evolved genomes were produced by
self-play, never against the CFR solver. Only their *scores* need redoing, not
their training. That is the difference between hours and weeks.

**The engine, the abstraction fitting, the tests.** Untouched.

**Kuhn and Leduc validation.** Analytic, and the C++ solver reproduces -1/18
exactly.

## Order, and cost

**1. Retrain the solvers.** About 70 seconds each with the native path, or a few
minutes for all six. Keep the Python-trained ones under a `_py` suffix rather
than deleting them: they are the only thing that makes `tests/test_native.py`
meaningful, and this project has twice wanted a withdrawn artifact back.

**2. Re-measure Phase 4 and the endpoint tests.** This is the real cost and it
is *unchanged by the port*, because it is Python evaluation at 40,000 hands per
matchup, not solving. Budget the same as the last pass. Parallelise across seeds:
it is iteration-budgeted, so it is safe to run concurrently, unlike the ladders.

**3. Slumbot, 7 hours of API.** Last, and only once the solver it measures is
final. It has been run twice already for want of that discipline.

**4. The crossover and the bucket sweep: do not re-run as they are.** Both are
wall-clock ladders and both now ask a question about a budget that no longer
exists. They should be redesigned rather than repeated. See below.

## What the speed actually buys, which is the point

These were all priced this week and refused as unaffordable. At 32.5x:

| | was | now |
|---|---|---|
| 250,000-iteration solver | 3h30m | **70s** |
| `[4,1]` taper to matched density | 4.6h | **~9 min** |
| `[4,2]` taper, 1.28M information sets | 27h | **~50 min** |
| full raise cap 2, 7.5M information sets | 158h, refused | **~5h** |
| bucket sweep top rung, 5120s | 134,500 iterations | **~4.4M** |

**The raise cap is the lever the Slumbot research pointed at**, and it stopped
being a two-day question. `docs/engine-comparison.md` set out why: Slumbot allows
eleven bet sizes and unlimited raise depth against our three and one, roughly a
700-fold difference in betting sequences, and its author's own judgment is that
bet resolution is what matters while card resolution can be cheap.

**And the estimator noise finding becomes actionable.** The card abstraction is
noise-limited: sd 0.0667 at 40 samples against bucket centroids 0.101 to 0.170
apart, so 42% of hands change bucket on a re-roll. Raising `equity_samples` to
200 costs 5x, which was unaffordable and now is not. The bucket sweep should be
re-run at that precision, because its conclusion was explicitly conditional on
it.

## The risk, stated plainly

The retrain is cheap and the re-measurement is not. The failure mode is starting
the retrain, getting halfway through the re-measurement, and leaving the project
with a mixture of Python-panel and C++-panel figures that look comparable and are
not. That is exactly what happened in August and it cost three withdrawn claims.

So: **retrain and re-measure as one pass, or not at all.** Every figure in
`NEXT.md` and `docs/RESULTS_SHEET.md` gets restated together, and anything not
yet redone is marked withdrawn rather than left standing.
