# Goal: do not lose to Slumbot

**The target.** A measured head-to-head result against Slumbot at **0 mbb/hand or better**,
over enough hands for the confidence interval to mean something.

## Why this is the right goal for this project

Every strength number here is self-produced. Kuhn's −1/18 and exact Leduc exploitability
prove the solver converges to the right thing, and the abstraction crossover (item 2 in
[`BACKLOG.md`](../BACKLOG.md)) measures two of this project's own strategies against each
other. Nothing measures this project against anyone else's work.

That gap is not cosmetic. **Item 1 closed with no usable bound**: after four defects fixed and
three valuation models, LBR still could not beat a 59,050-iteration strategy, so no-limit has
no exploitability figure at all right now. Slumbot answers the same question — how good is
this really — by a route that does not depend on this repo's own machinery being correct.

It is also the right opponent rather than merely an available one. Slumbot plays a fixed CFR
strategy at heads-up no-limit: same game, same technique, no adaptation between sessions, free
public HTTP API with no auth. Results are reported in mbb/hand and published work cites them,
so the number lands in an existing comparison.

## Is the goal reachable

Honestly: it is hard, and it is not absurd.

Against it — Slumbot is a strong near-equilibrium blueprint built with serious compute, and it
won the ACPC. GTO Wizard, a heavily-resourced commercial solver, beats it by 194 ± 41 mbb/hand
over 150,000 hands. That is the scale of effort that clears it comfortably.

For it — Slumbot is itself abstraction-based and does no real-time solving, so it is not an
equilibrium. The LBR literature (Lisý & Bowling, 2017) established that abstraction-based
no-limit bots of this generation retain substantial exploitable weaknesses. Break-even is a
target with a real path to it, unlike beating an actual equilibrium.

Reaching it plausibly needs: a finer bet abstraction, far more iterations, randomized action
translation, and eventually real-time subgame re-solving, which is the technique that separates
blueprint-only agents from the ones that beat strong opponents. Treat re-solving as the thing
the milestones below are walking toward.

## Milestones

Staged, so progress is visible long before the goal is met.

| | Milestone | Meaning |
|---|---|---|
| M0 | One legal hand played end to end against the live API | **Done, 31 August** |
| M1 | 10,000 hands, any result, with a CI | **Done — −987 ± 374 mbb/hand** (2 Sept, converged solver) |
| M2 | Within 200 mbb/hand | Competitive with the blueprint generation |
| M3 | Within 50 mbb/hand | Abstraction and translation are close to sound |
| M4 | **0 mbb/hand or better** | Goal |

M1 is the one that changes the project. Everything after it is improvement against a scale
that already exists.

## M1, as measured — 1 September 2026

**−1750.2 ± 524 mbb/hand**, 95% interval [−2274, −1226], over 10,000 hands
(`results/slumbot/m1.json`, `scripts/slumbot_measure.py`). We lose, heavily, and that is the
first figure in this repository that somebody else's agent produced.

**The health checks, which are what make it a measurement rather than a run that happened:**

| | |
|---|---|
| protocol errors | 0 of 10,000 hands |
| lookup miss rate | 8.7% (1,817/20,897), against the pilot's 9.5% |
| seat split | 5,000 / 5,000, exact |
| bets too large for the abstraction | 413, about 2% of decisions |

An 8.7% miss rate is the genuinely off-tree nodes — Slumbot re-raises and the solver knows one
raise per street — not a broken lookup. For contrast, the row this project withdrew was taken at
74.3%.

**What is actually playing.** A 100bb, one-raise-per-street, six-bucket, 4,000-iteration solver
against a 200bb unlimited-raise opponent built with serious compute. GTO Wizard beats Slumbot by
194 ± 41 mbb/hand; this is 1,750 the other way. The result is what it looks like, and M1 asked
for a number rather than a good one.

### `baseline_winnings` is not a second estimate of the same thing

Slumbot returns a `baseline_winnings` field per hand, and it was tempting: on the pilot it
correlated 0.85 with actual winnings and differencing cut the spread by 37%. The full sample
resolves what it is, and the answer is not "a variance-reduced win rate".

    raw win rate            −1750.2 mbb/hand
    differenced               −68.3 mbb/hand
    ⟹ the baseline's own mean ≈ −1681.9 mbb/hand

A control variate leaves the mean alone only when its own expectation is zero. This one averages
−1682, so differencing changes *what is being estimated*, not merely its precision. The
differenced figure says this agent performed about as well as Slumbot's baseline strategy did
holding the same cards — a real and separate fact, and not the win rate.

Reported as the result: **−1750**. Reporting the −68 would have been wrong by a factor of
twenty-five, in the flattering direction, and this is exactly the trap the cap-2 row set in
item 1. The measurement script prints both and refuses to prefer the second.

### Re-measured with a converged solver — 2 September

M1 was taken with the strategy that shipped, which turned out to have had about **two minutes of
training**: 4,000 iterations at the 32 iterations/second the crossover measured. Retraining the
*same* abstraction — same six buckets, same one-raise cap, same seed — for 150,000 iterations
(2h56m, 70.5 ms/iteration) and re-running gives:

| solver | vs Slumbot | miss rate |
|---|---|---|
| 4,000 iterations | −1750.2 ± 524 mbb/hand | 8.7% |
| **150,000 iterations** | **−986.6 ± 374 mbb/hand** | 10.4% |
| improvement | **+763.7 ± 644** | separated |

Zero protocol errors and an exact 5,000/5,000 seat split on both runs. **Training was worth
roughly half the gap**, and the remaining −987 is what the coarse abstraction, the 100bb depth
mismatch and the one-raise cap cost.

**Internal strength overstated the external gain by about 2.4x.** Head to head on this
project's own instrument the 150k solver beats the 4k one by +185 ± 13 BB/100 — fourteen
standard errors. Against Slumbot the same upgrade is worth +76 ± 64 BB/100, real but barely
separated. Beating your own previous agent is evidence about a third party, not a measurement of
it, which is the same lesson the Phase 4 intransitivity taught in a different place.

**The baseline field disagreed in sign again.** Differenced, this run reads **+384 ± 348** — the
wrong side of zero, claiming a win. Twice now, on two different solvers, reporting the
variance-reduced figure would have inverted the finding. The script prints both and refuses to
prefer the second.

**Iterations are not exhausted.** At 150,000 the solver had reached 25,154 of the abstraction's
49,200 information sets — barely half. There is likely more available from training alone before
abstraction becomes the binding constraint, and it is much cheaper than either widening the
abstraction or moving to 200bb.

### What M2 needs

Two things, and the second is the real one.

**More hands.** At the measured spread of 2,169 chips per hand, ±200 mbb/hand takes about 45,000
hands — roughly 30 hours at the API's pace of 0.4 hands/second. There is no shortcut through the
baseline, for the reason above.

**The stack depth.** Slumbot plays 200bb because that is the ACPC convention and what published
work reports against; this project's 100bb was an unexamined default. Training a solver at 200bb
is a one-time move onto the standard rather than an adaptation to one opponent — the distinction
matters, because retraining per opponent would leave no agent with a fixed identity and no number
comparable across runs. It would also invalidate the existing panel figures unless both solvers
are kept.

Doing that before spending 30 hours of API time is the sensible order.

## The hard part: action translation

Slumbot bets any legal amount. This project plays a six-action bet abstraction
(`abstraction/`). Bridging them is most of the engineering:

- **Inbound** — a Slumbot bet landing between two abstraction sizes has to be mapped onto one
  before a strategy can be looked up. Nearest-size mapping is itself exploitable. Randomized
  translation between the two neighbouring sizes is the standard fix.
- **Outbound** — the chosen abstract action has to become a concrete chip amount.

Build this as its own module with its own tests, separate from the agent. A translation bug is
indistinguishable from a weak strategy by looking at the result.

## Measurement discipline

Non-negotiable here, given item 1's history.

- **10,000 hands minimum.** mbb/hand over a short sample is noise.
- **Duplicate seating** — every hand played twice with seats swapped and identical cards.
  Cheap against a fixed opponent and removes a large share of the variance. AIVAT is better
  and more work.
- **Always a confidence interval.** Never a bare point estimate.
- **Results as JSON in `results/cfr/`**, one file, matching every other measurement here:
  mbb/hand, hand count, CI, agent version, translation-layer version.

**The specific trap, already seen in this repo.** Item 1's cap-2 row read +2.783 and "PROVES
EXPLOITABLE" one afternoon and −2.900 that evening against an identical untouched strategy,
because only the exploiter's internal valuation had changed. Action translation is the same
class of hazard: a modelling choice, invisible in the result, that moves the number. Freeze
and version the translation layer before quoting anything, and run a throwaway pilot of a few
hundred hands whose result is deliberately not read.

## Steps

1. Build the Slumbot API client; play one hand end to end (M0).
2. Build the translation layer with its own test suite.
3. Verify chip accounting against the engine, which the audit found sound — a mismatch means
   translation is wrong, not the engine.
4. Pilot a few hundred hands to shake out protocol bugs. Do not read the result.
5. Run 10,000+ hands with duplicate seating; write the JSON (M1).
6. Iterate on abstraction granularity, iteration count and translation, re-measuring at each
   change, toward M2–M4.

## Reporting

Report a loss as a loss. A CFR agent over a coarse abstraction losing to Slumbot by a
measured, bounded margin is a legitimate result and the first honest external one this project
has. The reason this file exists at all is that `CODEBASE_AUDIT.md` established what a
flattering self-computed number is worth.

## References

- Slumbot API reference implementation: https://github.com/Gongsta/Poker-AI/blob/main/slumbot/slumbot_api.py
- Lisý & Bowling (2017), *Equilibrium Approximation Quality of Current No-Limit Poker Bots*:
  https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf
