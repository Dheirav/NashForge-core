# NashForge — training plan: evolutionary search and PPO, heads-up first

**Status:** Phases 0–2 complete. Written 15 August 2026, last updated the same day.

The goal is a comparison: how close do evolutionary search and policy gradient get to a
solver-derived strategy, given the same budget on the same engine? Heads-up first, six-max
after, so that both methods can be measured against the CFR agent.

---

## Why this is worth doing now

Every previous attempt in this repository measured agents against **other agents bred on a
broken metric** — self-referential, which is how a reported best of 5213 BB/100 survived 89
training runs. [`CODEBASE_AUDIT.md`](../CODEBASE_AUDIT.md) has the details.

What exists now that did not: a **CFR agent as an independent yardstick**. It was validated
against Kuhn poker's analytic value of −1/18 and against exact exploitability on Leduc, and it
shares no lineage with anything a GA or PPO run will produce. That turns this from another
round of tournaments into a comparison with a reference point.

Heads-up first is not just variance reduction. It is the only table size where the CFR agent
can serve as the benchmark at all — the solver work is heads-up throughout.

---

## The observation: audited, 15 August

Before planning training, the 17-feature observation was measured rather than assumed.
1,500 sampled decision states per table size, true equity by Monte Carlo at 400 samples,
reproduced by `scripts/audit_observation.py`.

### It carries six dimensions in seventeen slots

| | heads-up | 6-handed |
|---|---|---|
| components explaining 90% of variance | **6** | **6** |
| live slots | 16 of 17 | 17 of 17 |

`players_in_hand` has a standard deviation of **exactly 0.0000** heads-up — it is always 2.
Two pairs are perfect duplicates: `players_behind`/`opponent_all_in` at r = −1.000 (heads-up
only) and `stack_normalized`/`commitment` at r = −1.000 (both sizes). Six-handed,
`players_behind`/`players_in_hand` sit at r = +0.979.

### All card information is in one slot

Predicting a state's true equity:

| | heads-up | 6-handed |
|---|---|---|
| from all 17 features | 0.421 | 0.481 |
| from `hand_strength` alone | 0.238 | 0.219 |
| **from everything except `hand_strength`** | **0.013** | **0.024** |

Remove that one feature and the other sixteen know essentially nothing about the hand.
Postflop the concentration is starker: all features reach 0.475, `hand_strength` alone
reaches 0.423.

### And that slot is lossy in a way training cannot fix

Grouping postflop states by near-identical `hand_strength` — states the agent physically
cannot tell apart — and measuring the true equity within each group:

| `hand_strength` | n | equity range | std |
|---|---|---|---|
| 0.73 | 38 | **0.907** | 0.173 |
| 0.58 | 43 | 0.890 | 0.253 |
| 0.60 | 35 | 0.861 | 0.231 |

The worst group spans 0.907 of equity: two hands the agent receives as an identical input can
be a 5% underdog and a 95% favourite. This is the decisive statistic, not the R² figures
above — those are linear fits and understate what a network could extract, whereas identical
inputs are an information limit no model escapes.

The cause is known and not accidental. Postflop, `hand_strength` is
`made_hand_strength(hole, board)` — scoring only the hand made so far. It is the same signal
the CFR work measured as **provably blind to draws**: on a two-heart board, the nut flush draw
and the same overcards without it are the same number. The
[abstraction crossover](abstraction-crossover.html) found that blindness costs real chips above
roughly ten minutes of training. The network observation has been carrying it the whole time.

### The fix is cheap, and was measured

Four binary features — flush draw, open-ended straight draw, paired board, three-to-a-flush —
computed from suit counts and rank gaps, no simulation:

| | before | after |
|---|---|---|
| R² overall | 0.421 | **0.503** |
| **R² postflop** | 0.475 | **0.637** |

A 34% relative gain postflop. The within-group spread falls only 0.168 → 0.158, so draws
explain part of the blindness rather than all of it — kicker quality and board interaction
remain invisible. That is a reason to expect more from a richer card representation later, not
a reason to skip the cheap win now.

---

## What to do about the features

**Remove two slots. No information is lost.**

- `commitment` — r = −1.000 with `stack_normalized` in both table sizes; algebraically the
  same number, since stack + contributed = starting stack.
- `players_in_hand` — constant heads-up, and 97.9% explained by `players_behind` six-handed.

**Keep `players_behind` and `opponent_all_in` both**, despite being perfect duplicates
heads-up. They are independent six-handed, and one shared layout across table sizes is worth
more than one wasted slot — the cost is a few weights, and it keeps the two configurations
comparable.

**Add the four draw and texture features.** 17 → 19 slots that carry more than the original
seventeen did.

**Do not add Monte Carlo equity per decision.** It costs ~10.6 ms per call at 400 samples,
against ~8 µs for made-hand strength. The CFR work measured the same trade-off from the other
side: equity bucketing costs 44.05 ms per solver iteration against 8.11 ms, a factor of 5.4.
Per-decision equity in a training loop making millions of decisions is not affordable, which
is exactly why the draw features matter — they recover much of what equity knows for nothing.

**Optional, unmeasured:** an "equity if I improve" scalar, or a two-card lookahead. Worth
testing after the cheap features land, not before.

This invalidates every saved genome. All of them came from the invalidated pipeline, so
nothing of value is lost.

---

## Decisions taken, 15 August

| | Decision | Why |
|---|---|---|
| Feature change | **Approved** — 17 → 19 | Measured; invalidates saved genomes, all of which came from the invalidated pipeline |
| Layout | **One shared layout** for both table sizes | Two layouts re-open the exact crack the audit found bugs in — `FeatureCache` and `get_state_vector` had drifted apart. Costs two dead slots heads-up |
| Budget axis | **Hands played**, not wall-clock | GA is CPU-parallel and PPO is not; wall-clock would compare hardware as much as algorithms. Wall-clock is still recorded alongside |
| PPO device | **CPU**, pending a profile | The network is a small MLP over 19 inputs; the Python engine is almost certainly the bottleneck. GPU adds a second memory pool on a box that has died twice from memory pressure |
| PPO self-play | Snapshot every ~10 updates, pool of last 5–10 | Sample from the pool most of the time, with some probability of facing the current policy |

## Progress

Updated as work lands. "Blocked" means waiting on a decision, and the decision is named.

| | Item | State |
|---|---|---|
| 0a | Tests for `rl/` | **done** — 9 tests, `tests/test_rl.py` |
| 0b | Duplicate play | **done** — `cfr/play.py`, ~14% noise reduction |
| 0c | Benchmark harness | **done** — `evaluation/`, 9 tests |
| 1 | Rebuild the observation, 17 → 19 | **done** — effective dimensions 6 → 9 |
| 2 | Evolutionary search, heads-up | **done** — learned to exploit randomness, nothing transferred |
| 3 | PPO with self-play, heads-up | **done** — closes the gap to the CFR agent; flat after 2M hands |
| 4 | The comparison | **done** — one panel, one budget axis; PPO reaches break-even in ~1 h, evolution never does |
| 5 | Six-max | blocked on 4; also needs the `play_match` stack-drift fix below |

## Phases

### Phase 0 — make the measurement trustworthy

The audit's Step 1, never carried out for networks.

- **Tests for `rl/`.** 1,917 lines, currently **zero coverage**. Minimum: chip conservation
  through the environment, observation shape and bounds, that the policy never emits an
  illegal action, that one PPO update runs without producing NaN. Training against an untested
  environment is how the previous round produced numbers nobody could trust.
- **Duplicate play.** *Done, 15 August.* `alternate_seats` swapped the policies but drew fresh
  cards each time, so position cancelled and card luck did not. `_play_one` now records each
  chance outcome and the replay reuses it, so both policies hold the same cards against the
  same opposing cards. On by default; `duplicate=False` reproduces the old behaviour.

  **It buys about 14%, not the large factor the audit implied.** Per-hand outcomes swing by a
  full ±200 stack depending on whether an all-in happened, and that is driven by the action
  sampling, which sharing cards cannot cancel. It is free and strictly better, so it stays on
  — but hand count remains the real variance lever, and coupling the action stream is the next
  option if more is ever needed.

  Guarded by two tests: that the noise drops, averaged over seeds because a single run's
  standard error is itself noisy; and that the *expectation* does not move, since a duplicate
  run that drifts off a symmetric matchup's true zero would silently corrupt every comparison
  built on the harness.
- **One benchmark harness.** Any agent against a fixed panel: random, always-call, and the
  **CFR agent**. Error bars on every figure, seats duplicated, hand counts sized from the
  measured noise floor rather than convention.

### Phase 1 — rebuild the observation — DONE, 15 August

17 → 19: `commitment` and `players_in_hand` removed, four draw and texture features added,
`get_state_features` deleted. Re-running `scripts/audit_observation.py` confirms it did what
this document claimed:

| | before | after |
|---|---|---|
| effective dimensions | 6 of 17 | **9 of 19** |
| dead slots (std ≈ 0) | 1 | **0** |
| perfect duplicate pairs | 2 | 1 (kept for six-max) |
| R² against true equity | 0.421 | **0.499** |
| R² postflop | 0.475 | **0.633** |
| within-group equity std | 0.175 | **0.086** |

The last row is the one that matters, and measuring it required correcting the audit. It had
grouped states by `hand_strength` alone — which *was* the entire card signature and no longer
is. Grouped by all five card features, states the agent genuinely cannot distinguish now
differ in true equity by half as much. Grouping on strength alone would have understated the
gain while appearing to measure the same thing.

**The change needed nine coordinated edits, not the four predicted.** The Phase 0a tripwire
caught `PokerEnv.OBS_SIZE` and `PPOConfig.obs_size`; the failing suite caught three more that
nothing pointed at: `FeatureCache.__slots__`, whose declared attribute list rejected the two
new cache fields on assignment; `training/config.py`'s `input_size`, which was still building
networks with 17 inputs; and a hardcoded `GENOME_SIZE = 3430` in the tests. That constant is
now derived from the network config, since a written-down width is exactly what disagreed with
reality here.

### Phase 2 — evolutionary search, heads-up — DONE, 15 August

Fifty generations, population 30, **24,000 hands per genome** — a budget set from the measured
noise floor rather than convention. Re-scoring the same population under different hands moves
a genome by 136 BB/100 at 2,000 hands, 66 at 8,000 and 31 at 24,000.

**The endpoint result.** Final genome against an untrained one from the same initial
distribution, both against the same panel at 40,000 hands (±14 BB/100):

| opponent | untrained | after 50 generations | difference | |
|---|---|---|---|---|
| random | −13.4 ± 14 | **+192.7 ± 15** | **+206.1 ± 39** | improved |
| always-call | −8.1 ± 14 | +0.5 ± 4 | +8.6 ± 28 | no change |
| CFR | −403.9 ± 13 | −370.1 ± 14 | +33.8 ± 37 | no change |

**Evolutionary search learned to exploit randomness, not to play poker.** It punishes an
opponent that folds and raises at random by 206 BB/100, gains nothing against a station that
never folds, and is still beaten by the solver by 370 BB/100 after fifty generations. That is
a sharper finding than the audit's prediction that the method would simply plateau — it did
learn, and what it learned did not transfer.

Only the panel makes that visible. Self-play fitness sat near zero throughout, as the zero-sum
property requires, and says nothing about whether anything was learned.

### Phase 2's three false results, and what caught each

Every one of them looked like a finding, and each was caught by a check that already existed.
Recorded because the pattern is the argument for the framework, more than any single number in
it.

**The per-generation panel curve.** Taken over 3,000 hands, ±57 BB/100, and read during the run
as rising from +111 to +214. Across eleven readings it scattered with a standard deviation of
56 — indistinguishable from a constant. A measurement that coarse does not return "no result";
it returns a plausible shape. The endpoint test at 40,000 hands found the real effect the
curve could never have resolved.

**"The evolved agent beats a game-theoretic solver."** The CFR row first read +60.8, alongside
a 74.3% lookup miss rate. The miss-rate counter exists precisely so a benchmark that has
quietly become a second random opponent shows up as a number. In the previous regime this
would have been published.

**"Half the abstraction is unreachable."** The explanation offered for that miss rate — that
the solver covers 47% of its information sets at 4,000 iterations and 51% at 59,050, so a
network opponent lands off its trajectory and more training cannot help — was itself wrong,
and was committed. Nothing was missing: every lookup found its key. The solver stores one
probability per *legal action at a node*, so arrays are mostly length 2 or 5, and the harness
discarded anything that was not length 6. That was 77.5% of successful lookups. The tell was
available and misread — a 100% hit rate against a random opponent should have ruled out
"structurally unreachable" immediately.

### Phase 3 — PPO with genuine self-play, heads-up — DONE, 20 August

Snapshot pooling was the prerequisite and was built first: PPO trained against `RandomOpponent`
or against a Hall of Fame pool that was empty, and would have been tainted if it were not, since
every genome in it was bred on the withdrawn metric. Self-play here means the policy against its
own periodic snapshots.

Three seeds, 8M hands each, about 4.7 hours apiece, with rungs kept at 500k, 2M and 8M so
Phase 4 gets a budget axis rather than a single endpoint. The rungs are points on one run, not
three runs — the 2M policy is the 500k policy trained further, which is what makes them
comparable.

**The endpoint result.** Each rung against an untrained policy from the same initialisation,
both against the same panel, both at 40,000 hands, all three seeds
(`scripts/endpoint_test_ppo.py`, `results/ppo/phase3_endpoint.json`). Gains are the mean across
seeds; the spread is the range across them.

| trained to | vs random | vs always-call | vs CFR |
|---|---|---|---|
| 500,000 | +215.9 (spread 216.9) | +261.9 (spread 397.8) | **+324.3** (spread 79.2) |
| 2,000,000 | +233.3 (spread 73.3) | +276.9 (spread 369.7) | **+394.2** (spread 32.5) |
| 8,000,000 | +147.1 (spread 135.3) | +451.3 (spread 636.7) | **+373.1** (spread 48.4) |

All 27 rows — three seeds, three rungs, three opponents — separated from zero in the improving
direction. The CFR lookup miss rate was 0.0% across all nine matchups.

**What it learned transferred, and that is the contrast with Phase 2.** Against the solver the
untrained networks scored −399.1, −385.9 and −366.3 BB/100; after 8M hands the same three scored
−34.8, +35.4 and −32.7, a mean of −10.7. PPO closes essentially the whole gap to the CFR agent.
Evolutionary search, given the same panel and the same bar, moved −403.9 → −370.1 and was scored
*no change*. Both halves of the audit's prediction now have numbers: the evolutionary method
plateaus, and policy gradient produces a strong player.

**The ladder is flat after 2M.** +394.2 at 2M against +373.1 at 8M, on spreads of 33 and 48 — the
last six million hands of each run, three quarters of the wall-clock, bought nothing that can be
measured. Phase 4 should spend its budget accordingly rather than assuming more is more.

**Only the CFR column is quotable.** Its seed spreads are 33–79 BB/100; the two baselines run
73–637. Against always-call at 8M the seeds scored +426.1, +145.5 and +782.2 — the direction is
certain and the magnitude is not resolved at all. The opponent from outside the lineage is the
stable yardstick and the hand-written baselines are not, which is worth carrying into Phase 5,
where the CFR agent cannot serve.

**What this is not.** It is break-even against *this* CFR agent — six buckets, one raise per
street — not against a strong solver, and not an exploitability figure. Item 1 closed with no
usable bound, so no-limit still has none.

### Phase 4 — the comparison — DONE, 20 August

All three families against the same panel, at the same 40,000 hands, on one budget axis
(`scripts/phase4_comparison.py`, `results/comparison/phase4.json`).

**The budget axis needed no new training.** The families laddered along different axes — CFR
along seconds, evolution along generations, PPO along hands — but all three had recorded
wall-clock all along: the CFR budgets are seconds by construction, `phase2_history.json` keeps
`seconds` per generation, and each PPO history keeps `seconds` per interval. The axis was
arithmetic over files that already existed.

| family | hands | vs random | vs always-call | vs CFR |
|---|---|---|---|---|
| CFR (the solver) | — | **+383.6** | **+719.6** | — |
| evolution, 50 generations † | 36,000,000 | +192.7 | +0.5 | −370.1 |
| PPO | 500,000 | +191.2 | +372.6 | +20.1 |
| PPO | 2,000,000 | +137.2 | +373.2 | +12.5 |
| PPO | 8,000,000 | +226.8 | +329.1 | **+36.4** |

† Measured under the pre-3-September raise sizing, where a pot-fraction raise was about 20%
smaller. Evolution was not retrained because its fitness cannot be selected on — see above — so
this row is not on the same footing as the other two. It is left rather than dropped because the
endpoint test behind it is sound; it is marked rather than silently mixed.

BB/100, big blind 2. The solver has no row against itself: a strategy against a copy of itself
is zero by symmetry, and printing that zero would put a structural identity in a column of
measurements.

**Evolutionary search spent 36,000,000 hands — seventy-two times PPO's smallest rung — and is
still 370 BB/100 behind the solver**, on a gain over its own untrained network of +33.8 ± 37,
which is no change. That comparison is separated many times over and is the firm result here.

**PPO's own position against the solver, at six seeds.** Three could not settle it, so seeds 3, 4
and 5 were trained to 8M hands and measured on the same instrument
(`results/ppo/phase3_endpoint_6seed.json`).

| rung | mean | sd | stderr | t | verdict |
|---|---|---|---|---|---|
| 500,000 | +20.1 | 71.6 | 29.2 | +0.69 | not separated from parity |
| 2,000,000 | +12.5 | 60.2 | 24.6 | +0.51 | not separated from parity |
| 8,000,000 | **+36.4** | 32.5 | 13.3 | **+2.74** | **separated** |

**PPO is level with the solver at 500,000 and 2,000,000 hands, and beats it at 8,000,000 by
+36.4 ± 13.**

Three seeds had read +60.2 at 500,000 hands and −1.7 at 2,000,000, which suggested an advantage
that appeared early and then dipped. Six seeds say the opposite: the 500,000 mean falls to +20.1
once seeds giving −0.2 and −80.7 are included, and the only rung separated from parity is the
largest. **More training does help; it took six seeds to see it, and the three-seed reading had
the shape backwards.** An earlier version of this section claimed PPO beat the solver on 500,000
hands. That was a mean quoted without checking it against its own spread — the specific failure
this document records four other instances of — it was withdrawn before these seeds were run, and
they confirm the withdrawal was right.

**The spreads answer the other question.** Standard deviation against the solver falls with
training: **71.6 → 60.2 → 32.5** at 500k, 2M and 8M, and the seed ranges fall likewise, 226 → 182
→ 102. Self-play converges to a consistent outcome given enough hands, and the wide spreads seen
after the raise-sizing fix are a property of **under-trained policies** rather than of the fix.
That is as close to an answer as is now available, and it points away from the earlier
speculation that the corrected sizing made self-play noisier.

Whether the fix changed the spreads at all remains formally unanswerable — 32.5 against 181.8 on
three samples each is close to meaningless, and the pre-fix checkpoints were overwritten, so no
further evidence can be gathered on that side. Both families spend the same currency, so this is a like-for-like
budget statement rather than a comparison of endpoints.

**The axis is hands, and it used to be wall-clock.** Wall-clock was chosen because it was the only
unit all three families recorded, and it turned out to encode whatever else the machine was doing:
the identical 8M-hand run read 4.81 h on a quiet box and 10.42 h when three seeds shared it with
another project's tuner. An axis that moves by a factor of two under load is measuring the
machine. **The earlier phrasing of this result — "reaches break-even in about an hour" — was
therefore substantially a statement about CPU contention, and is withdrawn.** Hands played is
exact, machine-independent, and what both learned families actually spend. The solver is off the
axis entirely: it traverses a tree rather than playing hands, and here it is the opponent rather
than a competitor.

**PPO's rows here are the six-seed means**, and only the 8,000,000 rung is separated from parity
(+36.4 ± 13, t = 2.74). The 500,000 and 2,000,000 rungs are not: +20.1 and +12.5 against standard
errors of 29.2 and 24.6. The table is quoted rather than ranked for that reason — the three PPO
rows cannot be ordered against each other on this evidence, only the largest can be told from the
solver.

An earlier version of this table carried the three-seed figures, +60.2 / −1.7 / +59.1, which put
the advantage at the smallest rung and a dip in the middle. Six seeds reverse that shape. The
numbers above supersede them; `results/comparison/phase4.json` and `phase4_refit.json` are the
earlier runs and should not be quoted.

**~~The table is non-transitive, and that is the most interesting thing in it.~~ Withdrawn 8
September — see "The intransitivity, withdrawn" below. Every edge in the paragraph that follows
was measured against the 4,000-iteration panel and none of them survived its replacement. Kept
because the reasoning was sound on the numbers available, and because what replaced it is
weaker.** PPO at 2M draws
level with the CFR agent head to head, yet the solver takes far more off both baselines —
+377.2 against random to PPO's +221.5, and +722.9 against always-call to PPO's +293.5. Two
agents level against each other extract very different amounts from the same weak opponents,
so strength here is not a scalar and any single-number ranking of the three families would be
a fiction. It also runs against the intuition that the near-equilibrium strategy should be the
less exploitative one. The mechanism is not established; it is recorded as an open question
rather than explained away.

**Two hazards were found while building it, both of which produce a plausible-looking wrong
chart.** `results/cfr/strength_signals.json` is in chips/hand while the endpoint tests are in
BB/100 — a factor of fifty at big blind 2, which plotted unconverted shows the solver as fifty
times weaker than PPO rather than stronger. And evolution's CFR row exists twice: the +60.8
in `phase2_endpoint.log` was taken beside a 74.3% lookup miss rate, and only the −370.1 in
`phase2_cfr_row.log`, at 0.0%, is usable. The comparison script normalises units once at the
boundary and refuses to run if the withdrawn row ever loses the miss-rate marker that
identifies it.

### The intransitivity, measured and explained — 20 August (withdrawn 8 September)

Phase 4's table did not rank, and three explanations were listed. Two were checked and failed:
the two families' rows are on one instrument, bit-for-bit
(`scripts/diagnostics/check_instrument.py`), and lifting the one-raise-per-street cap *widened*
the gap rather than closing it (`scripts/diagnostics/check_raise_cap.py`). The third was that
the intransitivity is genuine, and it now has a measurement
(`scripts/diagnostics/check_intransitivity.py`).

**The edge nobody had measured.** Phase 4's tournament graph had a hole in it: both learned
families were scored against the solver and the two baselines, never against each other.

| edge | BB/100 to the first named |
|---|---|
| CFR vs PPO | +10.4 — level |
| CFR vs evolution | +370.1 |
| **PPO vs evolution** | **+23.9** — per seed −12.7, +68.2, +16.3, spread 80.8 |

If strength were a scalar, PPO being level with the solver and the solver beating the evolved
genome by 370 would require PPO to beat it by about 360. It beats it by 23.9, which against a
spread of 80.8 is not separated from zero — one seed lost. **PPO is level with the solver and
also level with the genome the solver crushes**, and both cannot be true of a single ordering.
The edge was no part of the observation that raised the question, which is what makes it
evidence rather than a restatement of it.

**And the mechanism.** Counting what each agent actually does against a station that never
folds, over about 250,000 decisions each:

| | fold | check/call | raise ½ | raise pot | raise 2× | all-in |
|---|---|---|---|---|---|---|
| CFR solver | 1.5% | 24.2% | 20.9% | 21.0% | 17.3% | **15.0%** |
| PPO, 2M hands | 3.5% | 34.1% | 35.3% | 13.2% | 13.8% | **0.1%** |

The two raise at similar *rates* — 74.3% against 62.4% — so frequency is not the explanation.
**PPO has effectively eliminated the all-in from its strategy** and concentrates on the smallest
raise. Both agents face the identical cap, so this is a property of the policy rather than of
the rules.

That accounts for the whole table. Against an opponent who never folds, the value is in large
bets. PPO trained by self-play against snapshots of itself, and those snapshots fold; it never
faced an opponent worth jamming into, so it never learned to. Against the solver that costs it
nothing and the two are level. Against anything weak it leaves most of the value uncollected.

This is the audit's prediction that policy gradient would produce "a strong but exploitable bot",
made specific and measured: **self-play optimises for a peer, and produces a policy that cannot
punish a weak opponent.** It is also why no single-number ranking of these three families should
be quoted, in this report or anywhere downstream of it.

### The intransitivity, withdrawn — 8 September

Everything above this line is measured against the 4,000-iteration solver. When that panel was
replaced (see "The panel was a solver with two minutes of training"), both of the edges the argument rests on
moved, so the third was re-measured on the converged panel — three seeds, 40,000 hands each,
same script:

| edge | August, 4k panel | September, converged panel |
|---|---|---|
| CFR vs PPO (2M) | +10.4 | +73.5 |
| CFR vs evolution | +370.1 | +200.9 |
| PPO vs evolution | +23.9 (spread 80.8) | **+96.5** (per seed +75.2, +273.3, −58.9; spread 332.2) |

Transitivity predicted +359.7 in August against a measured +23.9 — a shortfall of 336, against a
spread of 81. It predicts **+127.4** now against a measured **+96.5**, with a standard error of
±96.5 across the three seeds. The shortfall is 31, a quarter of its own error bar. **There is no
intransitivity in the current table**, and the two learned families order exactly where the
solver's edges over them say they should.

The mechanism goes with it. The August action counts were taken against a solver that went all-in
on 15.0% of its decisions; the converged solver jams on **1.6%**:

| | score vs always-call | fold | check/call | raise ½ | raise pot | raise 2× | all-in |
|---|---|---|---|---|---|---|---|
| CFR solver, 250k | +603.4 | 0.3% | 45.4% | 29.4% | 16.0% | 7.3% | 1.6% |
| PPO, 2M hands | +397.3 | 2.7% | 42.4% | 23.8% | 19.2% | 11.7% | 0.2% |

~300,000 decisions each. The two raise at 54.3% and 54.9%, and PPO's average raise is the larger
of the two — it prefers pot and 2× where the solver prefers a half-pot bet — yet it collects 206
BB/100 less. Neither frequency nor sizing explains the gap; what is left is *which* spots are bet,
and this instrument counts actions rather than situations. PPO's 2.7% fold rate against an
opponent who never folds is a real defect of self-play against snapshots that do fold, but it is
worth a few BB/100, not two hundred.

**What replaces the August claim.** Not "strength is not a scalar" — the current table is
consistent with a scalar. The surviving, weaker statement is that the *baselines* rank badly: the
250k solver is decisively stronger than the 4k one and still scores worse against random
(+280.1 to +377.2). Rank agents by head-to-head, not by a baseline column.

**What this cost.** The August investigation was three diagnostics and a full explanation built on
two edges from a panel nobody had checked. Two of the three candidate explanations it ruled out
stayed ruled out; the one it accepted was an artefact of the fourth thing it did not think to
question. The lesson is filed with the other seven in `docs/RESULTS_SHEET.md`.

### The instrument defect found on 20 August, and how

Phases 3 and 4 were measured, then re-measured, because the panel's results depended on the
order they were taken in. Recording it here with the other three, since it is the fourth time
this project has had to distrust a number and the first one that no result would have revealed.

**What was wrong.** Three layers, all the same class:

1. `build_panel()` handed the random opponent and the solver **one** generator, so the solver's
   stream depended on how much the random matchup had drawn.
2. The policy samples its actions from **torch's global generator**, and nothing reseeded it
   between matchups. `torch.manual_seed` appeared once in the whole measurement path, inside
   `PPOTrainer.__init__`.
3. Callers build the panel **once** and reuse it, so each opponent's state carried across all
   twelve columns of the endpoint test.

Together: the same matchup read −28.9, −29.5 and −3.9 BB/100 depending only on what had been
measured before it. Every one of those is a valid sample. None was reproducible alone.

**How it was found, which is the part worth keeping.** Not by a result looking wrong — the
published numbers were fine. It came out of checking whether the Phase 4 non-transitivity was
an artefact: the instrument check passed, but a difference spotted while writing it led to
layer 1, and **the fix for layer 1 failed its own order-independence test**, which is what
exposed layers 2 and 3. A fix that had been committed without that test would have closed the
smallest of the three and reported the problem solved.

**The fix.** The panel hands out factories, and `panel_scores` builds a fresh opponent and
reseeds torch — from the evaluation seed and the opponent's *name*, so reordering cannot move a
number — for every matchup. `scripts/preflight_ppo.py` carried the same defect and got the same
fix. A matchup now returns the same figure measured alone, inside the full panel, with the panel
reversed, or on a reused panel object; all four differed before.

**What re-measuring changed: almost nothing, and that is the result.** Every Phase 3 figure moved
less than its own seed spread — at most 13.0 BB/100 against a spread of 216.9, and in the CFR
column no more than 3.6. The seed spreads themselves *tightened* (134/40/62 → 79/33/48), which is
what removing an uncontrolled source of variation should do. The conclusions stand and are now
reproducible.

`scripts/diagnostics/check_instrument.py` is kept as the check that would catch this class
returning.

### Evolutionary fitness cannot be selected on — 3 September

Evolution was to be retrained through the corrected raise sizing. `preflight_training.py` refused,
and it was right to.

**The check.** Score the same population twice under different hands and correlate the two
rankings. Below about 0.5 the ranking is mostly luck and selection optimises noise. It reported
**r = −0.07** at the preflight's default 120 hands per matchup.

**It is not my change.** Run against the pre-fix code the same check reads **r = −0.28** at
**487 BB/100** of noise per genome, against **−0.07** at **273** after. The corrected sizing
roughly halved the noise. This also kills the speculation that the fix is what widened PPO's seed
spreads: on this measurement it does the opposite.

**It is not under-sampling either.** The preflight scores at 120 hands per matchup where the real
run uses 6,000, so the first explanation was that the gate simply measures at a budget the run
never uses. Re-run at the real 6,000:

| hands/matchup | correlation | noise per genome |
|---|---|---|
| 120 | −0.07 | 273 BB/100 |
| **6,000** | **+0.12** | **59 BB/100** |

Fifty times the sampling cut the noise by a factor of five, exactly as the project's own scaling
predicts — and moved repeatability from −0.07 to +0.12. Nowhere near 0.5.

**And it is not card luck between genomes.** `training/evolution.py` already carries the machinery
for scoring a whole population against one fixed set of hands — `generate_eval_hand_seeds`,
`evaluate_population_fixed_hands` — but `train_generation` uses it only for a monitoring readout;
the fitness that drives selection comes from `evaluate_population`, which deals fresh random hands
per genome. Pointing selection at the shared-card path was the obvious remaining fix and costs
nothing. Scored under two different shared samples at 6,000 hands:

| | ranking agreement |
|---|---|
| shared cards across the population | pearson +0.14, **spearman +0.01** |
| independent random hands | +0.12 |

**No improvement at all.** Sampling is working — noise per genome fell from 93 BB/100 at 800 hands
to 33 at 6,000 — and the rankings still do not agree.

**The finding.** There is no signal being drowned out. These genomes are genuinely
near-indistinguishable: the real fitness differences between them are smaller than poker's
variance, whatever cards are dealt and however many. **Evolutionary fitness cannot be selected on
at any affordable budget**, and no sampling scheme fixes that.

This sharpens Phase 2 rather than contradicting it. The endpoint result stands — the evolved
genome beats a random opponent by +206 BB/100, measured at 40,000 hands by a sound instrument.
What cannot be claimed is that *selection* produced it: fifty generations ran on a signal with
r ≈ 0.12, which is drift. "Evolutionary search plateaus" was the prediction; "its fitness was
never strong enough to select on" is the measurement, and it is a better answer.

**Consequence for Phase 4.** Evolution is not retrained, so its row remains measured under the old
raise convention while CFR and PPO are on the corrected one. The table below marks it rather than
presenting the three families as being on one footing.

### Re-measured after the raise-sizing fix — 3 September

`training/fitness.py` was corrected to size a pot-fraction raise off the pot *after* the call,
matching `games/nolimit.py` and the standard convention. PPO trains through that function, so all
three seeds were retrained at 8M hands and the endpoint test re-run under the corrected
instrument (`results/ppo/phase3_endpoint_refit.json`).

**The result survives.** Gain over an untrained policy from the same initialisation, against the
CFR agent:

| rung | before the fix | after | move |
|---|---|---|---|
| 500,000 | +324.3 | +455.8 | +131.5 |
| **2,000,000** | **+394.2** | **+393.9** | **−0.4** |
| 8,000,000 | +373.1 | +454.7 | +81.6 |

Only one of nine cells moved beyond its own seed spread. The 2M rung — the one Phase 4 quotes —
shifted by 0.4 BB/100. So the divergence was real and was **not** what the Phase 3 finding rested
on, which is stronger evidence for that finding than the original measurement was: it now holds
with the two betting implementations in agreement.

**But the seed spreads widened five- to six-fold.**

| | before | after |
|---|---|---|
| 2M vs CFR | 32.5 | **181.8** |
| 8M vs CFR | 48.4 | 63.8 |
| 2M vs random | 73.3 | 192.1 |

Every apparent move in the table except one sits *inside* the new spread, so most of what looks
like change is seeds disagreeing more rather than the fix doing something. The cause is not
established. Two candidates: the corrected sizing genuinely makes self-play noisier, since bigger
raises commit more chips per decision, or these three seeds simply diverged more than the last
three. Distinguishing them needs more seeds.

**The practical consequence is a loss of confidence rather than a change of conclusion.** With a
spread of 182, the 2M rung's +393.9 is much softer than the old +394.2 ± 33 implied. The number
barely moved; what can be claimed about it did.

**The pre-fix checkpoints no longer exist.** `scripts/train_ppo.py` hardcodes its output directory,
so the retrain wrote over the August checkpoints. The published pre-fix numbers remain as records
in `results/ppo/phase3_endpoint.json` but the agents behind them are gone, so those figures cannot
be reproduced and no cleaner before/after comparison than the table above is available.

### The training lever, and where it runs out — 2 September

`results/cfr/nolimit_strategy.pkl` shipped with 4,000 iterations, which at the crossover's
measured 32 iterations/second is about **two minutes of training**. Retraining the same
abstraction — same six buckets, same one-raise cap, same seed — at 150,000 and 250,000
iterations, and ranking the three head to head on `evaluation.benchmark` at 40,000 hands:

| step | head-to-head gain |
|---|---|
| 4,000 → 150,000 | **+185.0 ± 13 BB/100** |
| 150,000 → 250,000 | **+12.3 ± 7 BB/100** |

**The lever is spent.** A 37× increase in training bought 185 BB/100; a further 1.67× bought
12.3 — separated from zero and not worth the three hours. Information sets reached went 25,154 at
150,000 and 25,154 at 250,000: coverage saturated, and the extra iterations refined a fixed set
of nodes rather than finding new ones. Roughly half the abstraction's 49,200 information sets are
unreachable in practice under this betting tree.

So **the abstraction is now the binding constraint, not training** — which is the answer to a
question that had never been asked, and the cheap one to ask first.

It transferred externally too, at a discount. Against Slumbot the 4,000-iteration solver scored
−1750 ± 524 mbb/hand and the 150,000-iteration one −987 ± 374, an improvement of +764 ± 644.
Internally the same upgrade is +185 ± 13 BB/100; externally it is +76 ± 64. **Internal strength
overstated the external gain by about 2.4×**, which is worth carrying: beating your own previous
agent is evidence about a third party, not a measurement of one.

#### The baselines rank these three exactly backwards

| | vs random | vs always-call | true strength |
|---|---|---|---|
| 4,000 | **+377.2** | +722.9 | weakest |
| 150,000 | +280.1 | **+827.7** | middle |
| 250,000 | +264.8 | +633.5 | **strongest** |

Against a random opponent the ordering is monotonically **inverted** — 377 → 280 → 265 as the
solver genuinely improves. Against the calling station there is no ordering at all.

This is the third independent sighting of the effect and the cleanest. As CFR converges toward
equilibrium it becomes *less exploitative* of weak opponents while becoming *stronger* against a
peer: equilibrium play is unexploitable, not maximally exploitative. The solver's own action
counts show it directly — the 4,000-iteration solver moved all-in on **15.0%** of its decisions
against a calling station, the 250,000-iteration one on **1.6%**.

That same 15.0% was the baseline August measured PPO's 0.1% against, and read as a defect of
self-play. Most of the gap was the panel converging rather than PPO failing; see "The
intransitivity, withdrawn".

**The panel's `vs random` column is therefore actively misleading as a strength signal for CFR
agents.** Anyone tuning on it would tune backwards. Rank on the head-to-head or not at all.

### The panel was a solver with two minutes of training — corrected 8 September

Every figure this project has published about PPO and evolutionary search was measured against
`results/cfr/nolimit_strategy.pkl`, which shipped with **4,000 iterations**. A 250,000-iteration
solver trained on the corrected game beats it head to head by **+165.2 ± 13 BB/100**, so it was
promoted to the panel (the old one is kept as `nolimit_strategy_4k.pkl`) and both families were
re-measured against it.

| family | hands | vs random | vs always-call | vs CFR |
|---|---|---|---|---|
| CFR (the solver, 250k) | — | +245.9 | +603.4 | — |
| evolution, 50 generations | 36,000,000 | +202.7 | −0.2 | **−200.9** |
| PPO | 500,000 | +191.2 | +372.6 | **−79.7** |
| PPO | 2,000,000 | +137.2 | +373.2 | **−73.5** |
| PPO | 8,000,000 | +226.8 | +329.1 | **−72.0** |

**Both learned families lose.** Three earlier claims fall:

| claim | vs the 4k panel | vs the 250k panel |
|---|---|---|
| PPO beats the solver at 8M | +36.4 ± 13 | **−72.0**, t = −10.02 |
| More training helps | +20.1 → +12.5 → +36.4 | **flat**: −79.7 → −73.5 → −72.0 |
| Evolution learned nothing transferable | +33.8 ± 37 | **+50.0 ± 20, separated** |

**The two errors point in opposite directions, for one reason.** An under-trained solver is far
more *exploitative* than a converged one: the 4k agent took +383.6 off random and +719.6 off
always-call, where the 250k agent manages +245.9 and +603.4. That exploitation flattered PPO's
relative standing and simultaneously buried evolution's improvement beneath a larger loss. It is
the same equilibrium-versus-exploitation trade recorded three times above, and the fourth
occasion on which changing the instrument — not gathering new data — overturned a finding.

**What survives.** PPO remains far ahead of evolutionary search, −72.0 against −200.9 on one
panel, and evolution still spent 36,000,000 hands to PPO's 500,000. That comparison is unaffected
by which solver holds the panel seat, because both families moved together.

**A correction to the fitness finding above.** "Fifty generations were largely drift" was too
strong. The ranking signal genuinely cannot be selected on — r = +0.12 at the real budget, and
shared cards do not help — but selection on it still produced +50.0 ± 20 BB/100, visible once the
opponent stopped drowning it out. A weak ranking and a real improvement are compatible.

**One thing this cost.** The all-in fix below prompted a 4h48m retrain on the expectation that it
would close the evaluation-path discrepancy. It did not; the calling-station bug did. The retrain
produced the solver now on the panel, so it was not wasted, but the cheaper bug should have been
found first — both candidates were visible and the expensive one was chosen.

Also fixed here: `phase4_comparison.py` printed "which is no change" as a **hardcoded string**,
written when evolution's gain was +33.8 ± 37 and correctly not separated. Against the new panel
the gain is +50.0 ± 20 and the frozen verdict would have gone on denying it. It is computed now.

### The evaluation paths disagreed because the calling station was raising — resolved 8 September

`train_nolimit.py`'s evaluation and `evaluation.benchmark` gave opposite verdicts on the same two
strategies. Two bugs were found chasing it; only the second was the cause.

**`cfr/play.py`'s `always_call_policy` was not a calling station.** It returns an array indexed by
*position in the legal-action list*, then wrote 1.0 at index `call_action = 1`. With nothing to
call, fold is dropped as dominated and the legal list is `[1, 2, 3, 4, 5]` — so index 1 is action
**2, raise half pot**. Verified directly: `legal [1,2,3,4,5] → picks action 2`. The baseline
raised every time checking was free, which is most of postflop.

`evaluation.benchmark`'s `always_call_agent` reads the mask by abstract action and was always
correct. So **every "vs always call" figure `train_nolimit.py` printed was measured against a
semi-aggressive opponent wearing a passive one's name** — including the +581.5 / +348.1 pair that
started the investigation.

Fixed by locating check/call's position in the legal list. The two paths now agree: the same
solver scores **+605.0** through benchmark and **+609.9** through `play_hands`, a gap of 4.8
BB/100, inside noise at 20,000 hands. It was 232.4 before. **The trainer's evaluation is
trustworthy again**, and the warning against comparing across the two paths is lifted.

**The all-in bug was real but was not this.** The traversal game had been producing decision nodes
after an all-in was called, asking a check/call from players holding nothing — `11/51/`,
`11/51/1` — where the engine ends the hand and goes to showdown. Those filler actions extend
`history`, which is the information-set key, so the same hand keyed differently in the two games
and 19.7% of sampled nodes had a legal-action list the engine's reconstruction could not match.

`current_player` now returns CHANCE while all players are all-in so the board runs out, and
`is_terminal` ends the hand once it is complete. The effect is measurable: **250,000 iterations
now reach 23,470 information sets against 25,154 before** — 1,684 fewer, the spurious nodes gone.

It did not close the evaluation gap, though. A solver was retrained for 4h48m on the expectation
that it would, and the gap widened rather than closing, which is what sent the search on to the
baseline. The retrain is not wasted — `nolimit_strategy_v2_250k.pkl` is the first solver trained
on a game that matches the engine — but the cheaper bug should have been found first.

### Superseded: the all-in divergence as first written — 8 September

The raise-sizing fix made engine and traversal betting agree to the chip, and
`tests/test_betting_equivalence.py` asserts it. The two evaluation paths still give **opposite
verdicts on the same two strategies**, measured at 20,000 hands against always-call after the fix:

| solver | `evaluation.benchmark` (engine) | `cfr.play.play_hands` (traversal) |
|---|---|---|
| 4,000 iterations | +729.6 | +497.2 |
| 150,000 iterations | +838.9 | +334.5 |
| | 150k **better** by +109.3 | 150k **worse** by −162.7 |

**The cause is what happens after an all-in.** In the engine, all-in followed by a call sends the
hand straight to showdown — measured: `over=True` after two actions, stacks `[0, 0]`. The
traversal game keeps producing decision nodes on every later street, asking for a check/call from
players who have no chips: `11/51/`, `11/51/1`, `11/51/11/`.

Those filler actions extend the history string, and **the history string is the information-set
key**. So the same hand produces different keys in the two games, and the solver is trained on
nodes the engine never visits. The solver takes all-in on about 15% of its decisions against a
calling station, so these lines are not rare.

`evaluation.benchmark._solver_actions` compounds it. It reconstructs the legal list from
`history.split("/")[-1]` — the current street only — and has no knowledge of stacks, so after an
all-in on an earlier street it reports raising as available again. Comparing its reconstruction
against the traversal game's own `legal_actions` over 15,640 sampled decision nodes: **3,088
disagree, 19.7%**, every example of the same shape:

    history='11/51/'   to_call=0   traversal=[1]   reconstructed=[1, 2, 3, 4, 5]

**Which side is right.** The engine is: real poker has no decisions left once both players are
all-in. The traversal game's extra nodes are an artefact of a traverser that advances streets
without checking whether anyone can still act.

**What follows.** `evaluation.benchmark` remains the path to trust, and the trainer's built-in
evaluation stays unfit for comparisons — but the reason is now known and is worse than
"unexplained": it is measuring a game with nodes that do not exist. More seriously, **the solver
is trained in that game too**, so a fraction of its table is fitted for situations it will never
face. That is a train/deploy mismatch of the same class as the raise sizing, and correcting it
means changing `games/nolimit.py` to terminate betting once all players are all-in, then
retraining every solver.

Not done here: it invalidates all three solvers and every CFR figure measured through them, so it
is a deliberate piece of work rather than a patch. Recorded with the measurement that found it.

### Two betting implementations, and a 20% divergence between them — found 2 September

Poker's betting is written twice here. `engine.PokerGame` is the audited one and is where every
published figure is measured. `games/nolimit.py` rebuilds it, deliberately and for a stated
reason: the engine mutates state in place and consumes a dealt deck, so a CFR traverser cannot
branch over an action and back up.

So the solver is **trained** in the traversal game and **scored** in the engine, and until now
nothing had checked that the two agree about chips. `tests/test_betting_equivalence.py` drives
identical abstract action sequences through both and compares pot and stacks.

**They agree exactly on every line that does not contain a pot-fraction raise** — fold, check,
call and all-in cost the same to the chip. **Pot-fraction raises diverge by about 20%:**

| | pot the raise is sized against |
|---|---|
| `games/nolimit.py:200` | `sum(contributions) + to_call` — after the call |
| `training/fitness.py:121` | `game.state.pot.total` — before the call |

```
raise-pot preflop, blinds 1/2, small blind acting
  traversal: call 1 -> pot 4 -> raise 4, total in 6, pot 8
  engine:    pot 3 -> raise 3, plus the call, total in 5, pot 7
```

The traversal game follows the standard convention — a pot-sized raise means calling first, then
betting the pot including your call.

**What it affects, and what it does not.** The abstraction crossover is untouched: `crossover.py`
and `head_to_head.py` reference the engine zero times, so both strategies were trained *and*
measured in the traversal game, under identical rules. Kuhn and Leduc are different games
entirely. PPO and evolutionary search train *through* `training/fitness.py`, so they learned the
engine's convention and are consistent with how they are measured. The Slumbot bridge sizes
raises after the call, matching the game its solver was trained in.

What is affected is narrow: **a CFR strategy trained in the traversal game and then measured in
the engine** — the CFR agent's rows as the panel opponent, Phase 4's `vs CFR` column, and the
viewer. Those numbers are real measurements; what is true is that the agent makes raises about
20% smaller than the ones it was fitted for, so it is a slightly *softer* benchmark than the
strategy it came from. Everything scored against it was scored under the same conditions, so the
comparisons between families hold.

**Why it is recorded rather than fixed.** Fixing `training/fitness.py` naively makes things
worse. `rl/poker_env.py:353` uses the same function for PPO's *training*, and evolution's too, so
changing the convention moves the mismatch off CFR — which retrains in three hours — and onto two
families that need seventeen. The only consistent fix retrains everything, and that is a
deliberate overnight job rather than a hotfix. **It should be fixed**; it is listed in
[`NEXT.md`](../NEXT.md) rather than left as folklore.

The test passes while the defect exists and fails once it is corrected, which is deliberate: it
characterises the divergence so it cannot be rediscovered by accident, and tells whoever unifies
the convention to delete it.

### Phase 5 — six-max

After heads-up is complete. Note that the CFR agent cannot serve as a benchmark there, so the
panel loses its yardstick and the comparison becomes relative again. Worth planning separately
rather than assuming the heads-up design transfers.

---

## What to expect, stated in advance

The audit predicts both outcomes, and recording them now is what makes them findings rather
than surprises later.

**Evolutionary search will plateau.** It receives one scalar per genome per generation and
performs no credit assignment, so it cannot learn "in this spot, this action". Better features
raise the plateau; they do not remove it.

**PPO will produce a strong but exploitable bot.** Self-play policy gradient does not converge
to a Nash equilibrium in imperfect-information games. That is a property of the method, not a
bug to be found.

The value is not beating CFR. It is measuring *where* each method plateaus and *by how much*,
against a benchmark that does not share their lineage.

---

## Hazards

**`reset_hand()` does not restore stacks.** Once a player busts, every subsequent hand is
instantly over with no player to act, and a loop that only calls `reset_hand()` spins forever
producing nothing — silently, with no error raised. This hung the feature audit's sampler
before it was found.

*Checked, 15 August: the existing training paths are safe.* `training/fitness.py` never calls
`reset_hand` — its game pool calls `game.__init__` with fresh stacks per hand, which is the
correct cash-game semantics. `training/self_play.play_match` reuses a table but guards it,
resetting every stack once fewer than two players have chips. New code that reuses a table
still needs the guard; the existing code has it.

**Six-max stack drift in `play_match`, for Phase 5.** That guard fires only when fewer than
*two* players remain solvent. Six-handed, a player who busts stays at zero while the other five
play on, so a long match drifts from a cash game toward a tournament with progressively unequal
stacks. Per-hand chip deltas stay correct — `hand_start_stacks` measures from the top of each
hand — but the *game being played* changes as it goes, and short stacks play differently.
Heads-up is unaffected: one bust leaves one solvent player, the guard fires, both stacks reset.
Worth resolving before six-max evaluation, not before heads-up.

**The machine is memory-limited, not CPU-limited.** 16 cores against 7.7 GiB, and WSL has been
terminated twice under memory pressure this week. Torch has CUDA available, so PPO can train
on the GPU; the evolutionary runs are CPU-parallel and are the ones to watch.

**Wall-clock budgets measure the machine.** Any budget-ladder experiment needs the quiet-machine
gate, and load should be sampled inside each training window rather than once per measurement —
see the crossover run's instrumentation.
