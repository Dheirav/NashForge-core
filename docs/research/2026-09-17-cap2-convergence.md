# Why the cap-2 solvers lose to the one-raise ones, and what the literature does about it

> **6 October 2026:** the numbers below were measured on the gate as it was in September. On the 6 October gate (stack
> cap, current miss handling) the lane Y 70bb pair, cold 10M against 20M warm, reads +1.9 ± 0.3 to the 10M, where this
> week read 2.8 ± 0.9 to the 20M. Re-check a figure here on today's instrument before quoting it
> (`2026-10-06-visit-bar.md`). No set in play uses a uniform cap-2 rung.


17 September 2026. Written after the cross-tree gates of Thursday morning and the warm-start
lanes of Thursday afternoon (NEXT.md, "What Thursday morning measured" and "lanes R, S and T").
Lane U's frozen-phase result is appended at the end when it lands.

## What was measured

Every cap-2 rung in the bot loses to the one-raise rung at the same depth, head to head, with
both bucketing the same way and the one-raise agent never re-raising: 19 to 30 BB/100 at 3M
and 10M iterations, at every depth from 12bb to 100bb. The one-raise strategy is a legal
strategy inside the cap-2 game, so a converged cap-2 solve cannot lose to it; the number is a
measure of how far from converged the cap-2 solves are.

The gap closes with iterations and shows no floor. The 18bb rung against its one-raise rung:

| iterations | cold | proportional warm start (lane S/T) |
|---|---|---|
| 3M | −35.1 ± 6.0 | |
| 10M | −26.0 ± 0.4 | −21.9 ± 1.0 (weight 1M), −21.5 ± 3.7 (weight 5M), −25.4 ± 3.1 (scale 10) |
| 50M | −12.7 ± 0.8 | |

The tree is fully reached by about 4M iterations (235k infosets, flat after that), and at 50M
replacing the random-on-miss fallback with check/call moves the number by one point where at
10M it moved twelve. So what remains is the quality of play at nodes the solver has visited
thousands of times, which is the slow part of MCCFR convergence, not holes.

## How this compares with published MCCFR runs

Brown and Sandholm's discounted-CFR paper reports external-sampling MCCFR on HUNL subgames in
terms of nodes touched: at about 10^10 nodes touched, plain and linear MCCFR sit at roughly 100
to 300 mbb/g of exploitability on their subgames 3 and 4 (their figures 10 and 11). Our 50M
iterations at 18bb is on the order of 10^10 node visits, and the cross-tree gap of 12.7 BB/100
is 127 mbb/g. So the solver is doing about what sampled CFR does at this budget. This is not a
defect in the port; it is the algorithm's rate.

The same paper's full-width variants (CFR+, DCFR) reach 1 mbb/g, "essentially solved", in
about 1,000 iterations on the same subgames. That is the standard research answer for an
abstraction of this size: build the abstract game explicitly, with chance as bucket-transition
tables computed once from sampled deals, and run full-width CFR+ or DCFR over it. Our abstract
game projects to 7.6M information sets at 18bb, which is well inside what those solvers handle
in memory. That is the structural fix and it is post-season work, because it is a new solver
path, not a parameter.

What the literature says about the levers we do have:

- **Linear discounting with MCCFR** is the recommended combination (the paper's "Linear MCCFR":
  multiply all regrets and strategy sums by n/(n+1) after each period of 10^7 nodes touched;
  the "initial discount" variant, a single 1/10 after the first period, does about as well).
  Our per-node cumulative discount, fixed on 15 September, is the per-iteration form of the same
  thing, so the rule is right. DCFR's β = 0 or negative is for full-width runs and is not shown
  to help MCCFR; CFR+'s regret floor is reported not to help with sampling. No change indicated.
- **Regret-based pruning** (Brown and Sandholm 2015, 2017; used in Pluribus): after a warm-up,
  most traversals skip actions whose cumulative regret is very negative, with exemptions for
  the river and for actions that end the hand. Pluribus's blueprint used it with linear MCCFR.
  It speeds a run by not traversing hopeless subtrees; reported gains are large in nodes touched,
  though the ceiling per iteration count is unchanged. Not implemented here; a day's work, and a
  candidate for post-season.
- **Warm starting** (Brown and Sandholm 2016): initialise regrets from an input strategy profile
  so CFR provably skips the iterations that profile is worth, at the cost of one traversal. The
  method sets *substitute regrets* from the counterfactual values of the input strategies, chooses
  T to match the input's quality, and reports that solving a coarser abstraction first and warm
  starting the full game from it improves convergence overall. Their experiment warm-started
  full Flop Hold'em from a 5,000-bucket MCCFR strategy at T = 70 with a scaling λ = 0.08.

## What our warm start did, and why it fell short

Lane S's warm start (`MCCFR::warm_start`, mode `proportional`) sets a node's regrets
proportional to the prior's probabilities at a chosen magnitude, so regret matching plays the
prior from the first iteration. That is the cheap form, not Brown and Sandholm's. It bought
about 6M cold iterations (−21.9 against −26.0 at 10M), a 1.6x speed-up, and heavier priors did
not improve it.

Two reasons, and the second is the one the literature addresses:

1. **The linear rule discounts a prior placed at iteration N by about N/T.** By 10M a prior at
   1M carries a tenth of its weight. Making it heavier (lane T) did not help, so this is not the
   binding constraint.
2. **The prior says nothing about the actions the coarser game did not have.** Proportional
   seeding gives the re-raise actions zero regret, and the shared nodes' probabilities are the
   one-raise equilibrium, which is not the cap-2 equilibrium at those nodes once re-raises exist.
   The run then has to *discover* that re-raising is good at each node by accumulating regret
   from scratch, which is most of the remaining work. Brown and Sandholm's substitute regrets
   are exactly the fix: regrets are set from the counterfactual values of every action under
   the prior, so an action the prior never took but which is better than the prior gets
   positive regret before the first real iteration.

## Lane U: the substitute regrets, sampled

`MCCFR::warm_start` mode `frozen` (trainer flag `--warm-mode frozen`): for the first N
iterations every node plays the prior (uniform where the prior has no entry) instead of regret
matching, regrets accumulate as normal, and the average is not accumulated; then regret matching
takes over. At the end of the phase each node holds N iterations of measured regret against the
prior, for every action including the re-raises. This is the paper's one traversal, sampled,
with T = N and λ = 1; on Kuhn a wrong prior held for 10% of the run recovers fully (−0.0557
against −1/18 = −0.0556), where the proportional form needed a light prior to do the same.

Results, 17 September evening. First on the old instrument (random on a miss, one training
run each): N = 3M −29.2 ± 1.6, N = 1M −30.6 ± 2.1, both worse than cold's −26.0; N = 300k
−29.6 ± 4.7; N = 30k −23.9 ± 2.4; **N = 100k −16.4 ± 2.3**. A long frozen phase measures regret
against an opponent who never re-raises and stores confident wrong lessons about aggression,
which the run then has to unlearn; a short one gives the direction of every action without
the bias. That is the paper's point that T must match the prior's quality, and the one-raise
prior's quality in the cap-2 game is low.

Then on the new instrument (`tools/xtree-gate.sh`, check/call on a miss), with the 100k
result replicated at two more seeds and a second cold seed for the run-to-run variance:

| 18bb cap-2 at 10M | vs one-raise, check/call on a miss |
|---|---|
| cold, seeds 0 and 1 | −21.6 ± 2.0, −17.8 ± 3.6 |
| frozen 100k, seeds 0, 1, 2 | −11.4 ± 1.4, −13.7 ± 2.2, −14.1 ± 2.2 |
| cold 50M, seed 0 | −8.1 ± 2.7 |

**The light frozen warm start is real and worth about 3x in iterations**: three seeds near
−13 against two cold seeds near −20, and −13 is where the cold curve sits at roughly 30M.
Two lessons about instruments came with it: random-on-miss inflated every gap and its noise
(cold 10M read −26.0 there and −21.6 here), and single training runs differ by about 4 points
seed to seed at 10M, so a solver claim needs two seeds.

## What to do with the result

- Post-season, first: every cap-2 rung warm-started (frozen 100k from its one-raise rung) at
  20M on six threads, about a night for the ladder, each gated on `tools/xtree-gate.sh` at two
  seeds, then the replay, then bursts; by the curve a warm 20M should land at or past cold 50M.
- In parallel, the full-width solver on the explicit abstract game (the chance tables exist:
  `scripts/cfr/chance_tables.py`, `results/cfr/chance/nolimit_18bb.npz`, tested), prototyped at
  8bb and gated the same way; it is the route to "solved" rather than "3x faster".
- The cross-tree gate against the one-raise rung, check/call on a miss, two seeds, is the
  instrument for solver work from now on; the same-tree gate reads +2.9 between solves that
  differ by 13 points on it, and random-on-miss measures the fallback.

## Sources

- Brown and Sandholm, "Strategy-Based Warm Starting for Regret Minimization in Games", AAAI 2016.
  https://ojs.aaai.org/index.php/AAAI/article/view/10056
- Brown and Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization",
  AAAI 2019. https://arxiv.org/abs/1809.04040 (the MCCFR section and figures 10 and 11)
- Brown and Sandholm, "Regret-Based Pruning in Extensive-Form Games", NIPS 2015; "Reduced Space
  and Faster Convergence in Imperfect-Information Games via Pruning", ICML 2017.
  https://arxiv.org/abs/1609.03234
- Brown and Sandholm, "Superhuman AI for multiplayer poker" (Pluribus), Science 2019, supplement:
  linear MCCFR with negative-regret pruning on 95% of traversals after a warm-up, river and
  hand-ending actions exempt.

## 19 September: the recipe taken to 100M, and where it stalls

Lane AB ran the 18bb and 12bb cap-2 rungs at 100M with everything on: frozen 100k warm start,
pruning at −300 stacks from 10%, three threads each (5.4 h and 3.5 h). Against the one-raise
rung, check/call on a miss:

| rung | 3M cold | 10M cold | 10M warm | 50M cold | **100M warm, pruned** |
|---|---|---|---|---|---|
| 18bb | −35.1 | −21.6 / −17.8 | −11.4 / −13.7 / −14.1 | −8.1 | **−8.0 ± 2.2** |
| 12bb | | | | | **−3.7 ± 3.1** (warm 20M: +4.2 ± 3.9) |

Same-tree, 100M against the 20M warm solve: 18bb +7.5 ± 3.0, 12bb +12.2 ± 7.0. So the solves
still improve on their own tree, and the cross-tree gap still narrows, but on a log scale it is
flattening: 18bb has gone −35, −22, −13, −8, and the last doubling of effort bought nothing over
cold 50M. 12bb is level with its one-raise rung at two budgets, which at a near push-or-fold
depth is about the ceiling.

Reading: the sampled solver, warm-started and pruned, gets a cap-2 rung to within about 8
BB/100 of the one-raise rung at 18bb for five hours of three threads, and no further at any
budget we can afford. That is good enough for companions (these are the best we have) and not
enough for the cap-2-primary plan. The remaining gap is either the sampled solver's floor at
this abstraction (the full-width solver's question) or the abstraction itself (six postflop
strength classes; a re-raised pot may need more). The 8bb full-width prototype, gated the same
way, is the next experiment, before any more iterations are spent.

## 19 September, evening: the instrument was wrong, and the corrected answer

Two defects in the cross-tree gate, found while chasing the full-width prototype's first
result: `play_pickles` played every gate at the benchmark's default 200 chips whatever the
rung, and `cfr_agent` rebuilt a node's action list from the history alone, so at nodes the
tree had stack-capped to fold/call (34% of a cap-2 rung's keys) it rejected the stored entry
and played the miss policy. Every number above this section was measured that way; the
one-raise trees' two-wide nodes matched by coincidence, which is why the defect only hurt
cap-2 solves and looked like "cap-2 is unconverged".

On the corrected gate (rung stack and blinds, stack cap honoured, check/call on a miss,
misses 0 to 0.4%): one-raise 18bb self-play +0.2 ± 0.2; **the full-width one-raise 8bb solve
ties the sampled one-raise rung, −0.6 ± 0.7**, so the abstract game with the preflop all-in
table is faithful; **the full-width cap-2 8bb solve, a converged equilibrium of the abstract
cap-2 game, loses 11.0 ± 0.2 to the one-raise rung with real cards**; the sampled cap-2 solves
read −7.6 (8bb 20M warm), −10.5 / −11.4 (12bb 20M / 100M), −15.5 / −12.0 / −11.4 (18bb 10M /
20M warm / 100M), −10.3 / −9.1 / −6.3 (50/70/100bb 20M warm), −15.3 (100bb 10M).

Reading: convergence is not the floor; the abstraction is. A converged cap-2 equilibrium on
six postflop strength classes is worse with real cards than the one-raise equilibrium on the
same classes, which is the action-abstraction pathology (Waugh et al. 2009): more betting
options on the same coarse cards give the solver decisions it cannot make well, and a
strategy that never takes them does better. Warm start and pruning still buy convergence
(10M to 20M warm is +3.5 at 18bb, 10M to 20M is +9 at 100bb) but the gap they close ends
near −10. The full-width solver is the instrument for the next question, which is the
abstraction: a 20-class postflop abstraction at 8bb, one-raise and cap-2 solved full-width
in about an hour, gated with real cards against the 6-class one-raise rung (lane AB2).

## 19 September, later: the third defect, and what the gate can and cannot say

`benchmark()` applied one raise cap to both seats and `play_pickles` passed none, so its
default of one raise held: every cross-tree gate forbade the cap-2 solve the re-raise it was
solved with. With each seat narrowed to its own tree (`raise_caps`), every cap-2 solve wins,
+7 at 8bb (sampled and full-width alike), +22 at 12bb, +45 to +48 at 18bb, +145 to +205 at
50 to 100bb, and the one-raise side misses 17 to 27% of its decisions (every re-raise it
faces), answered by check/call. So the gate now measures a bare one-raise pickle without a
companion, which is not the bot either. Settled: the cap-2 solves are sound and close to
converged (full-width and sampled agree at 8bb; 10M and 100M within 3 at 18bb), the abstract
game is faithful, and the earlier "abstraction is the floor" reading was the instrument. Not
settled, and not measurable with single pickles: whether a cap-2 primary beats a one-raise
primary that hands re-raises to a cap-2 companion. That needs a duel between whole bot
configurations over duplicate hands, or the arena.

## 20 September: what the literature says about card abstraction (for the 20-class question)

Read: Johanson, Burch, Valenzano and Bowling, "Evaluating State-Space Abstractions in
Extensive-Form Games" (AAMAS 2013), and Ganzfried and Sandholm, "Potential-Aware
Imperfect-Recall Abstraction with Earth Mover's Distance" (AAAI 2014). Both are about limit or
no-limit Texas Hold'em at the ACPC scale, and both compare abstractions by solving each and
playing the strategies against each other in duplicate hands, which is what our duel does.

What ours is: **E[HS]**, expected hand strength against a random hand, sampled (200 runouts),
k-means in one dimension into 6 classes per street, plus a board-texture class, imperfect
recall (the key is the current bucket and the betting). That is the 2007-era Hyperborean
design, "percentile hand strength" without even the E[HS²] nesting.

What the literature found, in order of effect:

1. **The feature matters more than the count.** Johanson 2013 ranked, at equal size (169 /
   9,000 / 9,000 / 9,000 buckets, imperfect recall), k-means over **hand-strength distribution
   histograms with earth mover's distance** (KE) first, k-means over **OCHS** (eight numbers per
   hand: equity against each of eight opponent-hand clusters) close behind, and percentile
   E[HS²]/E[HS] (PHS, ours) last, on both one-on-one play and CFR-BR exploitability. The reason
   is drawing hands: 6c6d and KcQc have E[HS] 0.634 and 0.633 and end up in the same bucket
   under E[HS], while their end-of-hand equity histograms are nothing alike (a low-potential
   pair against a high-potential two overcards). Ganzfried 2014 goes a step further,
   potential-aware EMD over *next-round* bucket distributions, and beat the KE abstraction by
   2.2 to 2.6 mbb/h at 169 / 5,000 / 5,000 / 5,000 buckets.
2. **Imperfect recall beat perfect recall at equal size**, every time, because it lets the
   buckets be spent on the present street. Ours is already imperfect recall.
3. **More buckets help, with diminishing returns, and the first-round lossless 169 is standard.**
   Ours has the lossless preflop. The ACPC bots ran thousands of postflop buckets; we run six.
4. **Texture in the key** is a crude form of OCHS's public-card awareness; keeping it is right.

So the answer to "why 20": it is not the number that the literature says matters, it is the
feature. Six E[HS] classes to twenty E[HS] classes refines the same one-dimensional signal and
still cannot separate 6c6d from KcQc, which is why the 8bb test moved nothing. The change with
evidence behind it is **the feature: cluster on the hand-strength histogram with EMD** (a
50-bin histogram of end-of-hand equity per situation, k-means with EMD as the distance, 1-D EMD
is a linear scan), which the fitting code can do in place of the scalar (`_postflop_strength`
returns one number; it would return a histogram, `_fit_kmeans_1d` becomes k-means with EMD).
Then the count: 20 or 50 classes of *that*. OCHS is the cheaper cousin (eight equities per hand,
L2 distance) and is what to try if EMD clustering is slow to fit.

Cost: fitting is one-off (the 3,000-situation sample per street becomes 3,000 histograms of
200 runouts, minutes); lookup at play time is the same equity sampling we do now, binned; the
tree size is set by the class count, not the feature. So a 20-class EMD abstraction costs the
same to solve as a 20-class E[HS] one, and the literature says it plays better.

Sources: Johanson et al. 2013 (ifaamas.org/Proceedings/aamas2013/docs/p271.pdf);
Ganzfried and Sandholm 2014 (ojs.aaai.org/index.php/AAAI/article/view/8816).

## 20 September: each planned change checked against what the field does

**1. Cap-3 betting tree at 50bb and above.** Pluribus's blueprint (Science 2019 supplement)
caps the number of raises implicitly through its sizes: on the turn and river at most three
sizes for the first raise (0.5x, 1x, all-in) and at most two for any further raise (1x,
all-in); preflop up to 14 sizes; the flop coarser. Libratus's river subgames: 0.25/0.5/1/2/4/8x
first in, 0.4/0.7/1.1/2x facing a bet, 0.4/0.7/2x after one raise, 0.7x only after more. So
the field's answer to "the third raise" is not a third full menu; it is **one or two sizes
for every raise past the first**, which keeps the tree small while always having a node.
Better than plain `--raise-cap 3` (which triples the tree with four sizes at every raise):
a tapered schedule, e.g. `(4, 2, 1)`: four sizes first in, pot and all-in for the re-raise,
all-in only for the third. The trainer already takes a tuple schedule. The tree grows by a
fraction, not 3x, and the third-raise hole closes. **Do this instead of cap-3.**

**2. Card abstraction.** Covered above: the feature (EMD over equity histograms, or OCHS)
before the count. Pluribus used 200 buckets per postflop round in the blueprint on "k-means
over domain-specific features" and 500 in search; Johanson's ACPC bots 9,000 per round on the
EMD histograms. Our six E[HS] classes are two orders of magnitude coarser and on the weakest
feature. **Do the feature first (EMD histogram or OCHS), at 20 to 50 classes**, one deep rung,
duel-gated.

**3. Action translation.** The bridge already uses Ganzfried and Sandholm's pseudo-harmonic
mapping, randomised (`abstraction/translation.py`); Pluribus uses the same, deterministic in
search and randomised in play. Nothing to change; the caveat from that paper stands: the
mapping is invariant only in pot fractions, which the bridge honours.

**4. River play.** Every modern bot re-solves the river (Libratus: nested safe subgame solving
from the turn; Pluribus: depth-limited search on rounds two to four; DeepStack: continual
re-solving). Our `cfr/river.py` is a river re-solver built on 16 September and switched off;
the blueprint's river node comes from a six-class abstraction, which is where the re-solver
adds the most (it plays the exact hand against the blueprint's range). **Turn it on for a
duel** (`--river-solve`, 8 s budget) before building anything else on the river; the
literature says this is the single largest gain available to a small-abstraction bot, and it
costs no training.

**5. Reads.** No literature; opponent modelling in this field is the reads we have. The one
established idea is to use them only where the equilibrium is indifferent (Ganzfried's
"safe exploitation"), which is what thresholds on measured counts approximate.

**6. The solver.** Linear MCCFR with pruning is exactly Pluribus's recipe (they estimate
linear at 3x over plain MCCFR and prune on 95% of iterations off the last round and off
terminal actions, which is what `set_pruning` does). Nothing to change.

Revised order, by expected gain per night of work: river re-solving on (a duel, no training);
tapered third raise `(4, 2, 1)` at 50bb and above; the EMD/OCHS abstraction at 20 to 50
classes on one deep rung; then the count.
