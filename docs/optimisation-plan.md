# Optimisation plan

Written 11 September 2026, after a day of measurement. The headline is that the
remaining speed and the remaining *quality* problem have the same fix, and it is
the thing serious engines already do.

## Where we are, measured

Today took the solver from **32.4 ms per iteration to 7.22 ms**, 4.3×, by
compiling the hand evaluator and moving the rollout's sampling inside it. An
ablation then split what is left:

| | ms/iteration | share |
|---|---|---|
| equity rollouts | 3.20 | 44.3% |
| rest of the bucket path (cache, key building) | 1.10 | 15.2% |
| traversal floor (MCCFR and game logic, no card work at all) | 2.92 | 40.5% |
| **total** | **7.22** | |

So **2.47× is the hard ceiling** for anything that optimises the card path, and
40 percent of runtime is untouchable without restructuring the inner loop.

And separately, the card abstraction is noise-limited. The equity estimate's
standard deviation at 40 samples is **0.0667** while adjacent bucket centroids
sit **0.101 to 0.170** apart, so **42 percent of situations change bucket on a
re-roll**. We are computing the wrong bucket, quickly.

## The insight

Those look like two problems and they are one. We compute equity **during
traversal**, about 43 times per iteration, using a sample count small enough to
be affordable and therefore too small to be accurate. Serious engines do not do
this at all: Slumbot's buckets are precomputed offline by clustering, and at
traversal time a bucket is an array index.

Doing the same removes the 44 percent *and* the 42 percent, because a
precomputed bucket can afford a sample count we could never pay for per node.

The combinatorics say it is affordable, which I did not expect:

| street | canonical situations | exact equity cost | table size at 1 byte |
|---|---|---|---|
| flop | ~1,300,000 | 1,070,190 comparisons, far too many | **1.3 MB** |
| turn | ~15,300,000 | 45,540 comparisons | **15 MB** |
| river | ~140,000,000 | **990 comparisons** | too large |

The flop and turn tables are trivially small. The river table is not, but the
river needs no table: with a complete board, equity against a random hand is an
**exact enumeration over 990 opponent holdings**, which at our current 7.0 M
evaluations per second costs 0.28 ms and with a lookup-table evaluator costs
**0.025 ms**, against the 0.074 ms we currently spend getting a noisy answer.

So: precompute flop and turn, enumerate the river exactly. Every street ends up
faster *and* exact or near-exact.

## The plan

### Phase 0 — parallelism, now, free

The solver is single-threaded and the machine has eight cores. Every experiment
we run has independent pieces. Running them concurrently is **3 to 4× on
experiment wall-clock** with no change to the solver and no effect on any answer.

**The one rule:** iteration-budgeted work only. A wall-clock-budgeted ladder run
in parallel reproduces the seed-2 contamination of 10 September, where
contention meant one arm bought a tenth of the traversals at the same nominal
budget.

### Phase 1 — precompute the flop and turn bucket tables

The work is a canonical index: a bijection from a suit-isomorphic (hole, board)
situation to an integer, so the table can be a flat array. That is the real
engineering here, and it is a known problem with known solutions.

Cost to build, at 1,000 samples per situation, which gives a standard deviation
of **0.0134** against today's 0.0667:

- flop: 1.3M situations x 2,000 evaluations = 2.6e9, about **6 minutes** at our
  current 7.0 M/s, and well under a minute with a table evaluator
- turn: 15.3M x 2,000 = 3.1e10, about **73 minutes** currently

Both are one-off, both parallelise perfectly across cores, and the output is
16.6 MB total.

**What it buys:** flop and turn bucketing become array indexing, removing most of
the 44 percent and most of the 15 percent. Bucket misassignment from Monte Carlo
noise drops from 42 percent toward the low single digits.

### Phase 2 — exact river equity

990 comparisons, no sampling, no noise. Cheaper than what we do now once the
evaluator is fast, and correct by construction rather than approximately right.

### Phase 3 — a lookup-table hand evaluator

Ours does 7.0 M evaluations per second compiled, by counting ranks and suits and
branching through nine hand classes. The [two-plus-two
algorithm](https://bostik.iki.fi/aivoituksia/projects/twoplustwo-hand-evaluator.html)
does it in **seven array lookups** with no branching, reported at 80 M/s on
random hands and 250 M/s enumerated;
[HenryRLee's](https://github.com/HenryRLee/PokerHandEvaluator) uses a perfect
hash in about 100 KB.

That is **11 to 35×** on evaluation, and it is what makes Phase 1's table build
take minutes instead of an hour and Phase 2's exact river cheap.

Phase 3 helps Phases 1 and 2 but is not required by them, which is why it is
last: the tables can be built with the evaluator we have.

### Phase 4 — the traversal floor, only if it still matters

2.92 ms of Python object churn: dict lookups, tuple rebuilding, string history
parsing, per node. Compiled, code of this shape typically runs 20 to 50× faster,
though that range is an assumption rather than something measured here.

**Do not start this until Phases 0 to 3 are done**, because it is the only item
that moves the hot path outside the 293 tests that are this project's defence
against its own history, and because after Phase 1 it will be nearly all of what
remains and therefore easy to justify or dismiss on measurement.

## What we are not doing, and why

**Suit isomorphism on the runtime cache: tried, reverted.** Correct, verified
across 20,000 situations against all 24 suit permutations, and a regression, 7.58
to 9.4 ms. The miss rate is dominated by the size of the situation space rather
than by suit redundancy. Note that Phase 1 uses suit isomorphism for something
different and entirely sound: shrinking a precomputed table, not deduplicating a
runtime cache.

**Lowering `equity_samples`: the measurement pointed the other way.** At 40 the
estimator already cannot resolve six buckets.

**Caching history parsing, scoring showdowns: tried, reverted, neutral.** Both
were ranked from a `cProfile` run whose per-call overhead inflates anything
called millions of times.

## Expected outcome

Phases 0 to 3 should land the solver near the 2.47× ceiling of the card path,
so roughly **3 ms per iteration**, while cutting bucket misassignment from 42
percent to a few percent. Combined with today's 4.3×, that is about **11× from
where this morning started**, with a materially better abstraction rather than
the same one computed faster.

A serious engine would still be ahead, mostly on Phase 4, and the honest estimate
of that remaining gap is 20 to 50× on the traversal alone.
