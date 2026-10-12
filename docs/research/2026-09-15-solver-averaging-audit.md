# Audit: how the C++ MCCFR accumulates regrets and the average strategy

Written 15 September 2026 by a read-only code audit, to explain why the 169-class preflop
abstraction (5,746 preflop information sets) is unconverged at rarely reached nodes after
3M iterations while the 6-class one (204) converges at 250k. Quotes are file:line.

## 1. Traversal

Textbook external sampling with alternating traversers; the traverser's reach probability
is never computed or used.

- `native/src/mccfr.hpp:108-112`: `train` loops both players as traverser every
  iteration, then `++iterations_`.
- The traverser enumerates every legal action (`mccfr.hpp:150-153`) and forms
  `value = Σ strategy[i]*values[i]`; the opponent's action is one sampled draw
  (`mccfr.hpp:144-145`). No reach variables in `walk` (`mccfr.hpp:119`), correct for
  external sampling. Same in `cfr/mccfr.py:78`.

Chance is not sampled once per iteration. `mccfr.hpp:122-123` calls `sample_chance` at
every chance node, so each traverser action branch that reaches a street deal draws its own
board (`nolimit_game.hpp:112-148`), and the root deal is redrawn for each traverser. This is
unbiased but high variance: `values[i] - value` (`mccfr.hpp:161`) compares actions evaluated
on different boards. One deal per iteration reused across the traverser's branches (common
random numbers) is the largest free variance reduction available.

## 2. Strategy averaging

Accumulated only at the opponent's nodes, unweighted: `mccfr.hpp:141-143`,
`strategy_sum[i] += weight * strategy[i]` with `weight = rule_.strategy_weight(t)`, which is
1.0 unless `linear_strategy` (`mccfr.hpp:88-90`). Mirrors `cfr/mccfr.py:105-106`.

This is Lanctot 2009's external-sampling average: visit frequency at an infoset owned by
player p equals π_chance · π_p, so unweighted accumulation already implements the
π_p-weighted average. It is correct, and that is precisely the problem here.

A node's `strategy_sum` accrues at a rate proportional to its owner's own current reach.
For a three-bet-shove node the owner's strategy gives the shove near-zero probability early
(regret matching gives exactly 0 to any action without positive regret, `mccfr.hpp:34-35`),
so the node is entered as opponent almost never, while its regrets keep improving because
when p is traverser the shove branch is enumerated unconditionally (`mccfr.hpp:150`).
Regrets converge; the average does not, frozen at the handful of near-uniform early visits
that each carried weight 1. Vanilla weighting makes iteration 1 count as much as iteration
3,000,000 (`cfr/updates.py:9-10` says exactly this).

Fix: linear averaging, so late sharp visits dominate early uniform ones by a factor of t.
`results/cfr/update_rules_leduc.json` shows linear best at 64k (0.0559 vs vanilla 0.0615),
but that experiment cannot see this effect: Leduc's 288 infosets are all densely visited.
The comparison that motivated the vanilla default never tested this regime.

## 3. Regret matching and update rules

Vanilla regret matching, no flooring by default (`mccfr.hpp:92-98`). Discount applied at
most once per node per iteration, before the contribution (`mccfr.hpp:166-174`).

`UpdateRule::linear()` (`mccfr.hpp:68`) was unreachable from Python: both bindings
hardcoded vanilla (`bindings.cpp:81` and `:120`). Fixed 15 September: `rule=` on
`NoLimitSolver` and `--update-rule` on the trainer. Note `linear()` discounts the strategy
sum by t/(t+1) each iteration (gamma = 1), which is equivalent to weighting iteration t by t.

## 4. `average_strategy()` and the arena lookup

`mccfr.hpp:51-57`: a node whose `strategy_sum` totals zero is returned as exact uniform.
Such nodes exist: created at `mccfr.hpp:131` while the owner was traverser and never
entered as opponent, with non-zero regrets. `evaluation/benchmark.py:191` treats it as a hit
and samples from it. Facing an all-in the legal set is exactly {fold, call}
(`nolimit.hpp:135`), so a never-averaged shove node yields exactly 50/50 and is
indistinguishable downstream from a deliberate mix. The `misses` counter cannot see it. This
fully explains the 50/50 at shove nodes in `docs/arena-plan.md`.

## 5. Rare-action starvation

No exploration floor anywhere: `sample` (`mccfr.hpp:176-184`) draws from the regret-matching
distribution, which returns hard zeros, so a subtree behind an opponent action at current
probability 0 receives zero strategy-sum updates for as long as that holds. With 169 classes
each class's share of reach drops about 28x. `results/cfr/ladder169/cap2_100bb.json` reports
2,223,483 infosets at 3M iterations, roughly one strategy-sum sample per infoset per budget.
Remedies: linear weighting (done above), or epsilon exploration on the opponent's sampling
with the matching importance weight.

## 6. Per-node cost

Keys are strings (`nolimit_game.hpp:177`) into `unordered_map<string, InfoSetNode>`
(`mccfr.hpp:189`): two allocations and a full hash per visit, and the traverser branch
hashes again at `mccfr.hpp:158`. `legal_actions` returns a vector by value
(`nolimit_game.hpp:102`). 6 to 169 classes adds one key character; the real cost is table
growth, rehashing and cache misses. `bucket_for` indexes the preflop table directly
(`nolimit_game.hpp:210-213`); the memo covers postflop rollouts only.
