# Review: converging a lossless preflop under external-sampling MCCFR

Written 15 September 2026 from a literature search plus a read of the solver code. Sources
are linked inline. The reviewer verified three things in the code before writing: the
native trainer runs one traversal per player per iteration with regrets updated on the fly
(already the alternating form); `discount_once` applied a single t/(t+1) factor at visit
time rather than the cumulative product since the last visit, so at rarely visited nodes the
`linear` rule collapsed to vanilla (fixed the same night, see `cfr/updates.py`); and a
preflop all-in is scored by sampling one runout, so every regret sample at the shove node
carries ±100bb of board noise.

## 1. Why rare infosets converge slowly under external sampling, and which fixes transfer

Gibson et al. (AAAI 2012, https://poker.cs.ualberta.ca/publications/AAAI12-generalmccfr.pdf)
bound the sampled counterfactual value by Δ/δ where δ is the minimum probability the
sampling scheme reaches an infoset, and show the regret bound scales with the estimator's
variance. A node reached rarely with high-variance samples is the worst case on both counts,
which is a three-bet shove split across 169 classes where each sample is a ±100bb runout.

Lanctot's thesis (section 4.6.2, https://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf):
the plain "stochastically-weighted averaging" we use is the correct unbiased scheme and
converges faster in practice than optimistic averaging. So the averaging estimator is not
the problem; the weighting across time is.

Brown and Sandholm (AAAI 2019, https://arxiv.org/abs/1809.04040) tested LCFR against vanilla
MCCFR under sampling: "discounted Monte Carlo CFR demonstrated superior performance in HUNL
compared to vanilla MCCFR", strongest where mistake actions are large. On CFR+: "the changes
present in CFR+ (a floor on regret at zero and linear averaging) do not lead to superior
performance when applied to MCCFR." DCFR (α 1.5, β 0, γ 2) was not tested with sampling;
with β = 0 halving negative regret every iteration, a single lucky draw can flip an action
positive under noisy samples, so do not use it with sampling without measuring. Their MCCFR
discounts globally every 10^7 nodes touched; Pluribus discounts everything every 10 minutes
for the first 400 minutes (https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf);
Deep CFR stores a weight equal to t when the entry was added
(https://arxiv.org/pdf/1811.00164). All equivalent to weighting iteration t's contribution
by t regardless of visit frequency. Our per-visit discount was not: a node visited at t1
then t2 got t2/(t2+1) ≈ 1 instead of t1/t2, invisible on Leduc where every node is visited
nearly every iteration.

Also transferable: VR-MCCFR (Schmid et al. 2019, https://arxiv.org/abs/1809.03057) reports an
order-of-magnitude speedup from baselines and that reduced variance "allows for the first
time CFR+ to be used with sampling". Probing gave 18 percent on abstract Hold'em (Gibson
2012). Regret-based pruning (Libratus, Pluribus) is about a factor of 2 in cost, not
convergence at rare nodes. Warm starting (Brown and Sandholm 2016,
https://ojs.aaai.org/index.php/AAAI/article/view/10056) needs a principled regret
initialisation from a strategy; seeding sums by hand is not that method.

## 2. How practical bots get a 169-class preflop right

Pluribus's blueprint used plain external-sampling MCCFR with lossless preflop and 200
buckets after, linear weighting for the first 400 minutes, pruning after 200 minutes, and it
did not accumulate the average strategy until 800 minutes had passed; it stored the average
only for the first betting round and used current-strategy snapshots after that. In play it
uses the final-iteration strategy "to avoid poor actions that are not completely eliminated
in CFR's weighted average strategy". Slumbot NL kept each of the 169 hands as its own bucket
and ran a single preflop task on a central machine with sampled boards but full evaluation
of hand vectors (https://cdn.aaai.org/ocs/ws/ws0979/7044-30516-1-PB.pdf), which is
Johanson's public chance sampling (https://poker.cs.ualberta.ca/publications/AAMAS12-pcs.pdf).
The standard trick exists (sample the board, enumerate the hands), but nobody does it only
for preflop; the blueprints out-iterate the problem (Pluribus: 8 days on 64 cores).
OpenSpiel's ES-MCCFR uses the same simple averaging as ours.

## 3. Ranking by expected gain per hour on this machine (3M iterations about 2 h)

1. Correct linear weighting, then rerun 3M. Small code change plus 2 h. Papers say about
   3x on iterations for LCFR under sampling; the reviewer's inference is that the gain is
   larger at our rare nodes because there the early uniform strategy dominated the sum.
2. Skip average accumulation for the first quarter of the run (Pluribus's 800 minutes).
   One line, zero cost, combine with 1.
3. Exact preflop all-in evaluation from a precomputed hand-vs-hand equity table
   (`results/cfr/equity_tables` exists). Zeroes the ±100bb board variance at exactly the
   shove nodes; the cheapest slice of the VR-MCCFR idea. Half a day, no per-iteration cost.
4. More iterations. Error at a node falls as 1/sqrt(visits); matching the 6-class solver's
   per-node visits needs 28x, about 58 h. Only after 1 to 3; 10M (7 h overnight) is the
   practical ceiling.
5. Intermediate class count (20 to 40 by equity). No bot in the literature does it; 4 to 8x
   more visits per class than 169 and still fixes the Chen-class lumping. Fallback.
6. Pruning (speed only), RM+/CFR+ (do not, per Brown and Sandholm), exhaustive preflop
   vector traversal (a week), warm start (needs the principled method).

Run first: 1 + 2 together, same seed and tree as `ladder169`. Confirming result: at the
shove node, mean |ΔP(fold)| between adjacent hands drops from 0.42 to under 0.1 and
direction changes from 114 to a few dozen, AA/KK call above 0.95, and the same-tree gate
versus the 6-class solver moves from −4.4 ± 1.8 to at least 0 at 40,000 hands. Log
per-infoset visit counts in the same run so the rare-node hypothesis is measured.

Further sources: Lanctot et al. 2009 (https://mlanctot.info/files/papers/nips09mccfr.pdf),
Farina, Kroer and Sandholm 2020 (https://arxiv.org/abs/2002.08493), Burch et al. 2012
(https://proceedings.neurips.cc/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf),
Brown and Sandholm 2017 pruning (https://arxiv.org/abs/1609.03234), slumbot2019
(https://github.com/ericgjackson/slumbot2019).
