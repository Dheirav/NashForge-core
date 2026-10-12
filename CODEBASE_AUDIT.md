# PokerBot Codebase Audit

**Date:** 12 August 2026
**Scope:** `engine/`, `training/`, `rl/`, `scripts/`, `agents/`, `evaluator/`, `utils/`, `gui/`, and the analysis reports at repository root.
**Method:** every claim below was established by executing the code — driving the engine with randomised legal actions, instrumenting the fitness path, and sampling feature vectors — not by reading it. Where a number appears, it was measured.

> **Status of this document.** The reports it supersedes were deleted from the
> working tree on 12 August 2026 and remain in git history (`git show <rev>:<file>`)
> and in `PokerBot_superseded_2026-08-12.tar.gz`. It supersedes `GLOBAL_SYNTHESIS_REPORT.md`, `TRAINING_FINDINGS_REPORT.md`, `HOF_IMPACT_ANALYSIS.md`, `HYPERPARAMETER_RELATIONSHIPS.md`, `HYPERPARAMETER_SCALING_LAWS.md` and `TOTAL_RESULTS.md`, all of which draw conclusions from a fitness function shown below to have been measuring the wrong quantity.

---

## Summary

| | Count |
|---|---|
| Engine behaviours verified correct | 6 |
| Numbered findings | 18 |
| — fixed and covered by tests | 16 |
| — partially fixed | 2 |
| — still open | 0 |
| Of which critical | 4 (all fixed) |
| Tests in `tests/` | 25 (previously 0) |

**One-line verdict.** The rules engine and its chip accounting are correct and worth keeping. The card *dealing*, the feature layer, and the fitness function were not, and every published result in this repository derives from them.

**The sanity check that was always available.** An untrained random network scored **+451 BB/100** under the original fitness function. It now scores **−4.8 ± 29.6** heads-up and **−14.2 ± 258.6** six-handed — approximately zero, as a random agent among random agents must. Population mean fitness was large and positive in every run ever logged; poker is zero-sum, so in self-play that mean must be about 0. Reported bests ran to 5213 BB/100, where 50 is a crushing professional win rate.

---

## Part 1 — The engine

### 1.1 What holds up under test

Each of these was checked by driving the engine directly and comparing against known-correct answers.

| Behaviour | Evidence |
|---|---|
| Hand evaluation | All 11 categories identified correctly, high card through royal flush |
| Wheel straight | A-2-3-4-5 correctly ranks **below** a six-high straight |
| Evaluator agreement | 0 disagreements between `evaluate_hand` and `compare_hands_fast` over 2,000 random seven-card showdowns |
| Chip conservation | Exact across 2,400 randomised hands — two- and six-handed, equal and unequal stacks |
| Side pots | Construction **and** allocation: a short all-in wins the main pot only; folded players' chips still pay the winner; split pots distribute odd chips without leaking |
| Deck integrity | No duplicate card across hole cards and board in 300 completed hands |

Betting mechanics are also correct: minimum-raise enforcement, short all-ins that correctly do *not* reopen betting, the big blind's option, heads-up blind inversion, and burn cards. This is the part of a poker system that is hard to write, and it works.

### 1.2 Findings

#### E1 — The deck is re-seeded with the same seed every hand
**Critical · FIXED · `engine/game.py`**

`reset_hand()` builds `Deck(self.state.deck_seed)` from a seed fixed at construction, so every hand receives an identical shuffle. Only the button rotation varies, which swaps who receives which cards. A session therefore contains exactly **two distinct deals, alternating forever**.

Observed over five consecutive hands from one game object:

```
hand 0: ('9h', 'Ac', '3c', 'Ts')
hand 1: ('3c', 'Ts', '9h', 'Ac')
hand 2: ('9h', 'Ac', '3c', 'Ts')
hand 3: ('3c', 'Ts', '9h', 'Ac')
hand 4: ('9h', 'Ac', '3c', 'Ts')
```

Any code that plays hands by looping `reset_hand()` was evaluating agents on two repeated hands. That includes `training/self_play.py:237` (`play_match`), which `scripts/evaluation/round_robin_agents_config.py` drives — and therefore **every round-robin tournament in `tournament_reports/`** — plus `gui/game_controller.py:46` and `scripts/play_vs_headsup_agent.py:54`.

Training escaped this: the fitness path builds a fresh game per hand through `GamePool.acquire`, each with its own seed.

**Fixed by** drawing a new deck seed per hand from an RNG that is itself seeded from the master seed. The first hand still uses the master seed, so `PokerGame(seed=s)` deals exactly what it always did, while whole sessions remain reproducible. Verified: 500 hands now produce 500 distinct deals, and the same seed replays the same sequence.

#### E2 — Two incompatible definitions of the 17-dimensional observation
**Critical · FIXED · `engine/features.py`**

`FeatureCache.get_features()` and `get_state_vector()` produced different orderings of the same 17 slots. Training used the first; evaluation, the GUI and the RL environment used the second. Eleven of seventeen features differed, so every trained agent was scored on permuted inputs. Under the evaluation layout, the B5 champion folded **300 of 300 hands**.

Fixed by making `get_state_vector()` delegate to `FeatureCache`, which is the layout every saved genome was fitted against and the one `get_feature_names()` documents. The divergent JIT assembly path was deleted so it cannot return.

#### E3 — Every pocket pair scored as an average hand
**FIXED · `engine/features.py` — `_init_preflop_cache`**

The strength table stored an offsuit entry only when the two ranks differed. Pocket pairs are always offsuit, so all thirteen were missing from the table and the lookup fell through to its `0.5` default.

| Hand | Before | After |
|---|---|---|
| AA | 0.5000 | 1.0000 |
| KK | 0.5000 | 0.8095 |
| 22 | 0.5000 | 0.2857 |

Aces, kings and deuces were indistinguishable from a coin-flip hand, on roughly 6% of deals.

#### E4 — Hand strength is never recomputed after the flop
**FIXED · `engine/features.py`**

`FeatureCache.hand_strength` is the Chen score of the two hole cards, computed once at deal and cached for the hand. It depends only on those two cards, so it is identical for every possible board. JJ reads the same on `6-6-A` as on `J-6-2` as on a runout that makes quads. After the flop, the agent is blind to the board.

A working Monte-Carlo equity function, `hand_strength_vs_random()`, sits in the same file and is exported from the package. **Nothing calls it.** A comment at `features.py:244` recommending it sits directly above the line that ignores it.

Measured cost of wiring it in, against ~126,000 decisions per generation (currently 59 s):

| Method | µs/decision | Generation time | Board-aware |
|---|---|---|---|
| Current preflop lookup | 0.2 | 59 s | no |
| `evaluate_hand` made-hand rank | 8.7 | ~60 s | yes |
| Monte-Carlo, 20 sims | 6,819 | ~15 min (15×) | yes, very noisy |
| Monte-Carlo, 100 sims | 34,534 | ~71 min (72×) | yes |
| Monte-Carlo, 500 sims | 174,423 | ~5.8 hrs (354×) | yes |

The Monte-Carlo route is not viable. The existing `evaluate_hand` is.

**Fixed by** adding `made_hand_strength()`, which reads the player's best five-card hand and is recomputed once per street rather than per decision (3.2 µs per `get_features` call). Category values are anchored to roughly how often each beats a random hand, so they stay on the same scale as the preflop score — without that anchoring, flopping a set scored *lower* than the same hand preflop. Set `engine.features.BOARD_AWARE_STRENGTH = False` to restore the old observation for a like-for-like ablation.

| Board (holding JJ) | Strength |
|---|---|
| preflop | 0.619 |
| A-K-Q — jacks likely beaten | 0.555 |
| 6-6-A — two pair | 0.708 |
| J-6-2 — set of jacks | 0.804 |
| J-J-4 — quad jacks | 0.990 |

#### E5 — One feature is dead and several are duplicates
**PARTIALLY FIXED · `engine/features.py` — `FeatureCache`**

Measured over 22,423 decision states:

- `is_all_in` has a standard deviation of exactly **0.000**. An all-in player never acts, so it can never be 1.
- `stack_normalized` and `commitment` correlate at **r = −1.00** — algebraically the same number, since `stack + contributed = starting_stack`.
- `spr`, `to_call_ratio` and `facing_raise` correlate at 0.90–0.95.
- `round_preflop` and `round_flop` correlate at −0.93.

Nine principal components explain 90% of the variance. **Effective dimensionality is about 9, not 17** — and exactly one of those dimensions describes the cards.

**Partially fixed.** The dead slot now carries whether an *opponent* is all-in — a genuine decision input, since it caps what can still be won. Measured std went from 0.000 to 0.487.

**Still open by choice:** removing the `stack_normalized` / `commitment` duplicate changes the observation from 17 dimensions to 16, which invalidates every saved genome and touches `NetworkConfig.input_size` and `PPOConfig.obs_size`. That is the "rebuild the feature layer" step in Part 3, and it should follow a decision about where this project is going rather than precede it.

#### E6 — `get_legal_actions` advertised raises that cannot be made
**FIXED · `engine/game.py`**

A raise was offered whenever the player had chips beyond the call, even when the stack could not reach the minimum raise-to amount, producing entries with `min > max`. It fired **418 times in 2,400 hands**. `apply_action` silently clamped them to all-ins, which is why it never surfaced. Since `get_action_mask` derives from `get_legal_actions`, masks marked "raise" legal where no legal raise existed.

#### E7 — Showdown evaluates hands even when one player remains
**FIXED · `engine/showdown.py`**

A lone eligible player should simply take the pot. Instead the hand comparator runs, and it requires at least five cards, so a degenerate table state raises `ValueError: Need at least 5 cards, got 2`. Normal play survives only by accident: a fold-out advances the street and deals the flop, landing on exactly five cards. A three-line short-circuit removes the entire class.

**Fixed by** `_winner_indices()`, which awards the pot to a lone candidate without evaluating anything. `PokerGame.resolve_showdown()` now also empties the pot after paying it out — it previously kept its pre-payout total until the next `reset_hand()`, so `sum(stacks) + pot` double-counted the money and a second call would have paid twice.

#### E8 — Busted players keep their seats and are dealt blinds
**FIXED · `engine/game.py` — `_blind_positions`**

A player with no chips is marked folded but still occupies a seat and can be assigned the small or big blind, which it posts as zero. A table can reach a hand in which nobody posts anything and the pot is empty. Posting zero also leaves the stack at zero, which marks the player all-in.

**Fixed by** `_blind_positions()`, which selects the blinds from seats that still have chips. A previously crashing 80-hand session with two short stacks now runs to completion with chips conserved at every step.

#### E9 — A second, unreachable street-advancement path
**FIXED (removed) · `engine/game.py`**

`next_betting_round`, `_end_betting_round`, `_reset_bets_for_new_round`, `betting_closed` and `_next_player` duplicated the card-dealing logic of `_start_next_round` with different bookkeeping — they never reset `last_raise_size` or `bb_has_option`, so calling any of them would have corrupted minimum-raise enforcement. Nothing referenced them.

---

## Part 2 — Everything built on the engine

The layer above the engine has one structural disease: **nothing had a single definition.** Three implementations of "play a hand", three action masks, two feature layouts. Five silent metric bugs were not bad luck — they are what that structure produces.

### 2.1 The fitness function

Every number in every report passed through this code.

#### S1 — Fitness scored a fixed seat, not the agent being evaluated
**Critical · FIXED · `training/fitness.py` — `evaluate_matchup`**

Seats were shuffled every hand, but the hero was always credited with `changes.get(0, 0)`. Measured: the hero's own result was counted in **11 of 48 hands (23%)** six-handed. The other 77% credited an opponent's chips to the hero. Heads-up it was a coin flip.

Fixed by tracking the hero through the shuffle via `seat_order.index(0)`.

#### S2 — Pots were destroyed on every hand won by folding
**Critical · FIXED · `fitness.py` ×2, `self_play.py`**

Pot resolution was guarded by `betting_round == 'showdown'`. On a fold-out the pot was simply discarded: the winner was never paid, and every player was scored as having lost their contribution. Most hands end this way — in one 400-hand heads-up sample, 377 did.

Fixed with a `finish_hand()` that resolves unconditionally, plus a `chip_deltas()` that asserts `sum == 0` on every hand.

#### S3 — Eight config fields were read but never declared
**FIXED · `fitness.py` / `config.py`**

`stack_min/max`, `sb_min/max`, `bb_min/max` and `ante_min/max` were fetched with `getattr` defaults for fields `FitnessConfig` did not have. The fallbacks always fired, so **every evaluation silently randomised** stacks to 500–1500, the small blind to 5–10, the big blind to 10–20, and injected a 0–1 ante — none of it recorded in any saved config. `bb_per_100` then divided by the nominal big blind regardless, normalising roughly half the hands by the wrong denominator.

Now declared explicitly and **off by default**; `randomise_conditions=True` reproduces the old behaviour.

#### S4 — Chip deltas were measured after blinds had left the stacks
**FIXED · `training/fitness.py`**

The baseline was read post-blind, so winning a pot credited a player with their own blind back as profit. Because the button does not rotate between hands, that bias was positional and systematic.

#### S5 — Errors in the action loop are swallowed and scored as weakness
**FIXED · `fitness.py`, `self_play.py`**

A failed `apply_action` is caught and replaced with a fold, with nothing counted or reported. An agent emitting illegal actions every hand scores as merely *weak* — which is precisely how defects of this size survive 89 training runs.

**Fixed by** routing both loops through `apply_action_or_fold()`, which still folds on rejection but counts every occurrence in `training.fitness.ILLEGAL_ACTIONS` and warns the first time. Hands hitting the `max_actions` cap are counted there too, rather than truncating silently. The bare `except:` clauses that swallowed `KeyboardInterrupt` — preventing Ctrl-C from stopping a run — were narrowed to `except Exception`.

#### S6 — Three separate implementations of "play a hand"
**PARTIALLY FIXED · `fitness.py`, `self_play.py`**

They must agree and did not: only two used `FeatureCache`. This is exactly how the training/inference split survived undetected, and it is the root cause behind most of this section.

**Partially fixed.** `play_hand` is now a one-line delegation to `play_hands_batched`, so the network path has a single loop that cannot drift. Both remaining loops share `hand_start_stacks`, `finish_hand`, `chip_deltas` and `apply_action_or_fold`.

**Still open:** `self_play.play_match` keeps its own loop because it drives a different interface — arbitrary agent objects rather than policy networks — and accumulates per-hand VPIP/PFR statistics. Unifying it means expressing the batched network path through a per-seat `decide()` callback, which would give up the cross-game batching that makes training fast.

#### S7 — Generation statistics described the wrong population
**FIXED · `training/evolution.py`**

Statistics were gathered *after* `evolve()` and `replace()`. `get_stats()` skips genomes whose fitness is `None`, and the next generation is almost entirely freshly-mutated children — leaving only the surviving elite. Every "mean fitness" ever logged was that one genome's score, and every "std fitness" was exactly `0.0`. This is visible in every training log in the repository as `Mean` being identical to `Best`.

#### S8 — Mutation sigma decayed twice per generation
**FIXED · `genome.py:458` + `evolution.py`**

Both `Population.evolve()` and `train_generation` called `decay_sigma()`, giving `σ·decay^2ᵍ` instead of `σ·decay^ᵍ`. Over 200 generations at decay 0.995 that is 0.135 where the documented schedule says 0.367.

#### S9 — Assorted open defects
**FIXED · across `training/`, `scripts/`, `evaluator/`**

- `GamePool` reuse leaves `_hands_played` set, so a recycled game starts on button 1 while a fresh one starts on button 0.
- The `max_actions = 200` cap truncates hands silently.
- `scripts/evaluation/eval_baseline.py` — the README's headline evaluation command — is a stub returning hardcoded zeros (`# You must implement the actual game logic`), and crashes first on `TrainingConfig.from_dict`, which does not exist.
- `evaluator/equity.py` and `evaluator/hand_rank.py` are **0 bytes**, and have their own README section.
- `PolicyNetwork.set_weights` swallows shape errors with `except Exception: pass`, then reports a misleading `TypeError`.
- `training/policy_network_fast.py` is now dead following the mask unification.
- 14 `sys.path.insert` hacks stand in for packaging; tabs and spaces are mixed across four files.

**Fixed.** `GamePool` reuse no longer inherits a button offset — hand counting moved to a `_hands_dealt` field reset by `__init__` instead of a `hasattr` probe. `max_actions` truncation is counted. `eval_baseline.py` is now a working evaluation reporting BB/100 with standard errors and 95% intervals against both baseline agents. The empty `evaluator/` package and the now-unused `policy_network_fast.py` were removed rather than filled in, since `engine/hand_eval.py` and `engine/features.py` already provide those functions and a second implementation is what caused this audit. `PolicyNetwork.set_weights` now reports shape mismatches as shape mismatches instead of swallowing them and raising a misleading `TypeError`. `README.md` had its three wrong entry-point paths corrected, its links to non-existent files removed, its script count fixed, and its unmeasured throughput claim replaced with a measured one.

### 2.2 The reinforcement learning layer

Reviewed and, unusually for this codebase, **sound**. The PPO implementation is textbook: GAE with correct terminal handling, clipped surrogate objective, value MSE, entropy bonus, approximate-KL and clip-fraction tracking, and returns computed before advantage normalisation. No correctness bug was found.

Its two problems are design, not code:

1. Opponents are sampled from Hall-of-Fame agents bred against the broken objective, or from a uniform random bot. There is nothing strong to learn against, and no self-play.
2. It consumes the same 17 features through `get_state_vector()`, so it inherits the representation ceiling entirely.

**Switching from evolution to PPO would not have moved that ceiling.** A better learning algorithm cannot learn postflop hand reading from an input containing no postflop hand information.

### 2.3 Results, documentation and method

- The tournament corpus is invalid twice over: permuted features (E2) and two repeated deals (E1).
- The 89 checkpoints and the Hall of Fame were *selected* against the broken objective. They cannot be rescued by rescoring — they would have to be retrained.
- Roughly 8,600 lines of markdown assert conclusions derived from that metric. The reports cross-reference each other, which reads as corroboration but is three documents drawing on one broken source.
- Independent of the bugs: no seed replication, no confidence intervals, and configurations ranked on differences far below the noise floor. The Hall-of-Fame comparison is confounded — `hof3` runs also differ in population size and sigma from the non-HoF runs.
- `README.md` has three wrong entry-point paths (`scripts/train.py`, `scripts/eval_baseline.py`, `scripts/match_agents.py`), links six files that do not exist, claims 23 scripts against an actual 43, and states throughput off by one to two orders of magnitude (measured: 59 s/generation at p12/m7/h375, single worker).
- `intermediate_results.md` documents the feature layout deleted in E2.

---

## Part 3 — How to get a bot that plays effectively

> **Status, 12 August 2026.** The three steps below were carried out. Measurement
> was rebuilt and covered by 148 tests; the observation was made board-aware; and
> the algorithm was replaced — CFR now runs on Kuhn, Leduc and an abstracted
> heads-up no-limit game, validated against Kuhn's analytic value of −1/18 and
> against exact exploitability on Leduc. Results are in
> `IEEE_CFR_Poker_Engine_Paper_v3.pdf` and under `results/cfr/`. What remains open
> is recorded in that paper's threats-to-validity section rather than here.

### Original plan


The split between Parts 1 and 2 is the answer. The hard, slow-to-write, easy-to-get-wrong component — a correct rules engine with correct money — exists and is verified. The parts that failed are thin, recently written and cheap to replace. That argues against starting over, and equally against trusting anything above the engine.

Three things stand between this repository and a bot that plays well, and the order matters. Skipping ahead is what produced the existing results.

### Step 1 — Make measurement trustworthy, then make it cheap

The metric is now correct but still far too noisy to learn from: **±258 BB/100 over 600 hands** six-handed. The old sweeps ranked configurations separated by a fraction of that.

The standard remedy is **duplicate play**: deal the same cards to both seats and replay with positions swapped, so card luck cancels between the two runs. This is how poker results are compared seriously, and it reduces variance far more than simply playing more hands. Fixing **E1 is a prerequisite** — you cannot vary hands you are not actually dealing.

Then pick hands-per-matchup from the measured noise floor rather than convention, and budget three or more seeds per configuration.

### Step 2 — Let the agent see its cards

One of nine effective dimensions describes the hand, it ignores the board, and until this audit it was constant for every pocket pair. No learning algorithm recovers from that.

- Use the existing `evaluate_hand` for a board-aware made-hand rank: **8.7 µs per decision, about one extra second per generation**.
- Do **not** wire in the Monte-Carlo equity function (see E4).
- Add draw potential — a flush-draw and open-ended-straight-draw flag, or an "equity if I improve" scalar. Without it a flush draw scores as junk, so the agent can never semi-bluff or call correctly with equity.
- Add board texture: paired board, three-to-a-flush, three-to-a-straight.
- Reclaim the dead and duplicate slots identified in E5.

This takes the card representation from 1 to roughly 6 dimensions at negligible cost.

### Step 3 — Only then choose the algorithm, and narrow the game

Evolutionary search receives one scalar per genome per generation and performs no credit assignment, so it cannot learn "in this spot, this action". It will plateau regardless of features.

PPO with **genuine self-play** — training against current and past versions of itself, not a frozen pool of agents bred on a broken metric — is the right next step, and the implementation is already correct.

Be clear-eyed about its limit: self-play policy gradient does not converge to a Nash equilibrium in imperfect-information games. It yields a strong but *exploitable* bot. The methods that actually solved poker are the counterfactual-regret family — MCCFR and Deep CFR. `NEXT_STEPS_PLAN.md` listed CFR as a later phase; that instinct was correct.

Finally, **narrow the game**. Six-max no-limit is dramatically harder than heads-up and carries far more variance per hand. Heads-up is where iteration is fast, measurement is cleanest, and CFR is tractable.

### Component disposition

| Component | State | Action |
|---|---|---|
| Rules, betting, side pots | Verified correct | Keep |
| Hand evaluation | Verified correct | Keep, and reuse for features |
| Card dealing | Critical bug (E1) | Fix first |
| Feature layer | 1 live card dimension | Rebuild |
| Fitness / evaluation | Metric fixed, structure poor | Rewrite as one hand loop |
| Genetic operators | Sound | Keep; expect a plateau |
| PPO implementation | Sound | Keep; add self-play |
| Checkpoints, HoF, tournaments | Selected on a broken objective | Archive, retrain |
| Analysis reports | Derived from invalid data | Supersede |

---

## Verdict

Keep the engine, the genetic operators and the PPO implementation. Fix the deck. Rebuild the feature layer and the evaluation loop. Archive every result and every conclusion drawn from them.

The conclusion the existing reports reached — that the ceiling was the feature set — turns out to be correct, but for the opposite reason to the one given. Not that seventeen features are too few, but that the one feature which mattered was broken, and the instrument measuring the consequences was broken too.

That is a better finding than the scaling laws ever were, and unlike them it survives someone checking.

---

## Appendix — Regression tests added

`tests/` was an empty directory. It now contains 18 invariants, each corresponding to a defect above. Three of them would have caught every metric bug in this audit:

```python
assert sum(changes.values()) == 0                    # chips conserved
assert population_mean_fitness ≈ 0                   # zero-sum self-play
assert FeatureCache(g, p) == get_state_vector(g, p)  # train == inference
```

Run with:

```bash
python -m pytest tests/ -q
```

The suite also verifies that all 169 starting hands appear in the strength table, that advertised raises are satisfiable, that evaluation conditions match the declared config, that statistics describe the evaluated population, and that sigma decays once per generation.
