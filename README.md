# NashForge-core

**The solver, the abstraction and the measurement toolkit behind NashForge: a heads-up no-limit Hold'em bot built on
counterfactual regret minimisation, and the instrument it was measured on against evolutionary search and PPO.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)

NashForge has won the [Chipzen](https://chipzen.ai) arena, a weekly heads-up no-limit Hold'em tournament for bots, in four
consecutive seasons: season 6 (15 to 20 September 2026), season 7, season 8 and season 9 (6 to 12 October 2026). The
platform keeps a public record of every rated hand with replays, so the results do not depend on this repository: the
season pages on chipzen.ai list the brackets and the matches, and anyone with a free login can open them.

This repository is the part of the project that can be read, run and checked by anyone. It is a **curated subset** of a
larger private repository, and the next section says exactly what is left out and why.

---

## What is here, and what is not

**Here:** the Hold'em rules engine, the game interface with Kuhn and Leduc, the solvers (vanilla CFR, external-sampling
MCCFR in C++ with four update rules, full-width CFR+, a river re-solver), the card and betting abstractions, exact
exploitability for small games, Local Best Response for large ones, the duplicate-deal panel and cross-tree gate, the
evolutionary and PPO agents the solver was compared with, a pygame viewer, and a Slumbot bridge with a small baseline
strategy, so a measurement against a third-party bot can be reproduced.

**Not here, on purpose:** the arena player and its reads of individual opponents, the scouting and copy tooling, the
solved strategies of the sets that play (only two small baseline strategies are included), and notes about individual
bots. A bot that is entered in a live competition gains little from being open and can lose a lot: a fixed strategy can
be best-responded to, and a published rule can be baited. Material about specific opponents is other people's work, so
only generalised findings belong in a public repository.

**If you need to verify the rest** (an employer, an organiser, someone reviewing the project), read access to the full
private repository can be given on request, through the GitHub profile [Dheirav](https://github.com/Dheirav).

---

## The comparison

The project started as a comparison. Three families of agent, CFR, evolutionary search and PPO, were trained on the same
abstracted game and measured on the same panel. One panel, 40,000 hands, every family against the same opponents, in big
blinds per 100 hands (`results/comparison/phase4_native.json`):

| family | training hands | vs random | vs always-call | vs the solver |
|---|---|---|---|---|
| CFR solver | | +258.7 | +647.3 | |
| evolutionary search, 50 generations | 36,000,000 | +202.7 | -0.2 | **-211.8** |
| PPO | 500,000 | +191.2 | +372.6 | **-84.6** |
| PPO | 8,000,000 | +226.8 | +329.1 | **-74.9** |

Both learned families lose to the solver, PPO by about 75 BB/100 at every training budget and evolution by about 200, and
evolution spent seventy times PPO's hands to finish further behind. The result was re-measured against an independently
implemented solver (the C++ core, seeded differently from the Python one) and every number moved by less than its own
interval. This is not because the learned agents are weak against ordinary opponents; both beat a random player by 200
BB/100. It is that self-play policy gradient and fitness-driven search do not converge to an equilibrium in an
imperfect-information game, while CFR does, so the solver finds the holes they leave.

---

## Why anyone should believe it

CFR is easy to write and hard to write correctly, and a solver that has converged to the wrong thing looks exactly like
one that has converged. The checks, in order of strength:

- **Kuhn poker** has an analytically known value of -1/18 to the first player, and the solver reproduces it.
- **Leduc Hold'em** is small enough to traverse exactly, so exploitability is computed, not estimated, and falls toward zero.
- **A full-width CFR+ solve** of the small no-limit games ties the sampled solver on the same tree, which says the
  sampling is not the error.
- **Self-play reads zero.** A solver against itself at its own stack, on the gate, with zero misses, is in the test suite. Every measuring instrument is pinned to read zero on a mirror, because three of them had
  defects that told confident wrong stories about the solver before that test existed.
- **The arena.** A public record of every rated hand, with replays, against bots built by other people.

Exploitability is measured as a lower bound only. Local Best Response is a greedy exploiter with a blind spot: it rated a
discounted-CFR solve as harder to exploit while that solve lost 24 BB/100 head to head, so no change is adopted on LBR
alone. Two further lessons are written into the instruments: a result read from a few thousand hands is mostly noise, and
a gap between two single training runs can be the run and not the change, so a go decision is replicated on a second seed.
---

## How the solver works

**The solver.** External-sampling Monte Carlo CFR with linear discounting, warm start from a smaller tree (mapped by
action, so a tree that inserts a size keeps what was learnt) and regret-based pruning, written in C++ (`native/`) and
bound to Python with nanobind. Discounted CFR was tried in four settings and loses to linear by 22 BB/100 or more. It
runs at about half a millisecond per iteration on one core.

**The abstraction.** 169 preflop hand classes (every distinct starting hand). Postflop, hands are clustered by k-means
over sampled equity against a random hand, with the board's flush and straight texture folded in; an equity-histogram
feature clustered by earth mover's distance (Johanson et al., 2013) is also implemented. Betting is a tree of a few raise
sizes per raise, with schedules that can taper so that later raises get fewer sizes, which is what makes a third raise
affordable, and a schedule can differ by street.

**The ladder.** One solver per stack depth, from 5 to 100 big blinds, because the right strategy at 100 big blinds is
nothing like the right strategy at 12. The player reads the effective stack, picks the rung, and switches as the blinds
climb.

**The instruments.** A solver is never believed on its own training curve. Each new one goes through a cross-tree gate
(against a simpler solver in the real game, two seeds), a duel (two whole bots over hundreds of arena-style matches on
the arena's blind schedule), a replay against the logged match record, and a burst of rated matches read by a
decomposition (by depth, by which solver decided each hand) with a correction for the luck of called all-ins, and by a
verdict that stays valid however often it is read. The gate and Local Best Response are in this repository; the duel
runner and the arena-side decomposition tools are not.

---

## Getting started

```bash
python3.12 -m venv venv
venv/bin/pip install numpy numba torch pygame websockets requests
native/build.sh                                   # needs cmake, ninja, nanobind; installs by rename
venv/bin/python -m pytest -q                      # the test suite
```

```bash
venv/bin/python scripts/preflight_training.py     # must pass before any training run
venv/bin/python scripts/cfr/train_nolimit.py --iterations 20000000 --raise-cap 4 2 1 \
    --stack 200 --big-blind 2 --texture --preflop-buckets 169 --update-rule linear \
    --threads 2 --output results/cfr/experiments/cap2_100bb.pkl
tools/xtree-gate.sh results/cfr/experiments/cap2_100bb.pkl      # the cross-tree gate
venv/bin/python -m gui.main                                     # play the solver yourself
```

Numba is on the hot path of every Python-side evaluation; the native module is what trains. The build installs by rename
so a rebuild under a running training is safe.

---

## Layout

| directory | what is there |
|---|---|
| `engine/` | Hold'em rules, betting, side pots, hand evaluation. Verified by the audit and the one part everything else is built on. |
| `games/` | The traversable game interface, Kuhn, Leduc and abstracted no-limit. CFR traverses a game rather than playing it. |
| `abstraction/` | Card classes (preflop, k-means over equity or equity histograms postflop, board texture) and the betting schedules. |
| `cfr/` | The Python solvers (vanilla, MCCFR, four update rules), exact exploitability for small games, full-width CFR+, the river re-solver, the flat strategy tables, Local Best Response. |
| `native/` | The C++ core: MCCFR, the abstract game, the hand evaluator, equity and histogram sampling. `build.sh` builds it. |
| `slumbot/` | The Slumbot bridge and player, for measuring against a third-party bot. |
| `chipzen/` | The generic parts only: the API client and the bet-size translation. |
| `evaluation/` | The benchmark loop, the duplicate-hand panel, the cross-tree gate. |
| `scripts/`, `tools/` | Entry points (training, gates, measurement) and the shell wrappers around them. |
| `training/`, `rl/` | Evolutionary search and PPO, both measured in the comparison above. |
| `results/` | Measurements as JSON, one file per question, and the two small baseline strategies. |
| `tests/` | The test suite. Was an empty directory before the audit. |
| `docs/` | The training plan, the retrain plan, the solver work notes and the method research. |

`CODEBASE_AUDIT.md` (12 August 2026) is authoritative on what in this repository is and is not trustworthy.

---

## History: the audit

This repository once trained poker agents with an evolutionary algorithm and published scaling laws, tournament rankings
and hyperparameter findings from them. All of it was wrong. `CODEBASE_AUDIT.md` established, by executing the code rather
than reading it, that the fitness function scored the wrong player and the deck re-dealt the same two hands every hand: an
untrained random network scored +451 BB/100 under that metric, and scores approximately zero under the corrected one, as it
must. The rules engine survived the audit intact and was kept. Everything built on the broken metric was removed. No number
from before the audit is quoted anywhere in the current documents.

What replaced it is CFR, because it provably converges toward a Nash equilibrium in imperfect-information games and, more
to the point, comes with ways to check that it has.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Read it, run it, learn from it, build on it and share what you build, for any
noncommercial purpose; entering it or a derivative of it in a prize competition is commercial use and is not licensed.
The solved strategies of the playing sets are not distributed at all.
