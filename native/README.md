# The native solver core

C++ port of the solver's hot path. Python keeps the evaluation harness, PPO, the
GUI, the scripts and the tests; this is the part where the time goes.

## Why a port at all

Measured 11 September by ablation, one iteration of 7.22 ms divides as:

    traversal floor (MCCFR + game logic, no card work)   40.5%
    equity                                               44.3%
      of which the rollout                               20.8% of total
        of which hand evaluation                         17.3% of total

So **2.47x is the hard ceiling** for every card-abstraction technique combined,
and a day of trying them confirmed it: precomputed equity tables, suit
isomorphism on the runtime cache, a rank-mask evaluator, and four smaller things
were each worth 1% or were regressions. The traversal floor is Python object
churn and no amount of Python removes it.

Measured, not assumed:

    hand evaluation      numba  9.6 M evals/sec   C++  43.6 M     4.5x
    40-sample rollout    numba  10.24 us          C++   2.15 us   4.8x

Projecting those onto the ablation puts a finished port near **15x**, and about
21x with a two-plus-two evaluator on top. Not the 50x I was asserting before
measuring it.

## Why C++ and nanobind

C++ because ChessBot is 97 C++ files and this is not the moment to also learn a
language. nanobind over pybind11 for roughly 4x faster compiles and 10x lower
call overhead; its one real casualty, multiple inheritance, is unused here.

Bindings rather than a standalone binary, which is the cheap route and the wrong
one: this project has had **two implementations of betting diverge twice**, the
20% raise-sizing error and the all-in non-termination. A separate trainer would
create a third. Bound, the C++ game serves `evaluation/benchmark.py` too, which
removes a divergence risk instead of adding one.

## What is exact and what is not

**Exact:** the evaluator, and later the game's transitions and payoffs. These are
deterministic, so they are pinned hand-for-hand against the Python.

**Statistical:** the Monte Carlo rollout. Reproducing numba's Mersenne Twister
stream in C++ would buy nothing, since a rollout is sampling; what must hold is
the distribution. `tests/test_native.py` checks it is unbiased rather than
identical.

## Building

    ./native/build.sh

Needs `nanobind` in the venv, cmake and ninja. The interpreter is passed
explicitly because CMake's own search finds the system Python and builds a
module that imports and then crashes on a subtly different ABI.

## Stages

1. **Evaluator and equity rollout.** Done. 180,000 hands across all ten hand
   classes, zero mismatches; rollout unbiased at +0.00044 +/- 0.00067.
2. **The no-limit game**, checked against `tests/test_betting_equivalence.py`,
   which exists precisely to catch two betting implementations disagreeing.
3. **MCCFR traversal and regret tables**, checked against Kuhn's -1/18 and exact
   Leduc exploitability.
4. **Swap in behind the existing interface.**
