#!/usr/bin/env bash
# Look at what a solve actually does, not only at what it scores.
#
#   tools/strategy-sweep.sh results/cfr/ladder_a          # every rung
#   tools/strategy-sweep.sh results/cfr/experiments/cap2_70bb_t421_20m_warm.pkl
#
# The gate and the replay measure a set over tens of thousands of hands, so a
# single node that is wrong a quarter of the time in a spot that arises once a
# match is invisible to both. Such a node once
# put a quarter of a weak suited hand on an all-in facing a small open, in one
# rung of one solve. Other solves on the same tree put almost none there, so it
# was one solve's convergence and not the tree, and a look at the table would
# have found it in a minute.
#
# This flags, per rung: nodes where a sized raise is legal and the shove still
# carries a large share, the hand classes driving it, and preflop nodes where
# an all-in answer to a small bet is heavy. Nothing here is automatically a
# defect; it is the list to read before a set plays.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TARGET=${1:?usage: strategy-sweep.sh <ladder dir or pickle> [shove-share threshold]}
THRESHOLD=${2:-0.15}
exec venv/bin/python scripts/strategy_sweep.py "$TARGET" --threshold "$THRESHOLD"
