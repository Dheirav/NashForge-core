#!/usr/bin/env bash
# The cross-tree gate: a cap-2 solve against the one-raise rung at the same
# depth, in the real game, check/call on a miss. This is the instrument for
# solver work since 17 September: a converged cap-2 solve cannot lose to the
# one-raise strategy, which is a legal strategy inside its game, so the number
# is how unconverged the solve is, in BB/100. The same-tree gate is nearly blind
# to the same change (+2.9 between solves 13 points apart on this one).
#   tools/xtree-gate.sh results/cfr/experiments/cap2_18bb_50m.pkl            # infers the rung
#   tools/xtree-gate.sh A.pkl results/cfr/ladder169l/nolimit_18bb.pkl out.json
#   PURIFY=all tools/xtree-gate.sh A.pkl ...     # gate the set as it plays when purified
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
first=$1
rung=$(basename "$first" | grep -oE '[0-9]+bb' | head -1)
second=${2:-$ROOT/results/cfr/ladder169l/nolimit_${rung}.pkl}
out=${3:-$ROOT/results/cfr/xtree/$(basename "${first%.pkl}")_vs_$(basename "${second%.pkl}").json}
mkdir -p "$(dirname "$out")"
"$ROOT/venv/bin/python" "$ROOT/scripts/cfr/play_pickles.py" "$first" "$second" \
  --hands "${HANDS:-40000}" --seeds ${SEEDS:-0 1 2} --on-miss call --purify "${PURIFY:-none}" --output "$out"
