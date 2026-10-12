#!/usr/bin/env bash
# Progress for a whole ladder retrain: every rung's state in one screen.
#
#   tools/ladder-progress.sh ladder169            # once
#   tools/ladder-progress.sh ladder169 --watch    # every two minutes
#
# Each rung's trainer prints "<done>/<total> ... eta N min" from its own
# measured rate, and that is the only ETA shown: the rungs not yet started
# have no rate to derive one from, so they are listed as pending rather than
# guessed at. Rungs are read from the logs the tools/train-*.sh scripts write.
name=${1:?usage: $0 <ladder-name> [--watch]}; shift
watch=""; [ "${1:-}" = "--watch" ] && watch=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/cfr/$name"
RUNGS="cap2_100bb cap2_70bb cap2_50bb nolimit_100bb nolimit_70bb nolimit_50bb nolimit_35bb nolimit_25bb nolimit_18bb nolimit_12bb nolimit_8bb nolimit_5bb taper42_12bb taper42_18bb taper42_25bb taper42_35bb"

log_for() {
  case "$1" in
    cap2_*)    echo "$HOME/pokerbot-scratch/cap2/${name}_$1.log" ;;
    taper42_*) echo "$HOME/pokerbot-scratch/ladder/${name}_$1.log" ;;
    nolimit_*) echo "$HOME/pokerbot-scratch/ladder/${name}_${1#nolimit_}.log" ;;   # train-ladder.sh drops the prefix
  esac
}

report() {
  local done_n=0 total=0 running=0
  echo "=== $(date '+%H:%M:%S %Z') — $name"
  for rung in $RUNGS; do
    total=$((total+1))
    local log; log=$(log_for "$rung")
    if [ -f "$OUT/$rung.pkl" ]; then
      done_n=$((done_n+1))
      printf '  %-14s done   %s\n' "$rung" "$(grep -E 'ms/iteration' "$log" 2>/dev/null | tail -1 | sed 's/^ *//')"
    elif [ -f "$log" ]; then
      running=$((running+1))
      local line; line=$(grep -E '^ +[0-9,]+/[0-9,]+' "$log" | tail -1 | sed 's/^ *//')
      if [ -n "$line" ]; then printf '  %-14s RUN    %s\n' "$rung" "$line"
      else printf '  %-14s RUN    fitting the abstraction, no rate yet\n' "$rung"; fi
      if grep -qE 'Traceback|Error' "$log"; then printf '  %-14s        FAILED, see %s\n' "" "$log"; fi
    else
      printf '  %-14s pending\n' "$rung"
    fi
  done
  echo "  rungs: $done_n done, $running running, $((total-done_n-running)) pending, of $total"
  for lane in "$HOME/pokerbot-scratch/$name"/lane*.log; do
    [ -f "$lane" ] || continue
    printf '  %s: %s\n' "$(basename "$lane" .log)" "$(tail -1 "$lane")"
  done
  for gate in "$OUT"/gate_*.json; do
    [ -f "$gate" ] || continue
    printf '  GATE %s: ' "$(basename "$gate" .json)"
    "$ROOT/venv/bin/python" -c "import json,sys; d=json.load(open('$gate')); print(f\"{d['mean']:+.1f} ± {d['stderr']:.1f} BB/100 to the 169-class solver over {d['hands']:,} hands x {len(d['seeds'])} seeds\")"
  done
  echo
}

if [ -n "$watch" ]; then
  while :; do report; sleep 120; done
else
  report
fi
