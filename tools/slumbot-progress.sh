#!/usr/bin/env bash
# Progress for a Slumbot run (scripts/slumbot_pilot.py or slumbot_measure.py).
#   tools/slumbot-progress.sh <logfile> [--watch]
# Hands done of total, elapsed since the log started, and an ETA from the
# measured rate of this run only. The win rate is deliberately not shown: at
# 1,000 hands it is about ±1,250 mbb/hand and reading it is the stopping-rule
# error. The miss rate is shown because it is the number the run is for.
# Without a path it follows the newest run log, so `--watch` alone works.
if [ -n "${1:-}" ] && [ "$1" != "--watch" ]; then log=$1; shift
else log=$(ls -t ~/pokerbot-scratch/slumbot/m1_*.log ~/pokerbot-scratch/slumbot/pilot*.log 2>/dev/null | grep -v _chain | grep -v _resume | head -1); fi
[ -n "$log" ] || { echo "usage: $0 [logfile] [--watch]; no run log found"; exit 1; }
watch=""; [ "${1:-}" = "--watch" ] && watch=1
report() {
  [ -f "$log" ] || { echo "no log yet at $log"; return; }
  local total; total=$(grep -oE '^[0-9]+ hands' "$log" | head -1 | grep -oE '[0-9]+')
  [ -z "$total" ] && total=$(grep -oE '[0-9,]+ hands against Slumbot' "$log" | head -1 | grep -oE '[0-9,]+' | tr -d ,)
  local line; line=$(grep -E '^ +[0-9,]+(/| hands)' "$log" | tail -1)
  local done; done=$(echo "$line" | grep -oE '[0-9,]+' | head -1 | tr -d ,)
  # Elapsed from the running process itself when there is one (the log's own
  # timestamps do not exist, and its birth time lied by two hours on 15 Sept
  # when a killed chain had created it earlier); the log's mtime otherwise.
  local elapsed pid
  pid=$(ps -eo pid,args | grep -E "^ *[0-9]+ venv/bin/python scripts/slumbot_(pilot|measure).py" | awk '{print $1}' | head -1)
  if [ -n "$pid" ]; then
    elapsed=$(ps -o etimes= -p "$pid" | tr -d ' ')
  else
    elapsed=$(( $(date +%s) - $(stat -c %Y "$log") ))
  fi
  echo "=== $(date '+%H:%M:%S %Z') — $(basename "$log")"
  if [ -n "$done" ] && [ "$done" -gt 0 ] && [ -n "$total" ]; then
    local rate eta; rate=$(awk -v d="$done" -v e="$elapsed" 'BEGIN{printf "%.2f", d/e}')
    eta=$(awk -v d="$done" -v t="$total" -v e="$elapsed" 'BEGIN{printf "%d", (t-d)*e/d/60}')
    # A resumed run counts hands from its checkpoint, so the rate derived here
    # overstates it; when the job prints its own rate and ETA, trust those.
    if echo "$line" | grep -q eta; then
      echo "  $done / $total hands, elapsed $((elapsed/60)) min this run; job's own rate and eta:"
    else
      echo "  $done / $total hands, elapsed $((elapsed/60)) min, $rate hands/s, eta $eta min"
    fi
    echo "  $line" | sed -E 's/\[[^]]*mbb\/hand so far\]//'
  else
    echo "  no hands counted yet (elapsed $((elapsed/60)) min)"
  fi
  grep -E 'miss rate|Traceback|Error|wrote' "$log" | tail -2 | sed 's/^/  | /'
  echo
}
if [ -n "$watch" ]; then while :; do report; sleep 120; done; else report; fi
