#!/usr/bin/env bash
# Progress for a multi-worker duel started with --progress-dir: matches done
# out of the total across workers, elapsed, and an ETA from the rate measured
# so far. It shows no win rate: a partial one is the stopping-rule error, and
# the duel prints the real one when it ends.
#   tools/duel-progress.sh [progress-dir] [--watch]
dir=$HOME/pokerbot-scratch/hist/duel_progress; watch=""
for arg in "$@"; do [ "$arg" = "--watch" ] && watch=1 || dir=$arg; done
report() {
  echo "=== $(TZ=Asia/Kolkata date '+%H:%M:%S IST') — $dir"
  [ -f "$dir/label" ] && echo "  $(cat "$dir/label")"
  shopt -s nullglob; files=("$dir"/worker_*); shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then echo "  no worker has finished a match yet (loading, or not started)"; echo; return; fi
  now=$(date +%s)
  read -r done total n start < <(cat "${files[@]}" | awk '{d += $1; t += $2; n++; if (s == "" || $3 < s) s = $3} END {print d, t, n, s}')
  workers=${WORKERS:-4}
  # Workers not reported yet hold the same share of the matches.
  [ "$n" -lt "$workers" ] && total=$(( total * workers / n ))
  el=$(( now - start ))
  line="  $done / $total matches, $n of $workers workers reporting, elapsed $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m"
  if [ "$done" -gt 0 ]; then
    left=$(( (total - done) * el / done ))
    line="$line, ETA $((left / 3600))h$(printf %02d $((left % 3600 / 60)))m ($(TZ=Asia/Kolkata date -d @$((now + left)) '+%H:%M IST')) from the measured rate"
  fi
  echo "$line"; echo
}
if [ -n "$watch" ]; then while true; do clear; report; sleep 60; done; else report; fi
