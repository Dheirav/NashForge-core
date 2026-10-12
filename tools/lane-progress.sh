#!/usr/bin/env bash
# Progress for a hist lane (~/pokerbot-scratch/hist/lane*.sh) that trains rungs
# in pairs and then sweeps, duels and runs a panel. The trainers print no rate
# while they build bucket tables (about 23 of each pair's 60 minutes on
# 24 September), so the training ETA comes from the wall time of the pairs that
# have finished, never from the iteration count. The duel and panel have no
# measured rate until they start, so they get a counter and no ETA.
#   tools/lane-progress.sh [lane-log] [--watch]
if [ -n "${1:-}" ] && [ "$1" != "--watch" ]; then lane=$1; shift
else lane=$(ls -t ~/pokerbot-scratch/hist/lane*.log 2>/dev/null | head -1); fi
[ -n "$lane" ] || { echo "no lane log found"; exit 1; }
watch=""; [ "${1:-}" = "--watch" ] && watch=1
dir=$(dirname "$lane")
report() {
  local now; now=$(date +%s)
  echo "=== $(date '+%H:%M:%S %Z') ($(TZ=Asia/Kolkata date '+%H:%M IST')) $(basename "$lane")"
  sed 's/^/  /' "$lane" | tail -20
  local start total done
  start=$(grep -m1 ' start$' "$lane" | cut -c1-19)
  total=$(grep -oE 'train [0-9]+bb \(' "$lane" | sort -u | wc -l)
  done=$(grep -cE 'train [0-9]+bb exit' "$lane")
  # Rungs the script will train but has not stamped yet are invisible here, so
  # the total is read from the script when it sits next to the log.
  local sh="${lane%.log}.sh"
  [ -f "$sh" ] && total=$(grep -oE 'run [0-9]+bb [0-9]+' "$sh" | wc -l)
  if [ -n "$start" ] && [ "$total" -gt 0 ]; then
    local el=$(( now - $(date -d "$start" +%s) ))
    printf '  rungs trained %d/%d, elapsed %dh%02dm' "$done" "$total" $((el/3600)) $((el%3600/60))
    # Pairs finish together, so the rate is per finished pair of rungs.
    local last; last=$(grep -E 'train [0-9]+bb exit' "$lane" | tail -1 | cut -c1-19)
    if [ "$done" -ge 2 ] && [ -n "$last" ]; then
      local span=$(( $(date -d "$last" +%s) - $(date -d "$start" +%s) ))
      local per=$(( span * 2 / done ))                  # seconds per pair
      local left=$(( (total - done + 1) / 2 ))           # pairs, the odd rung alone
      local eta=$(( $(date -d "$last" +%s) + left * per ))
      printf ', %d min a pair measured, training done about %s (%s)\n' $((per/60)) \
        "$(date -d @$eta '+%H:%M %Z')" "$(TZ=Asia/Kolkata date -d @$eta '+%H:%M IST')"
    else echo ", no pair finished yet so no rate"; fi
  fi
  # Running rungs: stamped "train <r> (" with no exit yet. Their log is the
  # newest by*_<r>.log, since lanes differ only in the prefix (by25, by60).
  for r in $(grep -oE 'train [0-9]+bb \(' "$lane" | awk '{print $2}' | sort -u); do
    grep -q "train $r exit" "$lane" && continue
    local f; f=$(ls -t "$dir"/by*_"$r".log 2>/dev/null | head -1)
    local line; line=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -E '^ +[0-9,]+/[0-9,]+' | tail -1 | sed 's/^ *//')
    [ -n "$line" ] || line="loading, no rate yet"
    echo "  $r RUN  $line"
  done
  # Duels print their own "<done>/<total> matches ... eta" counter, measured.
  for f in "$dir"/duel_*.log "$dir"/panel_*.log "$dir"/gate60_*.log "$dir"/rs_*.log; do
    # Older lanes' duel logs sit in the same directory; only this lane's count.
    [ -f "$f" ] && [ -n "$start" ] && [ "$(stat -c %Y "$f")" -ge "$(date -d "$start" +%s)" ] || continue
    echo "  $(basename "$f" .log): $(tr '\r' '\n' < "$f" | grep -vE '^\s*$' | tail -1 | cut -c1-110)"
  done
  echo
}
if [ -n "$watch" ]; then while true; do clear; report; sleep 60; done; else report; fi
