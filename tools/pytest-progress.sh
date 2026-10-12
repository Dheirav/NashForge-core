#!/usr/bin/env bash
# How far a background test run has got, and when it will finish.
#   tools/pytest-progress.sh [log] [--watch]
# Start the run so the elapsed time is real:
#   L=~/pokerbot-scratch/hist/pytest.log; date +%s > $L.start
#   nohup venv/bin/python -m pytest -q > $L 2>&1 &
#
# pytest -q writes a percentage at the end of each dot line, so the log holds
# the progress even though nothing prints it on its own. Collection is about
# half of this suite's 6m32s and reports nothing at all, so the first reading
# after it appears is what the rate is derived from: an ETA guessed before the
# first percentage would be wrong by the length of collection.
LOG=${1:-$HOME/pokerbot-scratch/hist/pytest.log}
[ "${2:-}" = --watch ] && WATCH=1 || WATCH=0
[ "${1:-}" = --watch ] && { WATCH=1; LOG=$HOME/pokerbot-scratch/hist/pytest.log; }
show() {
  [ -f "$LOG" ] || { echo "no log at $LOG"; return; }
  # The log's mtime moves as pytest writes to it, so elapsed read as zero;
  # the start stamp beside it is what the rate is derived from.
  started=$(cat "$LOG.start" 2>/dev/null || stat -c %Y "$LOG"); now=$(date +%s); elapsed=$((now - started))
  pct=$(tr -d '\n' < "$LOG" | grep -o '\[ *[0-9]\+%\]' | tail -1 | tr -dc 0-9)
  done_line=$(grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$LOG" | tail -1)
  if [ -z "$pct" ]; then
    echo "$(date '+%T')  collecting, ${elapsed}s elapsed, no percentage yet (no ETA until one appears)"
  else
    left=$(( elapsed * (100 - pct) / (pct > 0 ? pct : 1) ))
    echo "$(date '+%T')  ${pct}% of the suite, ${elapsed}s elapsed, about ${left}s left (rate measured from this run)"
  fi
  case "$done_line" in *passed*|*failed*|*error*) echo "  $done_line";; esac
}
if [ "$WATCH" = 1 ]; then while :; do show; grep -qE '[0-9]+ passed|[0-9]+ failed|error' "$LOG" 2>/dev/null && break; sleep 20; done
else show; fi
