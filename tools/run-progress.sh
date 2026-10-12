#!/usr/bin/env bash
# Progress for a long run that prints "  <done>/<total>  ..." lines --
# scripts/cfr/train_nolimit.py and scripts/slumbot_measure.py both do.
#
# Those lines carry done/total and an ETA derived from the job's own measured
# rate. What they cannot carry is whether the process is still alive and how long
# it has actually been running, so this adds those and shows the last few. Both
# halves exist because their absence cost a run: a 500,000-iteration attempt was
# killed at six hours having printed nothing to size it by and nothing to keep.
#
# A running mean in one of those lines is monitoring, not a result. Slumbot's
# interval is +/-374 mbb/hand at the full 10,000 hands and far wider before it;
# reading a favourable partial as an answer is the stopping-rule error.
#
#   tools/train-progress.sh <logfile> [pid]
#   tools/train-progress.sh <logfile> [pid] --watch
usage() { echo "usage: $0 <logfile> [pid] [--watch]" >&2; exit 2; }

log=$1; shift || usage
[ -f "$log" ] || { echo "no such log: $log" >&2; exit 2; }
pid=""; watch=""
for arg in "$@"; do
  case "$arg" in
    --watch) watch=1 ;;
    *) pid=$arg ;;
  esac
done

report() {
  echo "=== $(date '+%H:%M:%S %Z') — $(basename "$log")"
  if [ -n "$pid" ]; then
    if ps -p "$pid" >/dev/null 2>&1; then
      # ps, not pgrep: a pgrep pattern matches this script's own command line
      # and reports the job alive forever. That mistake cost four watchers here.
      echo "pid $pid  running, elapsed $(ps -p "$pid" -o etime= | tr -d ' ')"
    else
      echo "pid $pid  NOT running — finished, or died"
    fi
  fi
  # The progress lines, and whatever the trainer printed after them.
  grep -E '^ +[0-9,]+/[0-9,]+' "$log" | tail -3
  tail -3 "$log" | grep -vE '^ +[0-9,]+/[0-9,]+' | sed 's/^/  | /'
  echo
}

if [ -n "$watch" ]; then
  while :; do report; ps -p "${pid:-1}" >/dev/null 2>&1 || break; sleep 120; done
else
  report
fi
