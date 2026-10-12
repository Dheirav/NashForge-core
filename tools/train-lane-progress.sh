#!/usr/bin/env bash
# Progress for a scratch training lane that writes run.log stamps and one train_<rung>.log per trainer
# (~/pokerbot-scratch/potmenu/run.sh and its kind). Each running trainer's line is its own "<done>/<total> ... eta",
# measured from its rate; a trainer still fitting its abstraction has no rate, so it gets no ETA. A lane that measures
# afterwards lists its result names in <dir>/expected; those are counted, with an ETA from the rate since the
# lane's "measure at" stamp once two have finished.
#   tools/train-lane-progress.sh [lane-dir] [--watch]
dir=~/pokerbot-scratch/potmenu; [ -n "${1:-}" ] && [ "$1" != "--watch" ] && { dir=$1; shift; }
report() {
  echo "=== $(date '+%a %H:%M:%S IST') $dir"
  sed 's/^/  /' "$dir/run.log" 2>/dev/null | tail -12
  [ -s "$dir/run.log" ] || echo "  not started (waiting for its start condition)"
  for f in "$dir"/train_*.log; do [ -f "$f" ] || continue; r=$(basename "$f" .log); r=${r#train_}
    # The latest stamp for the rung decides: a rung re-run after an earlier exit is running again.
    case "$(grep "train $r " "$dir/run.log" 2>/dev/null | tail -1)" in *exit*|*"already trained"*) continue;; esac
    el=$(( $(date +%s) - $(stat -c %W "$f" 2>/dev/null || stat -c %Y "$f") ))
    line=$(tr '\r' '\n' < "$f" | grep -E '^ +[0-9,]+/[0-9,]+' | tail -1 | sed 's/^ *//;s/  */ /g')
    printf '  %-6s RUN %3d min  %s\n' "$r" $((el/60)) "${line:-fitting the abstraction, no rate yet}"
    grep -qE 'Traceback|Error' "$f" && echo "  $r FAILED, see $f"
  done
  if [ -s "$dir/expected" ] && grep -q "measure at" "$dir/run.log" 2>/dev/null; then
    total=$(grep -c . "$dir/expected"); done_n=0
    while read -r n; do [ -s "$dir/$n" ] && done_n=$((done_n + 1)); done < "$dir/expected"
    t0=$(date -d "$(grep -m1 'measure at' "$dir/run.log" | cut -c1-19)" +%s); el=$(( $(date +%s) - t0 ))
    printf '  measurements %d/%d, %dh%02dm in' $done_n $total $((el/3600)) $((el%3600/60))
    if [ $done_n -ge 2 ] && [ $done_n -lt $total ]; then per=$(( el / done_n ))
      printf ', about %s at the measured rate (%d s each)\n' "$(date -d @$(( $(date +%s) + (total - done_n) * per )) '+%H:%M IST')" $per
    else echo; fi
  fi
  [ -e "$dir/done" ] && echo "  lane finished"; return 0; }
if [ "${1:-}" = "--watch" ]; then while :; do clear; report; [ -e "$dir/done" ] && break; sleep 30; done; else report; fi
