#!/usr/bin/env bash
# Run a list of commands across cores, with progress.
#
# Scoring and training are embarrassingly parallel across seeds, rungs and
# matchups, and this machine has eight cores of which the solver uses one. That
# has been the largest unclaimed speedup in the project for some time.
#
# Process level rather than threads or multiprocessing, because agent closures
# cannot be pickled: `cfr_agent` and friends return local functions. The
# measurement scripts already expect this -- `endpoint_test_ppo.py --seed` and
# `bucket_sweep.py --seed-list` both produce per-seed JSON that `--merge` pools.
#
# ONLY for iteration-budgeted work. A wall-clock-budgeted ladder run in parallel
# reproduces the contaminated seed of 10 September, where contention meant one
# arm bought a tenth of the traversals at the same nominal budget. The ladders
# must stay sequential.
#
#   tools/parallel-run.sh --jobs 4 --log-dir ~/scratch/run < commands.txt
#   printf '%s\n' "cmd one" "cmd two" | tools/parallel-run.sh --jobs 3
set -uo pipefail

jobs=$(( $(nproc) / 2 ))
log_dir="${TMPDIR:-/tmp}/parallel-run-$$"
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) jobs="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$log_dir"

mapfile -t commands < <(grep -vE '^\s*(#|$)' || true)
total=${#commands[@]}
[ "$total" -gt 0 ] || { echo "no commands on stdin" >&2; exit 2; }

started=$(date +%s)
last_reported=0
echo "running $total commands, $jobs at a time, logs in $log_dir"

declare -a status
run_one() {
  local index=$1
  local log="$log_dir/job-$index.log"
  # Subshell: a command that calls `exit` would otherwise kill this wrapper
  # before it records anything, and the job would be reported as "no status"
  # rather than by its exit code.
  local code=0
  ( eval "${commands[$index]}" ) > "$log" 2>&1 || code=$?
  if [ "$code" -eq 0 ]; then
    echo "ok" > "$log_dir/job-$index.status"
  else
    echo "FAILED($code)" > "$log_dir/job-$index.status"
  fi
}

pids=()
for i in "${!commands[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do
    sleep 1
    done_now=$(ls "$log_dir"/*.status 2>/dev/null | wc -l)
    elapsed=$(( $(date +%s) - started ))
    if [ "$done_now" -gt 0 ] && [ "$done_now" != "$last_reported" ]; then
      # ETA from the measured rate, and divided by the number of workers:
      # elapsed*(remaining/done) is the SERIAL estimate, which overstated a
      # 6-job 2-worker run by 105 minutes against an actual 60 on 11 September.
      eta=$(( elapsed * (total - done_now) / done_now / jobs ))
      # Only when stdout is a terminal. Carriage returns into a log file leave
      # one enormous line, which is what that run produced.
      if [ -t 1 ]; then
        printf '\r  %d/%d done, %ds elapsed, eta %dm%02ds   ' \
               "$done_now" "$total" "$elapsed" $((eta/60)) $((eta%60))
      else
        printf '  %d/%d done, %ds elapsed, eta %dm%02ds\n' \
               "$done_now" "$total" "$elapsed" $((eta/60)) $((eta%60))
      fi
      last_reported=$done_now
    fi
  done
  run_one "$i" &
  pids+=($!)
done
wait

failed=0
for i in "${!commands[@]}"; do
  s=$(cat "$log_dir/job-$i.status" 2>/dev/null || echo "no status")
  [ "$s" = "ok" ] || { failed=$((failed+1)); echo ""; echo "  job $i $s: ${commands[$i]}"; echo "    log: $log_dir/job-$i.log"; }
done
elapsed=$(( $(date +%s) - started ))
printf '%s  %d/%d done in %dm%02ds, %d failed\n' \
       "$([ -t 1 ] && printf '\r')" "$total" "$total" $((elapsed/60)) $((elapsed%60)) "$failed"
exit $(( failed > 0 ))
