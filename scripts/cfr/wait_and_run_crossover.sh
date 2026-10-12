#!/usr/bin/env bash
#
# Wait for the machine to be genuinely free, then run the crossover experiment.
#
# The crossover experiment budgets training by wall-clock, so it measures the
# machine as much as the algorithm. Sharing a saturated box cost 27.5% of
# throughput when it was tried (725 iterations in 40s against 1,000 on a quiet
# machine), and — worse — contention that varies between the two arms being
# compared is a confound nothing in the results file would reveal.
#
# Two conditions are required, not one. Waiting only on the process would fire
# the moment it exits even if something else had picked up, and would be fooled
# if Linux recycled that PID onto an unrelated process.
#
#   1. the named process has exited
#   2. load average has stayed below the threshold across consecutive polls
#
# Usage:
#   scripts/cfr/wait_and_run_crossover.sh <pid-to-wait-for>

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

WAIT_PID="${1:-}"
POLL_SECONDS=600          # 10 minutes
QUIET_LOAD=4.0            # of 16 cores
QUIET_POLLS_REQUIRED=2    # sustained, not a momentary dip

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "watcher started; polling every $((POLL_SECONDS / 60)) minutes"

if [ -n "$WAIT_PID" ]; then
    log "waiting for pid $WAIT_PID to exit"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        log "  pid $WAIT_PID still running (load $(cut -d' ' -f1 /proc/loadavg))"
        sleep "$POLL_SECONDS"
    done
    log "pid $WAIT_PID has exited"
fi

log "waiting for load to settle below $QUIET_LOAD"
quiet=0
while [ "$quiet" -lt "$QUIET_POLLS_REQUIRED" ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk "BEGIN{exit !($load < $QUIET_LOAD)}"; then
        quiet=$((quiet + 1))
        log "  load $load — quiet ($quiet/$QUIET_POLLS_REQUIRED)"
    else
        [ "$quiet" -gt 0 ] && log "  load $load — busy again, resetting"
        quiet=0
    fi
    [ "$quiet" -lt "$QUIET_POLLS_REQUIRED" ] && sleep "$POLL_SECONDS"
done

log "machine is quiet; starting the crossover experiment"
exec env PYTHONPATH=. ./venv/bin/python -u scripts/cfr/crossover.py \
    --budgets 40 160 640 2560 \
    --seeds 3 \
    --lbr-hands 1000 \
    --rollout-samples 20 \
    --output results/cfr/crossover.json
