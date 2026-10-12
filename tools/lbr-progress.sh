#!/usr/bin/env bash
# Progress of a chunked LBR lane (~/pokerbot-scratch/hist/laneLBR.sh): chunks
# done out of the total per condition, elapsed, and an ETA from the measured
# rate. With --pool, the pooled result per condition, which is only a result
# once every chunk of that condition is in.
#   tools/lbr-progress.sh [dir] [--watch | --pool]
dir=results/cfr/lbr_ladder_2026-09-25; mode=""
cd "$(dirname "$0")/.."
for arg in "$@"; do case "$arg" in --watch|--pool) mode=$arg ;; *) dir=$arg ;; esac; done
chunks=${CHUNKS:-16}
report() {
  echo "=== $(TZ=Asia/Kolkata date '+%H:%M:%S IST') — $dir"
  start=$(grep -m1 " start:" "${LANE_LOG:-$HOME/pokerbot-scratch/hist/laneLBR.log}" 2>/dev/null | cut -c1-19)
  total=0; done_all=0
  for c in ${CONDS:-70bb_on 70bb_between 100bb_on 100bb_between}; do
    n=$(ls "$dir"/${c}_[0-9][0-9].json 2>/dev/null | wc -l); total=$((total + chunks)); done_all=$((done_all + n))
    echo "  $c: $n / $chunks chunks"
  done
  if [ -n "$start" ]; then
    el=$(( $(date +%s) - $(date -d "$start" +%s) ))
    line="  $done_all / $total chunks, elapsed $((el / 60))m"
    [ "$done_all" -gt 0 ] && left=$(( (total - done_all) * el / done_all )) && \
      line="$line, ETA $(TZ=Asia/Kolkata date -d @$(( $(date +%s) + left )) '+%H:%M IST') from the measured rate"
    [ "$done_all" -eq 0 ] && line="$line, no chunk finished yet so no rate"
    echo "$line"
  fi
  echo
}
pool() {
  "$HOME/Code/PokerBot/venv/bin/python" - "$dir" "$chunks" <<'EOF'
import glob, json, math, os, sys
d, want = sys.argv[1], int(sys.argv[2])
for c in (os.environ.get("CONDS") or "70bb_on 70bb_between 100bb_on 100bb_between").split():
    rows = [next(iter(json.load(open(f)).values())) for f in sorted(glob.glob(os.path.join(d, f"{c}_[0-9][0-9].json")))]
    if not rows:
        print(f"{c}: no chunks"); continue
    n = sum(r["hands"] for r in rows)
    mean = sum(r["mean"] * r["hands"] for r in rows) / n
    se = math.sqrt(sum((r["stderr"] * r["hands"]) ** 2 for r in rows)) / n
    bb = 100 / 2                                   # chips a hand to BB/100 at a two-chip big blind
    flag = "" if len(rows) == want else f"  (PARTIAL: {len(rows)} of {want} chunks; not a result)"
    verdict = "proves exploitable" if mean - 2 * se > 0 else ("slack or near zero: proves nothing" if mean + 2 * se > 0 else "negative: the bound is slack")
    print(f"{c:14s} {bb * mean:+7.1f} ± {bb * se:4.1f} BB/100 over {n:,} hands, {rows[0]['path']}: {verdict}{flag}")
EOF
}
case "$mode" in
  --watch) while true; do clear; report; sleep 60; done ;;
  --pool) pool ;;
  *) report ;;
esac
