#!/usr/bin/env bash
# Overnight THR x selection-metric sweep. Greedy-only clustering (fast), saves tiny per-pose
# assignment CSVs, then re-runs the offline metric evaluator after each THR completes so partial
# results are always available. Sequential = memory-safe on the shared box (one ~84 GB job at a
# time; the script's own watchdog kills at 90% system mem). Resumable: skips assignments already on disk.
set -u
cd /home/ebrahim/tts_abag
PY=.venv/bin/python
export NPROC="${NPROC:-64}"
export SKIP_MEDOID=1
export MAX_IT="${MAX_IT:-300}"                 # cap greedy tail so high-THR runs stay fast
export CLUSTER_OUT="results/csv_files/cluster_assign"
mkdir -p "$CLUSTER_OUT"
REPORT="steer/experiments/thr_sweep_report.txt"
LOG="steer/experiments/overnight_thr_sweep.out"

T7=("8ts0 A C" "8ts0 B C" "8a44 A D" "7yxu A C" "8a44 A C" "7zoz A C" "8vui B C")
# THR grid: ascending; big good/decoy basins resolve differently as the merge threshold tightens
THRS=(0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80)

echo "=== $(date '+%F %T') overnight sweep starting ===" | tee -a "$LOG"

# 1) wait for the currently-running full-medoid T7 sweep to finish (don't double up memory)
while pgrep -f dockq_cluster_T7.sh >/dev/null 2>&1; do
  echo "$(date '+%T') waiting for existing T7 sweep to finish..." | tee -a "$LOG"
  sleep 60
done
echo "$(date '+%T') existing sweep done (or none) — starting THR grid" | tee -a "$LOG"

for thr in "${THRS[@]}"; do
  export THR="$thr"
  for t in "${T7[@]}"; do
    set -- $t; cx=$1; c1=$2; c2=$3
    out="$CLUSTER_OUT/${cx}_${c1}${c2}_thr${thr}.csv"
    if [ -s "$out" ]; then
      echo "$(date '+%T') skip (exists): $cx $c1-$c2 THR=$thr" | tee -a "$LOG"; continue
    fi
    echo "=== $(date '+%T') CLUSTER $cx $c1-$c2 THR=$thr ===" | tee -a "$LOG"
    THR="$thr" $PY steer/experiments/dockq_cluster.py "$cx" "$c1" "$c2" \
        >> "steer/experiments/clust_${cx}_${c1}${c2}_thr${thr}.out" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then
      echo "!!! $(date '+%T') watchdog killed a run (mem) — pausing 5 min then continuing" | tee -a "$LOG"
      sleep 300
    fi
  done
  # re-evaluate all metrics across all THRs done so far
  echo "=== $(date '+%T') EVAL after THR=$thr ===" | tee -a "$LOG"
  { echo "==================== updated $(date '+%F %T') (through THR=$thr) ===================="
    $PY steer/experiments/eval_cluster_metrics.py
  } > "$REPORT" 2>&1
  echo "$(date '+%T') report updated -> $REPORT" | tee -a "$LOG"
done

echo "=== $(date '+%F %T') overnight sweep COMPLETE ===" | tee -a "$LOG"
