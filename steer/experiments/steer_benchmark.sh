#!/bin/bash
# steer_benchmark.sh <gpus> <targets_file> [per_target_seconds]
#
# Runs the LOCKED repulsion recipe on each target in <targets_file> (one per line),
# pipelined across <gpus> via a work-queue (each GPU pulls the next target when it frees,
# so load stays balanced regardless of per-target cost). Fair-comparison budget: each
# target runs to per_target_seconds wall-clock (set to the baseline's per-target GPU-hours
# tomorrow) with NO DockQ early-stop; decode+save every 16 runs for the DockQ-vs-compute
# curve. Early-stop: touch steer/experiments/steer_bench_state/STOP.
#
# Locked config (from the validated joint-z0 recipe + no-z0-nudge):
#   --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean
#   --joint-z0 --lr-z0 0.01 --no-z0-nudge --fused --chunk-size 32
#   --cycle-decode 16 --n-samples 5 --n-steps 200 --launch-batch 2
#
# TODO tomorrow: (1) targets_file = the re-validated mask0.1 failure list (+ do-no-harm
# controls), (2) per_target_seconds = baseline per-target GPU-hours (compute-match),
# (3) wire DockQ scoring of the cycle decodes (decode_dockq.py) as a background pass.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
EXP=steer/experiments
GPUS=${1:?gpu list, e.g. "1 7 3"}
TARGETS_FILE=${2:?file with one target per line}
BUDGET=${3:-21600}                       # per-target wall-seconds (default 6h; match baseline tomorrow)
RUNS=${RUNS:-512}                         # safety cap on runs per target
STATE=$EXP/steer_bench_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG=$EXP/steer_benchmark.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

grep -vE '^\s*#|^\s*$' "$TARGETS_FILE" > "$STATE/queue.txt"
log "LOCKED recipe M5/K3/mean/slm+joint-z0(no-nudge)/fused | targets=$(wc -l < "$STATE/queue.txt") gpus=[$GPUS] budget=${BUDGET}s/target"

pop_target(){                            # atomic pop of the first queued target
  exec 9>"$STATE/queue.lock"; flock 9
  local t; t=$(head -1 "$STATE/queue.txt" 2>/dev/null)
  [ -n "$t" ] && sed -i '1d' "$STATE/queue.txt"
  flock -u 9; echo "$t"; }

worker(){ local g=$1
  while :; do
    [ -f "$STATE/STOP" ] && break
    local t; t=$(pop_target); [ -z "$t" ] && break
    log "GPU $g -> steering $t"
    CUDA_VISIBLE_DEVICES=$g $PY "$EXP/repel_proven.py" \
      --target "$t" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --joint-z0 --lr-z0 0.01 --no-z0-nudge \
      --trials 1 --runs "$RUNS" --no-score --fused --chunk-size 32 \
      --cycle-decode 16 --time-budget "$BUDGET" \
      --n-samples 5 --n-steps 200 --launch-batch 2 --tag "${t}_bench" \
      >>"$STATE/steer_${t}.log" 2>&1
    log "GPU $g done $t (tag ${t}_bench)"
  done
  log "GPU $g worker exited (queue drained)"; }

pids=""
for g in $GPUS; do worker "$g" & pids="$pids $!"; done
wait
log "ALL TARGETS DONE"
