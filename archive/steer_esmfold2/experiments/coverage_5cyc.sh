#!/bin/bash
# coverage_5cyc.sh <gpus>
# Proven default recipe (mean, joint-z0 + no-z0-nudge, M5, K3, lr0.3, full bptt, fused),
# 5 cycles (80 runs) per target, NO stop-dockq -> records best_dockq at 5-cycle budget
# (the coverage number). 5 targets work-queued across the given GPUs. 8vvl adds --intra-c
# (its antigen misfolds; intra-antigen block is what rescued it). 8im0 intentionally dropped.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
EXP=steer/experiments
GPUS=${1:?gpu list, e.g. "4 5"}; NGPU=$(echo $GPUS | wc -w)
STATE=$EXP/cov5_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG=$EXP/coverage_5cyc.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

: > "$STATE/queue.txt"
# target|extra-flags
for spec in "8jyr|" "8urf|" "8vzo|" "8z8v|" "8vvl|--intra-c"; do echo "$spec" >> "$STATE/queue.txt"; done
log "coverage run: $(wc -l < "$STATE/queue.txt") targets x 5 cycles (80 runs), default recipe, no stop-dockq, gpus=[$GPUS]"

pop(){ exec 9>"$STATE/q.lock"; flock 9; local x; x=$(head -1 "$STATE/queue.txt" 2>/dev/null)
       [ -n "$x" ] && sed -i '1d' "$STATE/queue.txt"; flock -u 9; echo "$x"; }

worker(){ local g=$1 s=$(date +%s)
  while :; do
    [ -f "$STATE/STOP" ] && break
    local task; task=$(pop); [ -z "$task" ] && break
    local t=${task%%|*}; local extra=${task#*|}
    local tag="${t}_5cyc"
    log "GPU $g -> $t  (extra: ${extra:-none})"
    CUDA_VISIBLE_DEVICES=$g $PY "$EXP/repel_proven.py" \
      --target "$t" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --joint-z0 --lr-z0 0.01 --no-z0-nudge --bptt 0 $extra \
      --trials 1 --runs 80 --no-score --fused --chunk-size 32 --cycle-decode 16 \
      --n-samples 5 --n-steps 200 --launch-batch 2 --tag "$tag" \
      >>"$STATE/cov_${tag}.log" 2>&1
    log "GPU $g <- done $t"
  done
  echo $(( $(date +%s) - s )) >> "$STATE/gpusec.txt"; }

ticker(){ local start=$(date +%s); while :; do sleep 300; [ -f "$STATE/STOP_TICK" ] && break
  log "  ~$(awk "BEGIN{printf \"%.1f\", ($(date +%s)-$start)*$NGPU/3600}") GPU-h elapsed; remaining targets: $(wc -l < "$STATE/queue.txt" 2>/dev/null)"; done; }
ticker & TICK=$!
trap 'touch "$STATE/STOP" "$STATE/STOP_TICK" 2>/dev/null; kill $TICK 2>/dev/null' EXIT

: > "$STATE/gpusec.txt"
pids=""; for g in $GPUS; do worker "$g" & pids="$pids $!"; done
wait $pids
touch "$STATE/STOP_TICK"; kill $TICK 2>/dev/null; trap - EXIT
log "ALL DONE (~$(awk '{s+=$1} END{printf "%.1f", s/3600}' "$STATE/gpusec.txt") GPU-h)."
echo "=== coverage results (tag | best_dockq | gpu_hours) ===" | tee -a "$LOG"
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="tag")T=i;if($i=="best_dockq")D=i;if($i=="gpu_hours")G=i}} $T~/_5cyc/{printf "  %-18s dq=%-7s gpuh=%s\n",$T,$D,$G}' "$EXP/campaign_summary.csv" 2>/dev/null | tee -a "$LOG"
