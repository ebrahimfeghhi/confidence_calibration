#!/bin/bash
# steer_abl_quality.sh <gpus>
# Fixed-budget quality rerun (NO stop-dockq): does truncated-bptt / fewer-steps reach the
# SAME best DockQ as the full recipe? 8jyr + 8urf, 5 cycles (80 runs) each, best_dockq recorded.
# Configs: reference b0/M5, truncated b1/M5, truncated+fewer b1/M3, and b1/M3 with lr 0.5
# (test whether a bigger step compensates for fewer steps). Locked: K3/mean/slm+joint-z0/no-nudge/fused.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
EXP=steer/experiments
GPUS=${1:?gpu list}; NGPU=$(echo $GPUS | wc -w)
STATE=$EXP/steer_qabl_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG=$EXP/steer_abl_quality.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

: > "$STATE/queue.txt"
# bptt|M|lr : reference (full), truncated M5, and M3/M1 each uncompensated (lr0.3) vs displacement-matched
for t in 8jyr 8urf; do
  for cfg in "0|5|0.3" "1|5|0.3" "1|3|0.3" "1|3|0.5" "1|1|0.3" "1|1|1.5"; do echo "$t|$cfg" >> "$STATE/queue.txt"; done
done
log "quality rerun: $(wc -l < "$STATE/queue.txt") tasks (2 targets x 6 configs, fixed 5-cycle budget, no stop-dockq), gpus=[$GPUS]"

pop(){ exec 9>"$STATE/q.lock"; flock 9; local x; x=$(head -1 "$STATE/queue.txt" 2>/dev/null)
       [ -n "$x" ] && sed -i '1d' "$STATE/queue.txt"; flock -u 9; echo "$x"; }

worker(){ local g=$1 s=$(date +%s)
  while :; do
    [ -f "$STATE/STOP" ] && break
    local task; task=$(pop); [ -z "$task" ] && break
    local t=${task%%|*}; local r1=${task#*|}; local b=${r1%%|*}; local r2=${r1#*|}; local m=${r2%%|*}; local lr=${r2##*|}
    local tag="${t}_q_b${b}_m${m}_lr${lr}"
    log "GPU $g -> $t  bptt=$b M=$m lr=$lr"
    CUDA_VISIBLE_DEVICES=$g $PY "$EXP/repel_proven.py" \
      --target "$t" --lever slm --M "$m" --K 3 --lr "$lr" --init-frac 0.05 --loss mean \
      --joint-z0 --lr-z0 0.01 --no-z0-nudge --bptt "$b" \
      --trials 1 --runs 80 --no-score --fused --chunk-size 32 \
      --cycle-decode 16 --time-budget 21600 \
      --n-samples 5 --n-steps 200 --launch-batch 2 --tag "$tag" \
      >>"$STATE/q_${tag}.log" 2>&1
    log "GPU $g <- done $t bptt=$b M=$m lr=$lr"
  done
  echo $(( $(date +%s) - s )) >> "$STATE/gpusec.txt"; }

ticker(){ local start=$(date +%s); while :; do sleep 120; [ -f "$STATE/STOP_TICK" ] && break
  log "  ~$(awk "BEGIN{printf \"%.1f\", ($(date +%s)-$start)*$NGPU/3600}") GPU-h elapsed"; done; }
ticker & TICK=$!
trap 'touch "$STATE/STOP" "$STATE/STOP_TICK" 2>/dev/null; kill $TICK 2>/dev/null' EXIT

: > "$STATE/gpusec.txt"
pids=""; for g in $GPUS; do worker "$g" & pids="$pids $!"; done
wait $pids
touch "$STATE/STOP_TICK"; kill $TICK 2>/dev/null; trap - EXIT
log "ALL DONE (~$(awk '{s+=$1} END{printf "%.1f", s/3600}' "$STATE/gpusec.txt") GPU-h)."
echo "=== quality results (tag | best_dockq | gpu_hours) ===" | tee -a "$LOG"
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="tag")T=i;if($i=="best_dockq")D=i;if($i=="gpu_hours")G=i}} $T~/_q_b/{printf "  %-24s dq=%-7s gpuh=%s\n",$T,$D,$G}' "$EXP/campaign_summary.csv" 2>/dev/null | tee -a "$LOG"
