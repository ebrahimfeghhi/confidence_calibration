#!/bin/bash
# steer_efficiency_ablation.sh <gpus>
# 2x2 efficiency ablation on proven hitters (7txw, 8jyr, 8urf):
#   {bptt 0=full, 1=last-loop-only} x {M 5, 3}, everything else locked
#   (K3/mean/slm + joint-z0/no-nudge/fused). --stop-dockq 0.23 so each task records
#   GPU-hours-to-hit; capped at 80 runs (5 cycles) / 6h. 12 tasks work-queued across GPUs.
# Read results from campaign_summary.csv (best_dockq + gpu_hours + stop_reason by tag).
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
EXP=steer/experiments
GPUS=${1:?gpu list, e.g. "4 5"}; NGPU=$(echo $GPUS | wc -w)
STATE=$EXP/steer_abl_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG=$EXP/steer_efficiency_ablation.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

: > "$STATE/queue.txt"
for t in 7txw 8jyr 8urf; do for b in 0 1; do for m in 5 3; do echo "$t|$b|$m" >> "$STATE/queue.txt"; done; done; done
log "efficiency ablation: $(wc -l < "$STATE/queue.txt") tasks (3 targets x bptt{0,1} x M{5,3}), gpus=[$GPUS]"

pop(){ exec 9>"$STATE/q.lock"; flock 9; local x; x=$(head -1 "$STATE/queue.txt" 2>/dev/null)
       [ -n "$x" ] && sed -i '1d' "$STATE/queue.txt"; flock -u 9; echo "$x"; }

worker(){ local g=$1 s=$(date +%s)
  while :; do
    [ -f "$STATE/STOP" ] && break
    local task; task=$(pop); [ -z "$task" ] && break
    local t=${task%%|*}; local rest=${task#*|}; local b=${rest%%|*}; local m=${rest##*|}
    local tag="${t}_abl_b${b}_m${m}"
    log "GPU $g -> $t  bptt=$b M=$m"
    CUDA_VISIBLE_DEVICES=$g $PY "$EXP/repel_proven.py" \
      --target "$t" --lever slm --M "$m" --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --joint-z0 --lr-z0 0.01 --no-z0-nudge --bptt "$b" \
      --trials 1 --runs 80 --no-score --fused --chunk-size 32 \
      --cycle-decode 16 --stop-dockq 0.23 --time-budget 21600 \
      --n-samples 5 --n-steps 200 --launch-batch 2 --tag "$tag" \
      >>"$STATE/abl_${tag}.log" 2>&1
    log "GPU $g <- done $t bptt=$b M=$m"
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
TOTAL=$(awk '{s+=$1} END{print s+0}' "$STATE/gpusec.txt")
log "ALL DONE. ~$(awk "BEGIN{printf \"%.1f\", $TOTAL/3600}") GPU-h total."
echo "=== ablation results (tag | best_dockq | gpu_hours | stop_reason) ===" | tee -a "$LOG"
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="tag")T=i;if($i=="best_dockq")D=i;if($i=="gpu_hours")G=i;if($i=="stop_reason")S=i}}
         $T~/_abl_b/{printf "  %-22s dq=%-7s gpuh=%-6s %s\n",$T,$D,$G,$S}' "$EXP/campaign_summary.csv" 2>/dev/null | tee -a "$LOG"
