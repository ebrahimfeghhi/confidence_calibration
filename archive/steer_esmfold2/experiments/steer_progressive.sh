#!/bin/bash
# steer_progressive.sh <gpus> <targets_file> [budget_gpu_hours] [max_cycles]
#
# Breadth-first progressive steering: run cycle 1 (16 runs) for EVERY target, then cycle 2
# for every target, ... — each cycle resumes the previous via anchor checkpoints, so no work
# is redone. Within a cycle-round, targets are work-queued across the GPUs (expensive-first
# for load balance); a barrier between rounds guarantees cycle N-1 is checkpointed before N.
# Locked recipe: M5/K3/mean/slm + joint-z0/no-nudge/fused. Decode+score every cycle.
# Stops at budget_gpu_hours (default 100) or max_cycles (default 5). Early-stop: touch
# steer/experiments/steer_prog_state/STOP. Per-target checkpoints in steer_ckpts/<target>.ckpt.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
EXP=steer/experiments
GPUS=${1:?gpu list, e.g. "1 7 3"}; TARGETS_FILE=${2:?file with one target per line}
BUDGET_GPU_HOURS=${3:-100}; MAX_CYCLES=${4:-5}; CYCLE_RUNS=16
NGPU=$(echo $GPUS | wc -w)
CKPT_DIR=$EXP/steer_ckpts; mkdir -p "$CKPT_DIR"
STATE=$EXP/steer_prog_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG=$EXP/steer_progressive.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
echo 0 > "$STATE/total_sec"; BUDGET_SEC=$((BUDGET_GPU_HOURS*3600))

# targets, sorted expensive-first (L desc) for work-queue load balance
grep -vE '^\s*#|^\s*$' "$TARGETS_FILE" > "$STATE/targets_raw.txt"
$PY - "$STATE/targets_raw.txt" "$STATE/targets.txt" <<'PYEOF'
import sys; sys.path.insert(0,"."); sys.path.insert(0,"run/esm")
from paths import JSON_DIR; from run_foldbench_abag import load_complexes
Ls={n:sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences) for n,s in load_complexes(str(JSON_DIR))}
tg=[l.strip() for l in open(sys.argv[1]) if l.strip()]
tg.sort(key=lambda t: -Ls.get(t,0))     # expensive first
open(sys.argv[2],"w").write("\n".join(tg)+"\n")
PYEOF
NTGT=$(wc -l < "$STATE/targets.txt")
log "progressive steering: $NTGT targets, gpus=[$GPUS] ($NGPU), budget=${BUDGET_GPU_HOURS} GPU-h, up to $MAX_CYCLES cycles (${CYCLE_RUNS} runs/cycle)"

run_cycle(){ local g=$1 t=$2 cyc=$3 load=""
  [ "$cyc" -gt 1 ] && load="--load-ckpt $CKPT_DIR/${t}.ckpt"
  CUDA_VISIBLE_DEVICES=$g $PY "$EXP/repel_proven.py" \
    --target "$t" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
    --joint-z0 --lr-z0 0.01 --no-z0-nudge --trials 1 --runs $CYCLE_RUNS \
    --no-score --fused --chunk-size 32 --cycle-decode $CYCLE_RUNS \
    --n-samples 5 --n-steps 200 --launch-batch 2 \
    $load --save-ckpt "$CKPT_DIR/${t}.ckpt" --tag "${t}_prog_c${cyc}" \
    >>"$STATE/steer_${t}.log" 2>&1; }

pop(){ exec 9>"$STATE/q.lock"; flock 9; local x; x=$(head -1 "$STATE/queue.txt" 2>/dev/null)
       [ -n "$x" ] && sed -i '1d' "$STATE/queue.txt"; flock -u 9; echo "$x"; }

worker(){ local g=$1 cyc=$2 s=$(date +%s)
  while :; do [ -f "$STATE/STOP" ] && break; local t; t=$(pop); [ -z "$t" ] && break
    log "GPU $g -> $t cycle $cyc"; run_cycle "$g" "$t" "$cyc"; done
  echo $(( $(date +%s) - s )) >> "$STATE/round_gpusec"; }

ticker(){ while :; do sleep 120; [ -f "$STATE/STOP_TICK" ] && break
  local base rs=0; base=$(cat "$STATE/total_sec" 2>/dev/null || echo 0)
  [ -f "$STATE/round_start" ] && rs=$(( ($(date +%s) - $(cat "$STATE/round_start")) * NGPU ))
  log "  GPU-hours: ~$(awk "BEGIN{printf \"%.1f\", ($base+$rs)/3600}") / $BUDGET_GPU_HOURS"
done; }
ticker & TICK=$!
trap 'touch "$STATE/STOP" "$STATE/STOP_TICK" 2>/dev/null; kill $TICK 2>/dev/null' EXIT

for cyc in $(seq 1 "$MAX_CYCLES"); do
  [ -f "$STATE/STOP" ] && { log "STOP flag"; break; }
  total=$(cat "$STATE/total_sec")
  [ "$total" -ge "$BUDGET_SEC" ] && { log "budget reached ($(awk "BEGIN{printf \"%.1f\",$total/3600}") GPU-h)"; break; }
  log "=== CYCLE $cyc/$MAX_CYCLES  ($(awk "BEGIN{printf \"%.1f\",$total/3600}")/${BUDGET_GPU_HOURS} GPU-h used) ==="
  cp "$STATE/targets.txt" "$STATE/queue.txt"; : > "$STATE/round_gpusec"; date +%s > "$STATE/round_start"
  pids=""; for g in $GPUS; do worker "$g" "$cyc" & pids="$pids $!"; done
  wait $pids; rm -f "$STATE/round_start"
  round=$(awk '{s+=$1} END{print s+0}' "$STATE/round_gpusec")
  total=$(( $(cat "$STATE/total_sec") + round )); echo "$total" > "$STATE/total_sec"
  log "=== CYCLE $cyc done: +$(awk "BEGIN{printf \"%.1f\",$round/3600}") GPU-h -> $(awk "BEGIN{printf \"%.1f\",$total/3600}") GPU-h total ==="
done
touch "$STATE/STOP_TICK"; kill $TICK 2>/dev/null; trap - EXIT
log "ALL DONE. ~$(awk "BEGIN{printf \"%.1f\",$(cat "$STATE/total_sec")/3600}") GPU-h used. checkpoints in $CKPT_DIR/"
