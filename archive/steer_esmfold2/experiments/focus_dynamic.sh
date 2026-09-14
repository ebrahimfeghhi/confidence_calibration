#!/bin/bash
# focus_dynamic.sh — deepen the under-sampled ab-ag failures, but DROP any complex whose
# oracle-max DockQ >= SOLVE (0.80) and pour the freed GPU into the rest. Re-plans every round.
# Round-synchronized across GPUs; skip-existing makes it restart-safe.
# Off switch: touch steer/experiments/focusdyn_state/STOP
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RUN=esmfold2fast_tts
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_verify.csv"
SIX="9en2 8vzo 8gye 8vvl 8sgn 9jbq"
GPUS="3 7"
SOLVE=0.80
INCREMENT=10; MAX_SEED=2000
STATE=steer/experiments/focusdyn_state; mkdir -p "$STATE" results/csv_files; rm -f "$STATE/STOP"
LOG=steer/experiments/focus_dynamic.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# incremental OST scorer (skip-existing; self-stops on STOP)
( while [ ! -f "$STATE/STOP" ]; do $PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >>"$STATE/score.log" 2>&1; sleep 900; done ) &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT

# start seed = 1 + current max seed across the six (skip-existing covers laggards)
BASE=$($PY - "$PRED_DIR" $SIX <<'PYEOF'
import os,sys
pred=sys.argv[1]; six=sys.argv[2:]; mx=0
for c in six:
    d=os.path.join(pred,c)
    if os.path.isdir(d):
        for x in os.listdir(d):
            if x.startswith("seed_") and x[5:].isdigit(): mx=max(mx,int(x[5:]))
print(mx+1)
PYEOF
)
log "starting focus_dynamic on GPUs [$GPUS], SOLVE=$SOLVE, base seed=$BASE, six=[$SIX]"

k=0
while :; do
  [ -f "$STATE/STOP" ] && { log "STOP set — exiting"; break; }
  s0=$((BASE+INCREMENT*k)); s1=$((s0+INCREMENT-1))
  [ "$s1" -gt "$MAX_SEED" ] && { log "hit MAX_SEED"; break; }

  # who still needs work? (re-checked every round so 0.80-crossers drop out automatically)
  PLAN=$($PY steer/experiments/focus_plan.py "$OUT_CSV" "$SOLVE" $GPUS -- $SIX)
  [ -z "$PLAN" ] && { log "all six >= $SOLVE — DONE"; break; }
  log "round $k seeds $s0-$s1  plan: $(echo $PLAN | tr '\n' ' ')"

  pids=""
  while IFS= read -r line; do
    g="${line%%:*}"; cx="${line#*:}"
    [ -z "$cx" ] && continue
    CUDA_VISIBLE_DEVICES="$g" $PY run/esm/run_foldbench_abag.py --run_name "$RUN" \
      --complexes $cx --seeds $(seq "$s0" "$s1") --lm_mask_pct 0.1 --lm_dropout 0.3 \
      --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$STATE/gpu_$g.log" 2>&1 &
    pids="$pids $!"
  done <<< "$PLAN"
  wait $pids
  k=$((k+1))
done
kill $SP 2>/dev/null; trap - EXIT
log "focus_dynamic finished"
