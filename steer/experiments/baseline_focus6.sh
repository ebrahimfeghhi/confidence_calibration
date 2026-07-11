#!/bin/bash
# baseline_focus6.sh — DEEPEN the 6 under-sampled ab-ag failures (each stuck at ~51 seeds)
# to find which are genuinely hard vs just under-sampled (like 8jyr/8urf, which the 1024-seed
# baseline rescued to 0.96/0.40). Same baseline condition: mask0.1 / dropout0.3, fused, 5 samples.
# Progressive seed rounds from 51 up, skip-existing, indefinitely until STOP.
# Off switch: touch steer/experiments/focus6_state/STOP
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RUN=esmfold2fast_tts
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_verify.csv"
STATE=steer/experiments/focus6_state; mkdir -p "$STATE" results/csv_files; rm -f "$STATE/STOP"
LOG=steer/experiments/baseline_focus6.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# cost-balanced 3+3 (cost ~ L^2.8): GPU3 = 8gye(1216)+9en2(600)+9jbq(573); GPU7 = 8vzo(1006)+8sgn(868)+8vvl(707)
GPU3_CX="8gye 9en2 9jbq"
GPU7_CX="8vzo 8sgn 8vvl"
SEED_START=51; INCREMENT=10; MAX_SEED=2000

worker(){ local g=$1 cx=$2 k=0
  log "GPU $g focusing on: $cx"
  while :; do
    [ -f "$STATE/STOP" ] && break
    local s0=$((SEED_START+INCREMENT*k)) s1=$((SEED_START+INCREMENT*k+INCREMENT-1))
    [ "$s1" -gt "$MAX_SEED" ] && break
    CUDA_VISIBLE_DEVICES="$g" $PY run/esm/run_foldbench_abag.py --run_name "$RUN" \
      --complexes $cx --seeds $(seq "$s0" "$s1") --lm_mask_pct 0.1 --lm_dropout 0.3 \
      --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$STATE/gpu_$g.log" 2>&1
    log "GPU $g finished seeds $s0-$s1 ($cx)"; k=$((k+1))
  done
  log "GPU $g worker exited"; }

# incremental OST scorer (skip-existing; self-stops on STOP)
( while [ ! -f "$STATE/STOP" ]; do $PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >>"$STATE/score.log" 2>&1; sleep 900; done ) &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT

worker 3 "$GPU3_CX" & p3=$!
worker 7 "$GPU7_CX" & p7=$!
wait $p3 $p7
kill $SP 2>/dev/null; trap - EXIT
log "baseline_focus6 DONE (STOP set or MAX_SEED reached)"
