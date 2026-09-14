#!/bin/bash
# queue_var_8vzo.sh — waits for the 8jyr mean/kernel variance runs to finish, then launches the
# SAME experiment (10 trials x 1 cycle, mean on GPU4 / kernel on GPU5, --save-disto) on 8vzo.
# Off switch (before it launches): touch steer/experiments/varqueue_STOP
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
STOP=steer/experiments/varqueue_STOP
log(){ echo "[$(date '+%F %T')] $*" | tee -a steer/experiments/queue_var_8vzo.log; }
log "armed: waiting for 8jyr mean/kernel variance runs to finish, then 8vzo on 4/5"
while pgrep -f "8jyr_mean10" >/dev/null || pgrep -f "8jyr_kernel10" >/dev/null; do
  [ -f "$STOP" ] && { log "STOP set before launch — not starting 8vzo"; exit 0; }
  sleep 180
done
[ -f "$STOP" ] && { log "STOP set — not starting 8vzo"; exit 0; }
sleep 20   # let 4/5 settle
common="--target 8vzo --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --joint-z0 --lr-z0 0.01 --no-z0-nudge --bptt 0 --trials 10 --runs 16 --no-score --fused --chunk-size 32 --cycle-decode 16 --n-samples 5 --n-steps 200 --launch-batch 2 --save-disto"
tmux new-session -d -s vzo_mean   "CUDA_VISIBLE_DEVICES=4 $PY steer/experiments/repel_proven.py $common --loss mean   --tag 8vzo_mean10   2>&1 | tee steer/experiments/lv_mean_8vzo.out"
tmux new-session -d -s vzo_kernel "CUDA_VISIBLE_DEVICES=5 $PY steer/experiments/repel_proven.py $common --loss kernel --tag 8vzo_kernel10 2>&1 | tee steer/experiments/lv_kernel_8vzo.out"
log "8jyr finished -> launched 8vzo mean (GPU4) + kernel (GPU5), 10 trials x 16 runs, --save-disto"
