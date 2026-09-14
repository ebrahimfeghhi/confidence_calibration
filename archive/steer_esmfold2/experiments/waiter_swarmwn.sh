#!/bin/bash
# waits for the swarm_vs_seq campaign to complete, then runs the weight-noise A/B:
# swarmwn 8jyr (GPU3) + swarmwn 8z8v (GPU4), matched to the campaign's s_lm-noise swarm.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
STATE=steer/experiments/wnab_state; mkdir -p "$STATE"; rm -f "$STATE/STOP"
LOG="$STATE/waiter.log"; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "armed: waiting for CAMPAIGN COMPLETE before weight-noise A/B"
while ! grep -q "CAMPAIGN COMPLETE" steer/experiments/campaign_state/campaign.log 2>/dev/null; do
  [ -f "$STATE/STOP" ] && { log "STOP before launch"; exit 0; }
  sleep 120
done
sleep 20
common="--lever slm --gen-cloud 16 --cloud-wnoise 0.005 --M 5 --K 3 --lr 0.3 --init-frac 0.05 --bptt 0 --trials 3 --runs 12 --no-score --fused --chunk-size 32 --n-samples 5"
tmux new-session -d -s wn_8jyr "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 $PY steer/experiments/repel_proven.py --target 8jyr $common --tag 8jyr_swarmwn 2>&1 | tee $STATE/lv_8jyr_swarmwn.out"
tmux new-session -d -s wn_8z8v "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 $PY steer/experiments/repel_proven.py --target 8z8v $common --tag 8z8v_swarmwn 2>&1 | tee $STATE/lv_8z8v_swarmwn.out"
log "campaign done -> launched swarmwn 8jyr (GPU3) + 8z8v (GPU4), sigma=0.005"
