#!/bin/bash
# 8jyr_swarmwn already launched on GPU5; this waits for the campaign to finish, then runs swarmwn 8z8v on GPU3.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
S=steer/experiments/wnab_state; log(){ echo "[$(date '+%F %T')] $*" >> "$S/waiter2.log"; }
log "armed: waiting for CAMPAIGN COMPLETE, then swarmwn 8z8v on GPU3"
while ! grep -q "CAMPAIGN COMPLETE" steer/experiments/campaign_state/campaign.log 2>/dev/null; do
  [ -f "$S/STOP2" ] && { log "STOP2"; exit 0; }
  sleep 120
done
sleep 20
common="--lever slm --gen-cloud 16 --cloud-wnoise 0.005 --M 5 --K 3 --lr 0.3 --init-frac 0.05 --bptt 0 --trials 3 --runs 12 --no-score --fused --chunk-size 32 --n-samples 5"
tmux new-session -d -s wn_8z8v "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 $PY steer/experiments/repel_proven.py --target 8z8v $common --tag 8z8v_swarmwn 2>&1 | tee $S/lv_8z8v_swarmwn.out"
log "launched swarmwn 8z8v on GPU3"
