#!/usr/bin/env bash
# GPU5 queue: long-recycle 8a44 first, then the 7yxu nodrop trajectories.
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
while pgrep -f "run_long_recycle.sh" >/dev/null 2>&1 || \
      pgrep -f "N_cycle 30" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] long-recycle finished, starting 7yxu nodrop" | tee -a ops/gpu5_queue.log
./ops/run_traj_nodrop.sh 2>&1 | tee ops/traj_nodrop.log
