#!/bin/bash
# Reserve up to TARGET *fully-idle* GPUs with gpu_hold.py holders, and keep them.
# Two passes per scan: (1) count GPUs already running OUR processes (holder OR a real job),
# (2) grab fully-free GPUs only up to the deficit, so we never exceed TARGET total.
# A GPU running our real ConforNet job counts as "ours", so the watcher leaves it alone and
# re-holds it automatically once that job finishes (the GPU goes fully free again).
#
# Usage:  bash hold_gpus.sh [TARGET=4] [HOLD_GB=75]
set -u
TARGET=${1:-4}
HOLD_GB=${2:-75}
FREE_MB=600                              # "fully free" = less than this many MiB used
PY=/home/ebrahim/esm/.venv/bin/python
SCRIPT=/home/ebrahim/tts_abag/analysis/grad_steer/gpu_hold.py
ME=$(whoami)

echo "[hold_gpus] $(date '+%H:%M:%S') reserving up to $TARGET fully-idle GPUs (${HOLD_GB}GB each), user=$ME"
while true; do
  ours=0
  free_list=""
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
    pids=$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    mine=0
    for p in $pids; do
      [ -n "$p" ] && [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$ME" ] && mine=1
    done
    if [ "$mine" = 1 ]; then
      ours=$((ours + 1))
    else
      used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -lt "$FREE_MB" ] && free_list="$free_list $idx"
    fi
  done

  need=$((TARGET - ours))
  for idx in $free_list; do
    [ "$need" -le 0 ] && break
    CUDA_VISIBLE_DEVICES="$idx" nohup "$PY" "$SCRIPT" --gb "$HOLD_GB" > "/tmp/gpuhold_$idx.log" 2>&1 &
    echo "[hold_gpus] $(date '+%H:%M:%S') grabbed GPU $idx (holding ${HOLD_GB}GB)"
    need=$((need - 1))
    ours=$((ours + 1))
    sleep 5                             # let it allocate before the next scan
  done
  sleep 30
done
