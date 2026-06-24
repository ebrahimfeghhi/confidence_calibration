#!/bin/bash
# Wait for ONE fully-idle GPU, then run the ConforNet experiment on it (single GPU, fair use --
# not a multi-GPU hold). Exits when the experiment exits. Kill this session to stop waiting/running.
#
# Usage:  bash confornet_one.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FREE_MB=600                             # "fully free" = less than this many MiB used
PY=${ESM_PYTHON:-/home/ebrahim/esm/.venv/bin/python}
EXP="$HERE/steer_confornet.py"

echo "[confornet_one] $(date '+%H:%M:%S') waiting for ONE fully-idle GPU (all are busy right now)..."
while true; do
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt "$FREE_MB" ]; then
      echo "[confornet_one] $(date '+%H:%M:%S') GPU $idx is fully idle -> running steer_confornet"
      CUDA_VISIBLE_DEVICES="$idx" "$PY" "$EXP" --tag "g$idx" --seed-base "$((idx * 1000000))"
      echo "[confornet_one] $(date '+%H:%M:%S') steer_confornet (GPU $idx) exited"
      exit 0
    fi
  done
  sleep 60
done
