#!/bin/bash
# Watch for FULLY-IDLE GPUs and run a continuous ConforNet diversity experiment on each, up to
# TARGET. Honest GPU use: each GPU does real steer_confornet.py work (logs only small CSVs + tiny
# affines). Two-pass scan so it never exceeds TARGET GPUs running our processes; if a job dies the
# GPU frees and gets relaunched automatically.
#
# Usage:  bash confornet_runner.sh [TARGET=4]
set -u
TARGET=${1:-4}
FREE_MB=600                              # "fully free" = less than this many MiB used
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/analysis/grad_steer/steer_confornet.py
ME=$(whoami)

echo "[confornet_runner] $(date '+%H:%M:%S') will run steer_confornet on up to $TARGET fully-idle GPUs, user=$ME"
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
    last=$(cat "/tmp/confornet_launch_$idx" 2>/dev/null || echo 0)
    [ $(( $(date +%s) - last )) -lt 180 ] && continue   # cooldown: don't relaunch a GPU that may have just crashed
    date +%s > "/tmp/confornet_launch_$idx"
    CUDA_VISIBLE_DEVICES="$idx" nohup "$PY" "$EXP" --tag "gpu$idx" --seed-base "$((idx * 1000000))" \
        > "/tmp/confornet_gpu$idx.log" 2>&1 &
    echo "[confornet_runner] $(date '+%H:%M:%S') launched steer_confornet on GPU $idx"
    need=$((need - 1))
    ours=$((ours + 1))
    sleep 8
  done
  sleep 30
done
