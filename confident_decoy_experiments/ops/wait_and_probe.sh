#!/usr/bin/env bash
# Wait for a GENUINELY free GPU, then run the ConforNet memory probe on it, once.
#
# "Free" means zero compute processes, not merely free memory. Several cards here sit at 1.6 GB but
# 100% SM utilisation (john's RL jobs); taking one of those would contend for compute and slow
# someone else's run down. Memory headroom is not permission.
#
# Fires ONCE and exits. It does not chain into the 20-step optimisation -- the probe's peak-VRAM
# number is the thing that decides whether that run is even possible, so a human reads it first.
#
# Preflight (imports, input json, checkpoint) was done before arming this; the only thing that can
# have gone stale by firing time is GPU availability, which is what the loop checks.
set -uo pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$EXP/ops/confornet_probe.log
INTERVAL=${INTERVAL:-60}
MAX_HOURS=${MAX_HOURS:-24}
PDB=${PDB:-8che}
K=${K:-2}

mkdir -p "$EXP/ops"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Indices of GPUs with no compute processes at all.
free_gpus() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read -r idx uuid; do
    uuid=$(echo "$uuid" | tr -d ' ')
    if ! grep -qx "$uuid" <<<"$busy"; then echo "${idx// /}"; fi
  done
}

say "watcher armed: polling every ${INTERVAL}s for up to ${MAX_HOURS}h; target=${PDB} k=${K}"
say "current occupancy:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
ticks=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  gpu=$(free_gpus | head -1)
  if [ -n "${gpu:-}" ]; then
    sleep 5                                  # let a just-exited job release its context
    if ! free_gpus | grep -qx "$gpu"; then    # re-check: someone may have claimed it meanwhile
      say "GPU $gpu was taken during confirmation; continuing to wait"
      continue
    fi
    say "GPU $gpu is free -- launching probe"
    cd "$EXP" || exit 1
    $PY scripts/confornet_protenix.py --pdb "$PDB" --gpu "$gpu" --k "$K" --probe-memory 2>&1 \
      | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    say "probe exited rc=$rc on GPU $gpu"
    [ "$rc" -ne 0 ] && say "PROBE FAILED -- see traceback above; nothing else was launched"
    exit "$rc"
  fi
  ticks=$((ticks + 1))
  # heartbeat every ~30 min so the pane shows it is alive without spamming the log
  if [ $(( ticks % (1800 / INTERVAL) )) -eq 0 ]; then
    say "still waiting; occupancy: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | tr '\n' ';')"
  fi
  sleep "$INTERVAL"
done

say "gave up after ${MAX_HOURS}h with no free GPU; probe NOT run"
exit 2
