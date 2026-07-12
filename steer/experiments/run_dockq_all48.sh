#!/usr/bin/env bash
# Build full Option-B DockQ matrices (all 5 diffusion samples, N ~ 5125, 26.3M pairs) for all 48
# confident-decoy interfaces -- 23 train + 25 test.
#
# Shared box, shared GPUs. NEVER disturb another user:
#   - GPUS below must be verified idle at launch (GPU 0 and 5 belong to other users right now)
#   - the driver re-checks each GPU is still ours before dispatching to it
#   - aborts entirely if system CPU memory crosses ABORT_PCT
#   - each worker peaks at ~4 GB VRAM of 96 GB, and ~2 GB CPU RAM
# Resumable: dockq_torch.py skips any interface whose dockq_<tag>.npy already exists.
#
# Timings land in results/dockq_full/timings.csv (per-interface, for the paper).
set -u
cd /home/ebrahim/tts_abag
PY=.venv-gpu/bin/python
LOGDIR=steer/experiments/logs
OUT=results/dockq_full
mkdir -p "$LOGDIR" "$OUT"

GPUS="${GPUS:-1 2 3 4 6 7}"          # verified idle; 0 and 5 are other users'
ABORT_PCT="${ABORT_PCT:-85}"
MAIN="$LOGDIR/dockq_all48.log"

# 48 confident-decoy interfaces (train + test), from results/csv_files/train_test_split.json
IFACES=(
  # --- TRAIN (23) ---
  7ua2_AC 7yv1_AB 7yv1_AC 7yxu_AC 7zoz_AB 7zoz_AC 8a44_AC 8a44_AD 8a67_AD 8dcm_AC 8dcm_BC
  8dtn_AB 8oni_AE 8oni_AF 8oni_BE 8oni_BF 8q3j_AB 8q3j_AC 8sgn_AC 8sgn_BC 8ts0_AC 8ts0_BC 8vui_BC
  # --- TEST (25) ---
  7uvs_AC 7uvs_BC 7z1x_AB 8aok_AB 8av2_AC 8be1_AC 8be3_AB 8ivx_AB 8ivx_AC 8oxx_AB 8oxx_AC
  8slb_AB 8slb_AC 8uky_AC 8uky_BC 8ux6_AF 8ux6_DF 8vvl_AB 8vvl_AC 8vzo_AC 8w84_AC 8w84_AD
  8w84_BC 8w84_BD 9jbq_AC
)

sysmem_pct() { awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%.0f", 100*(t-a)/t}' /proc/meminfo; }
gpu_free()   { # $1 = index. free if <500 MiB used and <10% util
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v g="$1" '$1==g { print ($2 < 500 && $3 < 10) ? "free" : "busy" }'
}

echo "=== $(date '+%F %T') full Option-B DockQ: ${#IFACES[@]} interfaces on GPUs [$GPUS] ===" | tee -a "$MAIN"
for g in $GPUS; do
  s=$(gpu_free "$g")
  echo "  GPU $g: $s" | tee -a "$MAIN"
  [ "$s" = "busy" ] && { echo "!!! GPU $g is NOT free — refusing to launch. Fix GPUS and rerun." | tee -a "$MAIN"; exit 1; }
done

i=0
declare -A PIDS   # gpu -> pid
for f in "${IFACES[@]}"; do
  if [ -s "$OUT/dockq_${f}.npy" ]; then
    echo "$(date '+%T') skip (done): $f" | tee -a "$MAIN"; continue
  fi
  pct=$(sysmem_pct)
  if [ "$pct" -ge "$ABORT_PCT" ]; then
    echo "!!! $(date '+%T') system memory ${pct}% >= ${ABORT_PCT}% — ABORTING before $f" | tee -a "$MAIN"
    break
  fi

  # wait for a free slot
  while :; do
    for g in $GPUS; do
      p="${PIDS[$g]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
        GPU="$g"; break 2
      fi
    done
    sleep 5
  done

  echo "$(date '+%T') [$((++i))/${#IFACES[@]}] $f -> GPU $GPU  [sys mem ${pct}%]" | tee -a "$MAIN"
  CUDA_VISIBLE_DEVICES="$GPU" $PY steer/experiments/dockq_torch.py --build "$f" \
      > "$LOGDIR/dq_${f}.log" 2>&1 &
  PIDS[$GPU]=$!
  sleep 2
done

wait
n=$(ls "$OUT"/dockq_*.npy 2>/dev/null | wc -l)
echo "=== $(date '+%F %T') COMPLETE — $n/${#IFACES[@]} matrices built ===" | tee -a "$MAIN"
du -sh "$OUT" | tee -a "$MAIN"
echo "--- timings (results/dockq_full/timings.csv) ---" | tee -a "$MAIN"
column -s, -t "$OUT/timings.csv" 2>/dev/null | tee -a "$MAIN"
