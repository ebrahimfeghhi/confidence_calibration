#!/usr/bin/env bash
# Rebuild the 17 interfaces whose DockQ matrices were computed on the WRONG CHAIN PAIR.
#
# The scoring CSV names chains by their NATIVE letters; the model files use different letters and
# the `chain_mapping` column (ref -> model) translates. On multi-copy assemblies the mapping also
# CHANGES PER POSE, because OST maps the native onto whichever model copy fits. dockq_torch.py now
# selects the chain pair per pose. The other 31 interfaces have an identity mapping and are fine.
#
# Shared box: takes exactly ONE fully-idle GPU. Waits (polling every 30s) until one frees up, so
# we never contend with anyone else's job. ~60s per interface -> ~17 min total.
set -u
cd /home/ebrahim/tts_abag
PY=.venv-gpu/bin/python
LOG=steer/experiments/logs/rebuild17.log
mkdir -p steer/experiments/logs results/dockq_full

IFACES=(7uvs_AC 7uvs_BC 7z1x_AB 8a67_AD 8av2_AC 8be1_AC 8be3_AB 8oni_AE 8oni_AF 8oni_BE
        8oni_BF 8q3j_AB 8q3j_AC 8ux6_AF 8ux6_DF 8uky_AC 8uky_BC)

# a GPU is OURS to take only if it is completely idle: <500 MiB used AND <10% util
find_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 500 && $3 < 10 { print $1; exit }'
}

echo "=== $(date '+%F %T') rebuild ${#IFACES[@]} interfaces (per-pose chain mapping fix) ===" | tee -a "$LOG"
echo "waiting for ONE fully-idle GPU (polling every 30s)..." | tee -a "$LOG"

GPU=""
while [ -z "$GPU" ]; do
  GPU=$(find_free_gpu)
  if [ -z "$GPU" ]; then
    sleep 30
  fi
done
echo "$(date '+%T') GPU $GPU is free — claiming it" | tee -a "$LOG"

i=0
for f in "${IFACES[@]}"; do
  i=$((i + 1))
  echo "$(date '+%T') [$i/${#IFACES[@]}] $f -> GPU $GPU" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" $PY steer/experiments/dockq_torch.py --build "$f" \
      >> "$LOG" 2>&1
done

echo "=== $(date '+%F %T') REBUILD COMPLETE ===" | tee -a "$LOG"
ls results/dockq_full/*.npy 2>/dev/null | wc -l | xargs echo "matrices now present:" | tee -a "$LOG"
