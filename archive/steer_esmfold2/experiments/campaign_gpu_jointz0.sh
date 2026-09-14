#!/usr/bin/env bash
# campaign_gpu_jointz0.sh <gpu> <struct1> [struct2] [struct3] ...
#
# Same recipe/loss as campaign_gpu.sh (K=3/M=5 label-free distogram-repulsion
# campaign), but with --joint-z0 on: also optimizes the initial pair state z0
# (trunc-normal draw) jointly with s_lm, using its own smaller lr (z0 is O(L^2)
# and blows up at the shared lr=0.3 -- see repel_<tag>_steps.csv drift_z0).
# Runs targets sequentially on ONE gpu; each stops on a DockQ>=0.23 hit or the
# time budget, then the next target starts. Tag suffix "_jz0" keeps outputs
# separate from any prior non-joint-z0 run of the same target.
#
#   Usage:  ./campaign_gpu_jointz0.sh 0 8jyr 8urf 8vzo
#
# Knobs via env: BUDGET (12h=43200), NS, STEPS, LB, RUNS, CYC, CS, LRZ0 (0.01).
set -u
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/steer/experiments
GPU=${1:?need gpu index}; shift
TARGETS=("$@")
[ ${#TARGETS[@]} -ge 1 ] || { echo "need at least one target"; exit 1; }
BUDGET=${BUDGET:-43200}
NS=${NS:-5}; STEPS=${STEPS:-200}; LB=${LB:-2}; RUNS=${RUNS:-512}; CYC=${CYC:-16}; CS=${CS:-32}
LRZ0=${LRZ0:-0.01}

run_campaign () {
  local s=$1 tag="${1}_k3m5_jz0"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s (tag $tag) START  budget=${BUDGET}s joint-z0 lr-z0=${LRZ0} ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$EXP/repel_proven.py" \
      --target "$s" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --joint-z0 --lr-z0 "$LRZ0" \
      --trials 1 --runs "$RUNS" --no-score --fused --chunk-size "$CS" \
      --cycle-decode "$CYC" --stop-dockq 0.23 --time-budget "$BUDGET" \
      --n-samples "$NS" --n-steps "$STEPS" --launch-batch "$LB" --tag "$tag" \
      2>&1 | tee "$EXP/log_${tag}.txt"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s DONE ==="
}

for s in "${TARGETS[@]}"; do
  run_campaign "$s"
done
echo "=== [$(date '+%F %T')] GPU $GPU: ALL campaigns finished (${TARGETS[*]}) ==="
