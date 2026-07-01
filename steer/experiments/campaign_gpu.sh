#!/usr/bin/env bash
# campaign_gpu.sh <gpu> <struct1> <struct2>
#
# Runs two label-free K=3 / M=5 distogram-repulsion CAMPAIGNS back-to-back on ONE GPU.
# Each campaign is a single long-lived process (repel_proven.py --cycle-decode): it
# accumulates anchors in memory, batch-decodes every CYC runs, scores ab-ag DockQ+ipTM,
# and stops when any sample reaches DockQ >= 0.23 (a rescue) OR after BUDGET seconds
# (12h). On stop, struct2 begins. "Maintain the 16 anchors, do another 16" is automatic
# — the anchor list just keeps growing inside the one process.
#
#   Usage:  CUDA GPU index, then the two targets (order = run order)
#     ./campaign_gpu.sh 0 8im0 8vzo
#
# Knobs via env: BUDGET (12h=43200), NS (n-samples 2), STEPS (diffusion steps 200),
#   LB (launch batch 2), RUNS (hard run cap 512), CYC (decode cadence 16), CS (chunk 32).
set -u
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/steer/experiments
GPU=${1:?need gpu index}; S1=${2:?need struct1}; S2=${3:?need struct2}
BUDGET=${BUDGET:-43200}
NS=${NS:-5}; STEPS=${STEPS:-200}; LB=${LB:-2}; RUNS=${RUNS:-512}; CYC=${CYC:-16}; CS=${CS:-32}

run_campaign () {
  local s=$1 tag="${1}_k3m5"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s (tag $tag) START  budget=${BUDGET}s ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$EXP/repel_proven.py" \
      --target "$s" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --trials 1 --runs "$RUNS" --no-score --fused --chunk-size "$CS" \
      --cycle-decode "$CYC" --stop-dockq 0.23 --time-budget "$BUDGET" \
      --n-samples "$NS" --n-steps "$STEPS" --launch-batch "$LB" --tag "$tag" \
      2>&1 | tee "$EXP/log_${tag}.txt"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s DONE ==="
}

run_campaign "$S1"
run_campaign "$S2"
echo "=== [$(date '+%F %T')] GPU $GPU: BOTH campaigns finished ($S1, $S2) ==="
