#!/usr/bin/env bash
# campaign_gpu_fulldisto.sh <gpu> <struct1> [struct2] [struct3] ...
#
# Ablation: same recipe as campaign_gpu_jointz0.sh (K3/M5 label-free repulsion +
# joint z0/s_lm optimization), but the loss/anchors are the ENTIRE L×L distogram
# (--full-disto) instead of the targeted AB×C (+ intra-c) blocks. Tests whether
# targeting the interface (and sometimes intra-antigen) residues specifically is
# necessary, or whether an untargeted whole-distogram repulsion signal also works.
# Tag suffix "_fulldisto" keeps outputs separate from prior runs of the same target.
#
#   Usage:  ./campaign_gpu_fulldisto.sh 3 8jyr 8urf 8vvl
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
  local s=$1 tag="${1}_k3m5_fulldisto"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s (tag $tag) START  budget=${BUDGET}s full-disto joint-z0 lr-z0=${LRZ0} ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$EXP/repel_proven.py" \
      --target "$s" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --full-disto --joint-z0 --lr-z0 "$LRZ0" \
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
