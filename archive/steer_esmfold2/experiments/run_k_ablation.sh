#!/usr/bin/env bash
# 7txw K/BPTT ablation: 3 experiments, one per GPU, in parallel.
#
#   Usage:  ./run_k_ablation.sh <gpu_E1> <gpu_E2> <gpu_E3>
#   e.g.    ./run_k_ablation.sh 2 3 4
#
# Each experiment = steer (16 sequential greedy runs, NO decode) -> decode_batch
# (batched 0->17 launch + decode of all 16 leaves). The three run concurrently on the
# three given GPUs; logs go to log_<tag>.txt. Override knobs via env: NS (n-samples,
# default 5), STEPS (diffusion steps, 200), LB (launch batch, 4).
#
# NOTE: assumes the 3 GPUs are already free. Free them first (kill + relaunch) per the
# plan; this script does NOT kill anything.
set -u
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/steer/experiments
G1=${1:?need gpu for E1}; G2=${2:?need gpu for E2}; G3=${3:?need gpu for E3}
NS=${NS:-5}; STEPS=${STEPS:-200}; LB=${LB:-4}

run_exp () {            # $1=gpu  $2=tag  $3..=extra steer flags
  local gpu=$1 tag=$2; shift 2
  echo "[$tag] steer on GPU $gpu ..."
  CUDA_VISIBLE_DEVICES=$gpu $PY $EXP/repel_proven.py --target 7txw --lever slm --M 5 \
      --lr 0.3 --init-frac 0.05 --loss mean --trials 1 --runs 16 --fused --chunk-size 0 --tag "$tag" "$@" \
    && echo "[$tag] decode on GPU $gpu ..." \
    && CUDA_VISIBLE_DEVICES=$gpu $PY $EXP/decode_batch.py --tag "$tag" --target 7txw \
      --n-samples "$NS" --n-steps "$STEPS" --launch-batch "$LB" --fused --chunk-size 0 \
    && echo "[$tag] DONE"
}

run_exp "$G1" 7txw_k3          --K 3                      > "$EXP/log_7txw_k3.txt"          2>&1 &
run_exp "$G2" 7txw_k6          --K 6                      > "$EXP/log_7txw_k6.txt"          2>&1 &
run_exp "$G3" 7txw_mk369_bptt3 --multik "3,6,9" --bptt 3  > "$EXP/log_7txw_mk369_bptt3.txt" 2>&1 &
wait
echo "=== all 3 experiments finished — run: $PY $EXP/collate_k_ablation.py ==="
