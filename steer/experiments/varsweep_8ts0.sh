#!/bin/bash
# varsweep_8ts0.sh — 20 steering runs to measure trial-to-trial variance of the steered good-vs-decoy
# ipTM margin: good(617/2) & decoy(46/3), init-seeds 0-4, K in {1,3}, both levers, fused.
# Round-robin across GPUs 3 4 5, each GPU runs its queue sequentially.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
SC=steer/experiments/steer_to_dist_iptm.py
D=/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad/varsweep
mkdir -p "$D"
GPUS=(3 4 5)
jobs=()
for K in 1 3; do
  for spec in "good 617 2" "decoy 46 3"; do
    set -- $spec; w=$1; seed=$2; samp=$3
    for i in 0 1 2 3 4; do jobs+=("$K:$w:$seed:$samp:$i"); done
  done
done
echo "total jobs: ${#jobs[@]}"
for gi in 0 1 2; do
  g=${GPUS[$gi]}
  (
    for ((j=gi; j<${#jobs[@]}; j+=3)); do
      IFS=':' read K w seed samp i <<< "${jobs[$j]}"
      echo "[GPU $g] $w K$K i$i $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$g $PY $SC --target 8ts0 --which $w --seed $seed --samp $samp \
        --K $K --init-seed $i --fused --max-steps 150 > "$D/8ts0_${w}_K${K}_i${i}.log" 2>&1
    done
    echo "[GPU $g] DONE $(date +%H:%M)"
  ) &
done
wait
echo "VARSWEEP DONE $(date)"
