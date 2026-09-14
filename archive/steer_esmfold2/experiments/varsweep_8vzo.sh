#!/bin/bash
# varsweep_8vzo.sh — same variance sweep as 8ts0 but for 8vzo: good(166/3) & decoy(111/1),
# init-seeds 0-4, K in {1,3}, both levers, fused. Round-robin GPUs 3 4 5.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
SC=steer/experiments/steer_to_dist_iptm.py
D=/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad/varsweep
mkdir -p "$D"
GPUS=(3 4 5)
jobs=()
for K in 1 3; do
  for spec in "good 166 3" "decoy 111 1"; do
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
      echo "[GPU $g] 8vzo $w K$K i$i $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$g $PY $SC --target 8vzo --which $w --seed $seed --samp $samp \
        --K $K --init-seed $i --fused --max-steps 120 > "$D/8vzo_${w}_K${K}_i${i}.log" 2>&1
    done
  ) &
done
wait
echo "VARSWEEP_8vzo DONE $(date)"
