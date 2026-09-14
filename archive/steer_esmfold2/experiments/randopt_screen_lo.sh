#!/bin/bash
# randopt_screen_lo.sh — DOWN-EXTENSION of the Phase-1 screen: smaller sigmas {0.0005, 0.001}.
# Same 16 complexes, same 4 fixed directions (seeds 1-4), mask0/dropout0. Reuses the existing
# baseline (randopt_screen/baseline) — no baseline re-fold. 8 cells, 2 per GPU on 3/4/5/7.
# Finds whether the lift keeps rising below 0.002 (where the σ=0.002/0.005/0.01 sweep pointed).
#   bash steer/experiments/randopt_screen_lo.sh        # prepare only (write workers, no GPU)
#   bash steer/experiments/randopt_screen_lo.sh go     # dispatch into tmux gpu3/4/5/7
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
CX="8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo"
COMMON="--complexes $CX --seeds 0 --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log"
SIGMAS="0.0005 0.001"
declare -A GPU_SEED=( [3]=1 [4]=2 [5]=3 [7]=4 )

for g in 3 4 5 7; do
  seed=${GPU_SEED[$g]}
  w=$STATE/worker_lo_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"; echo "echo \"[g$g-lo] START \$(date)\""
    for sig in $SIGMAS; do
      echo "echo '[g$g] sigma=$sig seed=$seed'"
      echo "$PY $RF --run_name randopt_screen/s${sig}_seed${seed} --weight_perturb_sigma $sig --perturb_seed $seed $COMMON > $STATE/g${g}_s${sig}_seed${seed}.log 2>&1"
    done
    echo "echo \"[g$g-lo] DONE \$(date)\""
  } > "$w"; chmod +x "$w"
done
echo "Down-extension workers written (σ ∈ {0.0005,0.001}, seeds 1-4)."

if [ "${1:-}" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_lo_g$g.sh" Enter
    echo "  dispatched worker_lo_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED. Score after: bash steer/experiments/randopt_score.sh ; python steer/experiments/randopt_stats.py"
else
  echo "PREPARE-ONLY. To launch: bash steer/experiments/randopt_screen_lo.sh go"
fi
