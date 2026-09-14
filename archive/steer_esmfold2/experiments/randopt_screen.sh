#!/bin/bash
# randopt_screen.sh — PHASE-1 weight-perturbation go/no-go screen (Neural Thickets / RandOpt).
#
# Folds 16 ab-ag complexes at mask0/dropout0 (input diversity OFF, so the ONLY thing varying vs
# baseline is the weights) under:
#   - 1 clean baseline (no perturbation)  -> deterministic default
#   - 12 weight perturbations = 3 sigmas {0.002,0.005,0.01} x 4 fixed noise directions (seeds 1-4)
#     w += sigma*N(0,1); same seed -> same DIRECTION across sigmas, so a table column is one
#     direction swept over magnitude.
# One perturbed model per run_name (applied once via --perturb_seed, reused across all 16 complexes).
# Sharded across GPUs 3/4/5/7: each GPU owns one direction's 3-sigma sweep; GPU3 also does baseline.
#
# Usage:
#   bash steer/experiments/randopt_screen.sh          # PREPARE only: write per-GPU worker scripts + print launch cmds (no GPU use)
#   bash steer/experiments/randopt_screen.sh go        # actually dispatch the workers into tmux sessions gpu3/4/5/7
# Monitor:  tmux attach -t gpu3   (or gpu4/gpu5/gpu7)   |  logs in steer/experiments/randopt_screen_state/
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
CX="8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo"
COMMON="--complexes $CX --seeds 0 --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log"
SIGMAS="0.002 0.005 0.01"
declare -A GPU_SEED=( [3]=1 [4]=2 [5]=3 [7]=4 )

# write one worker script per GPU (a sequence of run_foldbench invocations, each its own log)
for g in 3 4 5 7; do
  seed=${GPU_SEED[$g]}
  w=$STATE/worker_g$g.sh
  {
    echo "#!/bin/bash"
    echo "cd /home/ebrahim/tts_abag"
    echo "echo \"[g$g] START \$(date)\""
    if [ "$g" = "3" ]; then
      echo "echo '[g3] baseline (clean weights, mask0/dropout0)'"
      echo "$PY $RF --run_name randopt_screen/baseline $COMMON > $STATE/g3_baseline.log 2>&1"
    fi
    for sig in $SIGMAS; do
      echo "echo '[g$g] sigma=$sig seed=$seed'"
      echo "$PY $RF --run_name randopt_screen/s${sig}_seed${seed} --weight_perturb_sigma $sig --perturb_seed $seed $COMMON > $STATE/g${g}_s${sig}_seed${seed}.log 2>&1"
    done
    echo "echo \"[g$g] DONE \$(date)\""
  } > "$w"
  chmod +x "$w"
done
echo "Worker scripts written to $STATE/worker_g{3,4,5,7}.sh"
echo "Output predictions -> ~/esm/foldbench_predictions/randopt_screen/{baseline,s<sigma>_seed<n>}/"

if [ "${1:-}" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_g$g.sh" Enter
    echo "  dispatched worker_g$g -> tmux gpu$g (CUDA_VISIBLE_DEVICES=$g)"
  done
  echo "LAUNCHED. Monitor: tmux attach -t gpu3 | tail -f $STATE/*.log"
else
  echo
  echo "PREPARE-ONLY (no GPU used). To launch, run with 'go':"
  echo "  bash steer/experiments/randopt_screen.sh go"
fi
