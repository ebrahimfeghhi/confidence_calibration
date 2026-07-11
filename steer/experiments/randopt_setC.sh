#!/bin/bash
# randopt_setC.sh — same weight-perturbation experiment on a REPRESENTATIVE cross-section of 16 ab-ag
# complexes (disjoint from set A and set B), spanning the full masked ipTM-selection-quality range:
# 7 fails / 2 partial / 7 works. De-biases the A(fails)/B(works) extremes -> base rate of ipTM-
# selectable perturbation gain. Runs baseline + 100 apply-once draws @ σ=0.002 across GPUs 3/4/5/7.
#   baseline -> randopt_screen_C/baseline   |   draws -> randopt_mine_s002_C/<complex>/seed_N/
#   bash steer/experiments/randopt_setC.sh        # prepare only
#   bash steer/experiments/randopt_setC.sh go     # launch
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
CXC="8oxx 8yxi 8im0 8be1 9g2g 8a67 8sgn 8fdd 7yds 8qz3 8fas 8rp8 8q6r 8be2 8duz 8qf4"
COMMON="--complexes $CXC --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log"
declare -A RANGE=( [3]="1 25" [4]="26 50" [5]="51 75" [7]="76 100" )

for g in 3 4 5 7; do
  read lo hi <<< "${RANGE[$g]}"
  w=$STATE/worker_setC_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"; echo "echo \"[setC g$g] START \$(date)\""
    if [ "$g" = "3" ]; then
      echo "$PY $RF --run_name randopt_screen_C/baseline $COMMON --seeds 0 >> $STATE/setC_baseline.log 2>&1"
    fi
    echo "for N in \$(seq $lo $hi); do"
    echo "  $PY $RF --run_name randopt_mine_s002_C --weight_perturb_sigma 0.002 --perturb_seed \$N --seeds \$N $COMMON >> $STATE/setC_g$g.log 2>&1"
    echo "done"
    echo "echo \"[setC g$g] DONE \$(date)\""
  } > "$w"; chmod +x "$w"
done
echo "Set-C workers written (representative 16: baseline + 100 draws @ σ=0.002, 25/GPU)."

if [ "${1:-}" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_setC_g$g.sh" Enter
    echo "  dispatched worker_setC_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED set C."
else
  echo "PREPARE-ONLY. To launch: bash steer/experiments/randopt_setC.sh go"
fi
