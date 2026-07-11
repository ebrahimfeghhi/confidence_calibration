#!/bin/bash
# randopt_mine_fast.sh <sigma> [go] — apply-once weight-perturbation mining at a given sigma.
# 100 draws (seeds 1-100) across GPUs 3/4/5/7, mask0/dropout0. Each draw = one run_foldbench with
# --perturb_seed N (perturbation applied ONCE) --seeds N, writing to randopt_mine_s<tag>/<complex>/
# seed_N/ (skip-existing resumes). tag = sigma without the leading "0." (0.001->001, 0.003->003).
#   bash steer/experiments/randopt_mine_fast.sh 0.001        # prepare only
#   bash steer/experiments/randopt_mine_fast.sh 0.001 go     # dispatch to tmux gpu3/4/5/7
set -u
cd /home/ebrahim/tts_abag
SIGMA=${1:?usage: randopt_mine_fast.sh <sigma> [go]}
GO=${2:-}
tag=${SIGMA#0.}
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
CX="8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo"
RUN=randopt_mine_s${tag}
COMMON="--complexes $CX --weight_perturb_sigma $SIGMA --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log --run_name $RUN"
declare -A RANGE=( [3]="1 25" [4]="26 50" [5]="51 75" [7]="76 100" )   # 100 draws / 4 GPUs

for g in 3 4 5 7; do
  read lo hi <<< "${RANGE[$g]}"
  w=$STATE/worker_fast_${tag}_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"
    echo "echo \"[fast $tag g$g] START \$(date) draws $lo-$hi\""
    echo "for N in \$(seq $lo $hi); do"
    echo "  $PY $RF $COMMON --perturb_seed \$N --seeds \$N >> $STATE/fast_${tag}_g$g.log 2>&1"
    echo "done"
    echo "echo \"[fast $tag g$g] DONE \$(date)\""
  } > "$w"; chmod +x "$w"
done
echo "Apply-once workers for sigma=$SIGMA written -> $RUN (100 draws, 25/GPU)."

if [ "$GO" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_fast_${tag}_g$g.sh" Enter
    echo "  dispatched worker_fast_${tag}_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED sigma=$SIGMA (apply-once, ~2.5h)."
else
  echo "PREPARE-ONLY. To launch: bash steer/experiments/randopt_mine_fast.sh $SIGMA go"
fi
