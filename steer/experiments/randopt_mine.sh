#!/bin/bash
# randopt_mine.sh — 96 weight-perturbation draws at σ=0.002 (the sweep-confirmed optimum), across
# GPUs 3/4/5/7, mask0/dropout0. Legacy per-seed perturbation mode: ONE model load per GPU, and the
# fold seed IS the perturbation seed (seeds 5-100; seeds 1-4 already done in the screen). One
# run_name; each draw -> seed_<N>/ dir. Feeds the density estimate + expert selection at the best σ.
#   bash steer/experiments/randopt_mine.sh        # prepare only (write workers, no GPU)
#   bash steer/experiments/randopt_mine.sh go     # dispatch to tmux gpu3/4/5/7
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
CX="8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo"
RUN=randopt_mine_s002
COMMON="--complexes $CX --weight_perturb_sigma 0.002 --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log --run_name $RUN"
declare -A RANGE=( [3]="5 28" [4]="29 52" [5]="53 76" [7]="77 100" )   # 96 draws / 4 GPUs = 24 each

for g in 3 4 5 7; do
  read lo hi <<< "${RANGE[$g]}"
  w=$STATE/worker_mine_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"
    echo "echo \"[mine g$g] START \$(date) seeds $lo-$hi\""
    echo "$PY $RF $COMMON --seeds $(seq -s' ' $lo $hi) > $STATE/mine_g$g.log 2>&1"
    echo "echo \"[mine g$g] DONE \$(date)\""
  } > "$w"; chmod +x "$w"
done
echo "Mine workers written: 96 draws (seeds 5-100) at σ=0.002, 24/GPU -> $RUN"

if [ "${1:-}" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_mine_g$g.sh" Enter
    echo "  dispatched worker_mine_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED (~2.5-3h). Then: eval_ost on $RUN + randopt_mine_stats.py"
else
  echo "PREPARE-ONLY. To launch: bash steer/experiments/randopt_mine.sh go"
fi
