#!/bin/bash
# randopt_setB.sh — the SAME weight-perturbation experiment on a NEW set of 16 ab-ag complexes.
# REVIEW-GATED: prepare-only unless run with "go". Runs, on the new 16, the mask0/dropout0 baseline
# + 100 apply-once draws at sigma=0.002 across GPUs 3/4/5/7:
#   baseline -> randopt_screen_B/baseline   |   draws -> randopt_mine_s002_B/<complex>/seed_N/
# Score after:  eval_ost --pred_dir ~/esm/foldbench_predictions/randopt_screen_B/baseline --out_csv results/csv_files/randopt_screen_B/baseline.csv
#               eval_ost --pred_dir ~/esm/foldbench_predictions/randopt_mine_s002_B     --out_csv results/csv_files/randopt_mine_s002_B.csv
# Analyze:      RANDOPT_CX="$CXB" RANDOPT_SCR=results/csv_files/randopt_screen_B \
#               RANDOPT_MINE=results/csv_files/randopt_mine_s002_B.csv python steer/experiments/randopt_mine_stats.py
#   bash steer/experiments/randopt_setB.sh        # prepare only (no GPU) — DEFAULT, review-gated
#   bash steer/experiments/randopt_setB.sh go     # launch (only after user review)
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
STATE=steer/experiments/randopt_screen_state; mkdir -p "$STATE"
# CONTRAST 16 where masked ipTM-SELECTION WORKS (ipTM-selected DockQ 0.87-0.95, ~0 gap to oracle);
# disjoint from set A, GT present, >=50 seeds. Tests whether the perturbation gain is ipTM-SELECTABLE
# when the good dock is also the confident one (mirror of set A, where selection is broken).
CXB="7txw 7xld 8f60 8jeo 8s0l 8pih 8d9z 8f5n 8trs 8b1a 8aon 8c3l 8q7o 8t59 8pmz 8tmz"
COMMON="--complexes $CXB --lm_mask_pct 0 --lm_dropout 0 --num_diffusion_samples 5 --fused --chunk_size 32 --no_config_log"
declare -A RANGE=( [3]="1 25" [4]="26 50" [5]="51 75" [7]="76 100" )   # 100 draws / 4 GPUs

for g in 3 4 5 7; do
  read lo hi <<< "${RANGE[$g]}"
  w=$STATE/worker_setB_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"; echo "echo \"[setB g$g] START \$(date)\""
    if [ "$g" = "3" ]; then   # baseline (no perturbation) once, on GPU3, before its draws
      echo "$PY $RF --run_name randopt_screen_B/baseline $COMMON --seeds 0 >> $STATE/setB_baseline.log 2>&1"
    fi
    echo "for N in \$(seq $lo $hi); do"
    echo "  $PY $RF --run_name randopt_mine_s002_B --weight_perturb_sigma 0.002 --perturb_seed \$N --seeds \$N $COMMON >> $STATE/setB_g$g.log 2>&1"
    echo "done"
    echo "echo \"[setB g$g] DONE \$(date)\""
  } > "$w"; chmod +x "$w"
done
echo "Set-B workers written (new 16: baseline + 100 draws @ sigma=0.002, 25/GPU)."

if [ "${1:-}" = "go" ]; then
  for g in 3 4 5 7; do
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && CUDA_VISIBLE_DEVICES=$g bash $STATE/worker_setB_g$g.sh" Enter
    echo "  dispatched worker_setB_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED set B."
else
  echo "PREPARE-ONLY (REVIEW-GATED). Launch only after review: bash steer/experiments/randopt_setB.sh go"
fi
