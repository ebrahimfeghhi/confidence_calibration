#!/bin/bash
# randopt_score.sh — OST DockQ scoring for the Phase-1 weight-perturbation screen cells.
# CPU-only (OST). Runs eval_ost_foldbench.py per cell (DockQ only; the eval already drops
# --rigid-scores, per the known OST bug). One DockQ CSV per cell. Resumable/idempotent.
#   bash steer/experiments/randopt_score.sh
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
PRED=~/esm/foldbench_predictions/randopt_screen
OUT=results/csv_files/randopt_screen; mkdir -p "$OUT"
cells="baseline"
for sig in 0.0005 0.001 0.002 0.005 0.01; do for seed in 1 2 3 4; do cells="$cells s${sig}_seed${seed}"; done; done
for c in $cells; do
  [ -d "$PRED/$c" ] || { echo "skip $c (no predictions yet)"; continue; }
  echo "=== scoring $c ==="
  $PY eval/eval_ost_foldbench.py --pred_dir "$PRED/$c" --out_csv "$OUT/$c.csv"
done
echo "scored CSVs -> $OUT/  (now: $PY steer/experiments/randopt_stats.py)"
