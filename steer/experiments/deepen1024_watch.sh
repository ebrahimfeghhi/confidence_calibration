#!/bin/bash
# deepen1024_watch.sh — wait for the deepen-to-1024 folding to finish, then score the new seeds
# (resume-safe append to the canonical deep CSV) and refresh the depth analysis + figures. Detects
# completion when no run_foldbench_abag processes remain. Safe to re-run manually at any time.
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
STATE=steer/experiments/deepen1024_state
echo "[watcher] START $(date) — waiting for folding to finish"
while [ "$(ps -eo args | grep -c '[r]un_foldbench_abag')" -gt 0 ]; do sleep 120; done
echo "[watcher] folding done $(date) — scoring new seeds"
$PY eval/eval_ost_foldbench.py \
  --pred_dir "$HOME/esm/foldbench_predictions/esmfold2fast_tts" \
  --out_csv results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv > "$STATE/refresh_score.log" 2>&1
echo "[watcher] scoring done $(date) — refreshing analysis + figures"
$PY steer/experiments/severe_selection_at_depth.py > "$STATE/refresh_selection.txt" 2>&1
$PY steer/experiments/plot_confident_decoy.py       >> "$STATE/refresh_selection.txt" 2>&1
$PY steer/experiments/tts_seed_status.py            > "$STATE/refresh_status.txt" 2>&1
echo "[watcher] REFRESH COMPLETE $(date)"
