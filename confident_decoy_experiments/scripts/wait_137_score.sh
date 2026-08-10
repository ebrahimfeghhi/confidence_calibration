#!/bin/bash
# Wait for the dockqdonor template (donor seed137/sample3) to finish its 10 seeds inside
# allpose8che10, then score every template complete at that moment and report POOLED ipTM selection.
#
# Why this donor: it is the one template known to reach DockQ 0.958 on 8che_BE (58% of its poses above
# 0.85 at 50 seeds). The matched-budget question is whether that reproduces at 10 seeds, and whether
# pooled ipTM over the whole templated arm finds it.
#
# Code lives in tts_abag (git-tracked); DATA still lives at /home/ebrahim/confident_decoy_experiments
# via paths.py. Do not repoint EXP while the run is live.
set -u
S=/home/ebrahim/tts_abag/confident_decoy_experiments/scripts
E=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
J=$E/artifacts/sweep_out/allpose8che10/allpose8che10/8che_allpose_137_3
while :; do
  n=$(ls -d $J/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)
  tot=$(ls -d $E/artifacts/sweep_out/allpose8che10/allpose8che10/*/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] 137_3: $n/10 seeds | run total $tot/2500"
  [ "$n" -ge 10 ] && break
  if ! pgrep -f "sweep_sched.py.*allpose8che10_jobs" >/dev/null; then
    echo "SCHEDULER GONE at $tot/2500 -- scoring what exists"; break
  fi
  sleep 600
done
echo "=== scoring all complete templates ==="
$PY $S/score_allpose.py --run allpose8che10 --workers 16
