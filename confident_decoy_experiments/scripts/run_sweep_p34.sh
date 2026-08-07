#!/bin/bash
# Phases 3+4 of the sweep (templates already built by build_sweep_templates.py).
# Both phases are resumable: sweep_sched.py runs only the seeds still missing, so this script can be
# killed and relaunched without losing completed work.
set -u
EXP=/home/ebrahim/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
cd $EXP
echo "=== [$(date +%H:%M)] PHASE 3: GPU runs (GPUs 0-3), shortest-job-first (banks results early; 7z2m runs last) ==="
$PY scripts/sweep_sched.py > $EXP/artifacts/decoy_logs/sweep_sched.log 2>&1
tail -4 $EXP/artifacts/decoy_logs/sweep_sched.log
echo "=== [$(date +%H:%M)] PHASE 4: scoring ==="
$PY scripts/score_sweep.py > $EXP/artifacts/decoy_logs/score_sweep.log 2>&1
tail -14 $EXP/artifacts/decoy_logs/score_sweep.log
echo "=== [$(date +%H:%M)] SWEEP COMPLETE ==="
