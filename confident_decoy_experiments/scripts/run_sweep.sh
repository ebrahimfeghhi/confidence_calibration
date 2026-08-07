#!/bin/bash
# Driver for the 7-run template sweep (both interface sets, seeds 101-150).
# Chains: phase1 fold-selection table -> phase2 build templates -> phase3 GPU runs -> phase4 scoring.
# Each phase is resumable, so re-running this script after a kill picks up where it left off.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
L=$EXP/artifacts/decoy_logs
cd $EXP

echo "=== [$(date +%H:%M)] PHASE 1: wait for fold_selection_table (27 pdbs) ==="
while pgrep -f fold_selection_table.py >/dev/null; do sleep 60; done
n=$($PY -c "import json;print(len(json.load(open('artifacts/fold_selection_table.json'))))" 2>/dev/null || echo 0)
echo "phase1 complete: $n/27 pdbs"
if [ "$n" -lt 27 ]; then
  echo "phase1 short -> rerunning once to fill gaps"
  $PY scripts/fold_selection_table.py >> $L/fold_selection.log 2>&1
  n=$($PY -c "import json;print(len(json.load(open('artifacts/fold_selection_table.json'))))" 2>/dev/null || echo 0)
  echo "phase1 after retry: $n/27"
fi

echo "=== [$(date +%H:%M)] PHASE 2: build templates ==="
$PY scripts/build_sweep_templates.py > $L/build_sweep.log 2>&1
tail -8 $L/build_sweep.log

echo "=== [$(date +%H:%M)] PHASE 3: GPU runs (GPUs 0-3) ==="
$PY scripts/sweep_sched.py > $L/sweep_sched.log 2>&1
tail -4 $L/sweep_sched.log

echo "=== [$(date +%H:%M)] PHASE 4: scoring ==="
$PY scripts/score_sweep.py > $L/score_sweep.log 2>&1
tail -12 $L/score_sweep.log
echo "=== [$(date +%H:%M)] SWEEP COMPLETE ==="
