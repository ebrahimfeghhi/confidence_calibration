#!/bin/bash
# Native-template arm on the freshly classified set: all confident decoys + all generation failures
# that do not already have a native50 run at seeds 101-150. Resumable at seed granularity.
set -u
EXP=/home/ebrahim/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
cd $EXP
echo "=== [$(date +%H:%M)] native arm: GPU runs ==="
$PY scripts/sweep_sched.py --jobs $EXP/artifacts/native_run_jobs.json \
    --out $EXP/artifacts/native_run_out --logs $EXP/artifacts/native_run_logs \
    --status $EXP/artifacts/native_run_status.json > $EXP/artifacts/decoy_logs/native_run_sched.log 2>&1
tail -4 $EXP/artifacts/decoy_logs/native_run_sched.log
echo "=== [$(date +%H:%M)] NATIVE ARM GENERATION COMPLETE ==="
