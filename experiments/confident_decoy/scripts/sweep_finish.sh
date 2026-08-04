#!/bin/bash
# Finisher: make sure every job reaches 50/50 seeds, THEN score.
#
# Why this exists: the scheduler pops each job once. Jobs that were cut short (the watchdog incident on
# 2026-08-01 killed four mid-run) are never retried by that pass, so phase 3 can "finish" with
# incomplete pools. This runs repeated scheduler passes until a pass adds nothing, then scores.
set -u
EXP=/home/ebrahim/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
cd $EXP
count() { $PY -c "
import json,os
jobs=json.load(open('artifacts/sweep_jobs.json')); SE=range(101,151); n=0
for j in jobs:
    d=f\"artifacts/sweep_out/{j['run']}/{j['name']}\"
    n+=sum(1 for s in SE if os.path.exists(f'{d}/seed_{s}/predictions/{j[\"name\"]}_summary_confidence_sample_4.json'))
print(n)"; }

echo "=== [$(date +%H:%M)] finisher: waiting for the current scheduler/driver to exit ==="
while pgrep -f "scripts/sweep_sched.py" >/dev/null || pgrep -f "run_sweep_p34.sh" >/dev/null; do sleep 120; done

for pass in 1 2 3; do
  before=$(count)
  echo "=== [$(date +%H:%M)] retry pass $pass (seeds now $before/5400) ==="
  $PY scripts/sweep_sched.py > $EXP/artifacts/decoy_logs/sweep_sched_retry$pass.log 2>&1
  after=$(count)
  echo "=== [$(date +%H:%M)] pass $pass added $((after-before)) seeds (total $after/5400) ==="
  [ "$after" -le "$before" ] && break
done

echo "=== [$(date +%H:%M)] PHASE 4: scoring ==="
$PY scripts/score_sweep.py > $EXP/artifacts/decoy_logs/score_sweep.log 2>&1
tail -14 $EXP/artifacts/decoy_logs/score_sweep.log
echo "=== [$(date +%H:%M)] SWEEP COMPLETE ==="
