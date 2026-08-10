#!/bin/bash
# Score allpose8che10 once ALL 2500 seeds (12500 poses) are done -- the matched-budget answer.
# Compare against freedeep8che: 2500 free seeds / 11835 poses, best DockQ 0.630, ZERO poses above
# 0.85, pooled top-ipTM selection 0.064 (8che_BE).
#
# Gates on OUTPUT COUNT, not on the scheduler exiting: a scheduler can die early and "process gone"
# has been mistaken for "done" before. If it dies short, this says so loudly and scores what exists
# rather than silently reporting a partial arm as complete.
set -u
S=/home/ebrahim/tts_abag/confident_decoy_experiments/scripts
E=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
O=$E/artifacts/sweep_out/allpose8che10/allpose8che10
while :; do
  tot=$(ls -d $O/*/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)
  if [ "$tot" -ge 2500 ]; then echo "[$(date +%H:%M)] COMPLETE $tot/2500"; break; fi
  if ! pgrep -f "sweep_sched.py.*allpose8che10_jobs" >/dev/null; then
    echo "[$(date +%H:%M)] WARNING scheduler gone at $tot/2500 -- arm is INCOMPLETE, scoring anyway"
    break
  fi
  echo "[$(date +%H:%M)] $tot/2500"
  sleep 1200
done
echo "=== null-template audit (must be 0, else the arm is void) ==="
grep -h "Found 0 templates" $E/artifacts/allpose8che10_logs/*.log 2>/dev/null | wc -l
echo "=== FINAL SCORE: allpose8che10 ==="
$PY $S/score_allpose.py --run allpose8che10 --workers 16
