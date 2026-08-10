#!/bin/bash
# Score the COMPLETE 2500-seed free pool for 8che, so the free arm is budget-matched to
# allpose8che10 (250 donors x 10 seeds = 2500 seeds / 12500 poses).
#
# Why this exists: earlier free-run numbers covered only 2266 of the 2500 seeds. Seeds 0-100
# had never been scored at all (they live in predictions_template/8che, not the freedeep tree)
# and 133 seeds inside the freedeep range were missing from its OST cache. Every free number
# quoted before this run -- best 0.630, two poses above 0.6, selection 0.064 -- came from that
# short pool and must be restated from this output.
#
# Chained BEHIND wait_full_score.sh rather than run alongside it: both drive 16 OST workers
# under taskset 0-63, and running them together would contend for the same cores. Waits for
# that PID to exit, then scores. Does NOT touch the running script (bash reads scripts
# incrementally; editing one mid-run can corrupt execution).
set -u
S=/home/ebrahim/tts_abag/confident_decoy_experiments/scripts
E=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
WAITPID=${1:-}

if [ -n "$WAITPID" ]; then
  echo "[$(date +%H:%M)] waiting for allpose scoring (pid $WAITPID) to finish"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 300; done
  echo "[$(date +%H:%M)] pid $WAITPID gone"
fi
# belt and braces: never start while any OST run is still active
while pgrep -f "score_allpose.py" >/dev/null; do
  echo "[$(date +%H:%M)] score_allpose still running, holding"; sleep 300
done

echo "=== FREE POOL, ALL 2500 SEEDS ==="
$PY $S/score_free_complete.py --workers 16
