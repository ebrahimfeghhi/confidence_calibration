#!/bin/bash
# Start 8che_BE's 50-seed run as soon as the 8av2_AC one finishes -- one arm at a time on GPUs 0-3.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%H:%M)] $*"; }
say "waiting for the 8av2_AC deep run to finish"
while :; do
  alive=0
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in *sweep_sched.py*av2deep_jobs*) alive=1; break;; esac
  done
  [ "$alive" -eq 0 ] && break
  sleep 120
done
say "8av2_AC done -- preflight for 8che_BE"
$PY $EXP/scripts/run_health.py --preflight --jobs $EXP/artifacts/chedeep_jobs.json --seeds 50 || {
  say "PREFLIGHT FAILED -- not launching"; exit 1; }
say "launching 8che_BE 20 templates x 50 seeds on GPUs 0-3"
mkdir -p $EXP/artifacts/chedeep_logs
$PY $EXP/scripts/run_health.py --watch --jobs $EXP/artifacts/chedeep_jobs.json \
    --out $EXP/artifacts/sweep_out/leader20 --logs $EXP/artifacts/chedeep_logs \
    --seeds 50 --interval 1800 > $EXP/artifacts/decoy_logs/chedeep_health.log 2>&1 &
$PY $EXP/scripts/sweep_sched.py --jobs $EXP/artifacts/chedeep_jobs.json \
    --out $EXP/artifacts/sweep_out/leader20 --logs $EXP/artifacts/chedeep_logs \
    --status $EXP/artifacts/chedeep_status.json
say "8che_BE generation COMPLETE"
