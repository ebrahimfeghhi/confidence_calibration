#!/bin/bash
# PASS 1: the three remaining template arms at 25 interfaces each, ONE AT A TIME on GPUs 0-3.
# PASS 2: 7zau_AB for all four arms, deferred to the end.
#
# 7zau is deferred because it is by far the largest complex (N_token 2520, ~90 GB/GPU, ~20 min/seed
# vs 6-20 min for a whole 50-seed job elsewhere). Left in the queue it holds three idle GPUs for hours
# and blocks every later arm for the sake of one data point. Each arm therefore reports n=25 first,
# and pass 2 fills in the 26th for all arms together.
#
# Every arm is VALIDATED before scoring: scripts/validate_run.py checks that templates actually loaded
# ("Found 1 templates", not "Found 0"), that seeds are complete and that every a3m is well-formed.
# score_arm.py independently refuses to score an arm whose templates never loaded. This is the guard
# added after topdockq/topiptm/sposefv were found to have measured untemplated runs.
#
# ABSOLUTE PATHS ONLY: sweep_sched runs inference with cwd=/home/ebrahim/Protenix, so a relative --out
# silently resolves against that directory (cost several hours of confused bookkeeping once already).
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "[$(date +%H:%M)] $*"; }

wait_for_sched() {   # $1 = fragment identifying the scheduler's --jobs file
  while :; do
    alive=0
    for p in /proc/[0-9]*; do
      c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
      case "$c" in *sweep_sched.py*"$1"*) alive=1; break;; esac
    done
    [ "$alive" -eq 0 ] && break
    sleep 120
  done
}

finish_arm() {       # $1 arm, $2 jobs file
  local ARM=$1 JOBS=$2
  say "$ARM generation finished; validating"
  $PY $EXP/scripts/validate_run.py --jobs "$JOBS" \
      --out $EXP/artifacts/sweep_out/$ARM --logs $EXP/artifacts/${ARM}_logs 2>&1 | tail -12
  say "$ARM scoring"
  $PY $EXP/scripts/score_arm.py --arm $ARM --jobs "$JOBS" \
      --out $EXP/artifacts/sweep_out/$ARM --logs $EXP/artifacts/${ARM}_logs 2>&1 | tail -32
  say "$ARM done -> artifacts/${ARM}_scores.json"
}

# ---------- pass 1 ----------
say "waiting for the running topiptm26 scheduler"
wait_for_sched topiptm26_jobs
finish_arm topiptm26 $EXP/artifacts/topiptm26_jobs25.json

for ARM in ifacelddt26 globallddt26; do
  say "launching $ARM (25 interfaces)"
  $PY $EXP/scripts/sweep_sched.py \
      --jobs $EXP/artifacts/${ARM}_jobs25.json \
      --out $EXP/artifacts/sweep_out/$ARM \
      --logs $EXP/artifacts/${ARM}_logs \
      --status $EXP/artifacts/${ARM}_status.json 2>&1 | tail -12
  finish_arm $ARM $EXP/artifacts/${ARM}_jobs25.json
done

say "PASS 1 COMPLETE"
for f in dockqdonor topiptm26 ifacelddt26 globallddt26; do
  n=$($PY -c "import json;d=json.load(open('$EXP/artifacts/${f}_scores.json'));print(f'{sum(1 for v in d.values() if v[\"rescued\"])}/{len(d)}')" 2>/dev/null)
  say "  $f rescued $n"
done

# ---------- pass 2: 7zau for every arm ----------
say "PASS 2: 7zau_AB across all four arms (resumes from 17/50 banked seeds for dockqdonor)"
$PY $EXP/scripts/sweep_sched.py \
    --jobs $EXP/artifacts/pass2_7zau_jobs.json \
    --out $EXP/artifacts/sweep_out/pass2 \
    --logs $EXP/artifacts/pass2_logs \
    --status $EXP/artifacts/pass2_status.json 2>&1 | tail -12
say "PASS 2 generation finished -- 7zau scoring is per-arm and left for manual review"
say "QUEUE COMPLETE"
