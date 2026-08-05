#!/bin/bash
# Score leader20 interfaces as their 20 template runs complete (CPU/OST only, no GPU contention).
# The method's answer is POOLED across an interface's templates, so an interface is scored only when
# all of its templates are done -- a partial pool is not the method's output.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
while :; do
  before=$($PY -c "import json,os;p='$EXP/artifacts/leader20_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  $PY $EXP/scripts/score_leader20.py --workers 10 >/dev/null 2>&1
  after=$($PY -c "import json,os;p='$EXP/artifacts/leader20_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  # only speak when a NEW interface has been scored; otherwise stay silent (the 30-min health
  # check already reports liveness, and repeating an unchanged tally every 10 min is pure noise)
  if [ "$after" -gt "$before" ]; then
    $PY $EXP/scripts/leader20_table.py 2>&1 | tail -40
  fi
  sched=0
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in *sweep_sched.py*leader20*) sched=1; break;; esac
  done
  n=$($PY -c "import json,os;p='$EXP/artifacts/leader20_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  if [ "$sched" -eq 0 ] && [ "$n" -ge 26 ]; then echo "[$(date +%H:%M)] all 26 interfaces scored -- watcher exiting"; break; fi
  sleep 600
done
