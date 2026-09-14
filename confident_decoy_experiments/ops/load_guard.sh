#!/usr/bin/env bash
# Stop the rescore if box load approaches the 188 ceiling. Other users (kao ran at 1496% CPU) can
# push the box over on their own, so our job must yield rather than assume it owns the machine.
CEIL=${CEIL:-170}
while true; do
  L=$(awk '{print int($1)}' /proc/loadavg)
  if [ "$L" -gt "$CEIL" ]; then
    if pgrep -f "abag_all_101_150.py --complexes" >/dev/null; then
      echo "[$(date '+%T')] load $L > $CEIL -- pausing rescore" | tee -a ops/load_guard.log
      pkill -STOP -f "abag_all_101_150.py --complexes"  # covers both rescore and fullscore
    fi
  else
    if pgrep -f "abag_all_101_150.py --complexes" >/dev/null; then
      pkill -CONT -f "abag_all_101_150.py --complexes" 2>/dev/null
    fi
  fi
  sleep 30
done
