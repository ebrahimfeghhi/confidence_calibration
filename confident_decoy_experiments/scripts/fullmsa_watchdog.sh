#!/bin/bash
# Watchdog for the full-MSA N=4 round: keeps the scheduler and the report job alive,
# and exits as soon as the report JSON is written (so the harness notifies immediately).
EXP=/home/ebrahim/confident_decoy_experiments
LOGD=$EXP/artifacts/fullmsa_logs
REPORT=$EXP/artifacts/fullmsa_n4_report.json
PY=/home/ebrahim/Protenix/.venv/bin/python
CPY=/home/ebrahim/miniconda3/bin/python

log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/watchdog.log"; }
log "watchdog start"

for i in $(seq 1 2880); do            # up to ~24h at 30s ticks
  if [ -f "$REPORT" ]; then
    log "REPORT READY -> $REPORT"
    exit 0
  fi

  # is the N=4 generation finished?
  sched_done=$(grep -c "ALL_DONE phase1 N=4" "$LOGD/n4_scheduler.log" 2>/dev/null || echo 0)

  # keep the scheduler alive until it reports ALL_DONE
  if [ "$sched_done" -eq 0 ] && ! pgrep -f fullmsa_n4_sched.py >/dev/null; then
    log "scheduler died -> restarting"
    nohup $CPY $EXP/scripts/fullmsa_n4_sched.py >> "$LOGD/n4_scheduler.log" 2>&1 &
  fi

  # keep the report/scorer alive until it writes the JSON
  if ! pgrep -f fullmsa_n4_report.py >/dev/null; then
    log "report job not running -> restarting"
    nohup $PY $EXP/scripts/fullmsa_n4_report.py >> "$LOGD/n4_report.log" 2>&1 &
  fi

  sleep 30
done
log "watchdog timed out without a report"
exit 1
