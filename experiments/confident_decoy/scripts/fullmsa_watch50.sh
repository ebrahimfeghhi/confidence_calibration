#!/bin/bash
# Watchdog for the full-MSA N=50 round. Keeps the queue scheduler and the N=50 scorer alive.
# The queue scheduler is safe to restart: it rebuilds the missing-seed list from disk, but it does
# NOT check for in-flight jobs, so only restart it when no fullmsa inference process is running.
EXP=/home/ebrahim/confident_decoy_experiments
LOGD=$EXP/artifacts/fullmsa_logs
REPORT=$EXP/artifacts/fullmsa_n50_report.json
PY=/home/ebrahim/Protenix/.venv/bin/python
CPY=/home/ebrahim/miniconda3/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/watchdog50.log"; }
log "watchdog50 start"
for i in $(seq 1 5760); do
  [ -f "$REPORT" ] && { log "REPORT READY"; exit 0; }
  running=$(ps -eo pid,cmd | grep inference.py | grep -v grep | grep -c fullmsa_out)
  sched_done=$(grep -c "ALL_DONE queue N=50" "$LOGD/queue50.log" 2>/dev/null || echo 0)
  if [ "$sched_done" -eq 0 ] && ! pgrep -f fullmsa_queue_sched >/dev/null 2>&1 && [ "$running" -eq 0 ]; then
    log "queue scheduler died and no jobs in flight -> restarting"
    nohup $CPY $EXP/scripts/fullmsa_queue_sched.py 50 >> "$LOGD/queue50.log" 2>&1 &
  fi
  if ! pgrep -f fullmsa_n50_report.py >/dev/null 2>&1; then
    log "scorer not running -> starting"
    nohup $PY $EXP/scripts/fullmsa_n50_report.py >> "$LOGD/n50_report.log" 2>&1 &
  fi
  sleep 30
done
log "watchdog50 timed out"; exit 1
