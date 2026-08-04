#!/bin/bash
# Safe watchdog for the final stretch of the full-MSA N=4 round.
# - Never launches generation jobs (8the seeds 101-104 were launched directly, one per GPU),
#   so it cannot duplicate in-flight work the way the previous per-PDB scheduler could.
# - Only keeps the SCORER alive, and exits the moment the report JSON appears.
EXP=/home/ebrahim/confident_decoy_experiments
LOGD=$EXP/artifacts/fullmsa_logs
REPORT=$EXP/artifacts/fullmsa_n4_report.json
PY=/home/ebrahim/Protenix/.venv/bin/python

log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGD/watchdog2.log"; }
log "watchdog2 start"

for i in $(seq 1 2880); do
  if [ -f "$REPORT" ]; then
    log "REPORT READY"
    exit 0
  fi
  if ! pgrep -f fullmsa_n4_report.py >/dev/null 2>&1; then
    log "scorer not running -> starting"
    nohup $PY $EXP/scripts/fullmsa_n4_report.py >> "$LOGD/n4_report.log" 2>&1 &
  fi
  sleep 30
done
log "watchdog2 timed out"
exit 1
