#!/bin/bash
# Start allpose8che10 only after freedeep8che genuinely finishes (2349/2349 outputs).
# Gate on output count, not process exit -- a scheduler can die early and "process gone" has been
# mistaken for "done" before. Preflight re-runs at fire time and a failure ABORTS rather than launching.
set -u
E=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
O1=$E/artifacts/sweep_out/freedeep8che/freedeep8che
log(){ echo "[$(date '+%m-%d %H:%M')] $*"; }
while :; do
  n=$(ls -d $O1/*/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)
  if [ "$n" -ge 2349 ]; then log "freedeep8che complete: $n/2349"; break; fi
  if ! pgrep -f "sweep_sched.py.*freedeep8che_jobs" >/dev/null; then
    log "ABORT: freedeep8che scheduler gone at only $n/2349 outputs -- not chaining"; exit 1
  fi
  sleep 600
done
log "preflight for allpose8che10"
$PY $E/scripts/run_health.py --preflight --jobs $E/artifacts/allpose8che10_jobs.json \
  --out $E/artifacts/sweep_out/allpose8che10 --seeds 10 || { log "ABORT: preflight failed"; exit 1; }
mkdir -p $E/artifacts/allpose8che10_logs
log "launching allpose8che10 on GPUs 0-3"
cd /home/ebrahim/Protenix
setsid nohup taskset -c 0-63 $PY $E/scripts/sweep_sched.py \
  --jobs $E/artifacts/allpose8che10_jobs.json --out $E/artifacts/sweep_out/allpose8che10 \
  --logs $E/artifacts/allpose8che10_logs --status $E/artifacts/allpose8che10_status.json \
  --gpus 0,1,2,3 --n-sample 5 > $E/artifacts/allpose8che10_sched.log 2>&1 < /dev/null &
sleep 240
null=$(grep -h "Found 0 templates" $E/artifacts/allpose8che10_logs/*.log 2>/dev/null | wc -l)
ok=$(grep -h "Found .* templates" $E/artifacts/allpose8che10_logs/*.log 2>/dev/null | wc -l)
log "template check: $ok loaded, $null NULL"
[ "$null" -gt 0 ] && log "WARNING: null templates -- arm would be VOID"
setsid nohup $PY $E/scripts/run_health.py --watch --jobs $E/artifacts/allpose8che10_jobs.json \
  --out $E/artifacts/sweep_out/allpose8che10 --logs $E/artifacts/allpose8che10_logs \
  --seeds 10 --interval 1800 > $E/artifacts/allpose8che10_health.log 2>&1 < /dev/null &
log "watcher up; chain done"
