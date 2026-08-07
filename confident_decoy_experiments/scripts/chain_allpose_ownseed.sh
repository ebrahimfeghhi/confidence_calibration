#!/bin/bash
# Launch allpose8cheseed only AFTER allpose8che genuinely completes.
# Gate on OUTPUT COUNT (250/250 sample_4 confidence JSONs), not on the scheduler process exiting --
# a scheduler can exit early on a crash, and "process gone" has previously been mistaken for "done".
# Preflight is re-run at fire time; a non-zero exit ABORTS the launch rather than starting blind.
set -u
E=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
O1=$E/artifacts/sweep_out/allpose8che/allpose8che
log(){ echo "[$(date +%H:%M)] $*"; }

while :; do
  n=$(ls -d $O1/*/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)
  if [ "$n" -ge 250 ]; then log "allpose8che complete: $n/250"; break; fi
  if ! pgrep -f "sweep_sched.py.*allpose8che_jobs" >/dev/null; then
    log "ABORT: allpose8che scheduler gone but only $n/250 outputs exist -- not chaining"; exit 1
  fi
  sleep 120
done

log "running preflight for allpose8cheseed"
$PY $E/scripts/run_health.py --preflight \
  --jobs $E/artifacts/allpose8cheseed_jobs.json \
  --out $E/artifacts/sweep_out/allpose8cheseed --seeds 1 || { log "ABORT: preflight failed"; exit 1; }

mkdir -p $E/artifacts/allpose8cheseed_logs
log "launching allpose8cheseed on GPUs 0-3"
cd /home/ebrahim/Protenix
setsid nohup taskset -c 0-63 $PY $E/scripts/sweep_sched.py \
  --jobs $E/artifacts/allpose8cheseed_jobs.json \
  --out $E/artifacts/sweep_out/allpose8cheseed \
  --logs $E/artifacts/allpose8cheseed_logs \
  --status $E/artifacts/allpose8cheseed_status.json \
  --gpus 0,1,2,3 --n-sample 5 \
  > $E/artifacts/allpose8cheseed_sched.log 2>&1 < /dev/null &
sleep 120
tmpl=$(grep -h "Found .* templates" $E/artifacts/allpose8cheseed_logs/*.log 2>/dev/null | wc -l)
null=$(grep -h "Found 0 templates" $E/artifacts/allpose8cheseed_logs/*.log 2>/dev/null | wc -l)
log "template load check: $tmpl loaded, $null NULL"
if [ "$null" -gt 0 ]; then log "WARNING: null templates present -- arm would be void"; fi

setsid nohup $PY $E/scripts/run_health.py --watch \
  --jobs $E/artifacts/allpose8cheseed_jobs.json \
  --out $E/artifacts/sweep_out/allpose8cheseed \
  --logs $E/artifacts/allpose8cheseed_logs \
  --seeds 1 --interval 1800 \
  > $E/artifacts/allpose8cheseed_health.log 2>&1 < /dev/null &
log "watcher up; chain done"
