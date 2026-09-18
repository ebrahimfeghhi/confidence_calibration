#!/usr/bin/env bash
# Watcher for the high-spread self-templating sweep: reports every INTERVAL and REMEDIATES.
#
# WHAT IT WATCHES. 60 fp32 arms (10 high-spread seeds x [5 per-pose + 1 all-5]). Per tick it prints
# arms done, per-seed progress, and flags: a dead worker with arms left, an orphaned claim (a claim
# with no scored rows and no live worker -- the known hole in claim-before-work), a stalled csv, or a
# Traceback in either log.
#
# WHY IT REMEDIATES. fp32 needs ~90 GB, so only ONE worker fits per card; a silent worker death halves
# throughput and nothing else notices. Relaunch is gated on the card being free of our own worker, so
# it can never double up and OOM.
#
# Stop with: touch <REPO>/artifacts/highspread/WATCH_STOP
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
DC=/home/ebrahim/tts_abag/diffusion_commit
PY=/home/ebrahim/Protenix/.venv/bin/python
H=$REPO/artifacts/highspread
MAN=$REPO/artifacts/highspread_manifest.tsv
CSV=$H/results.csv
CLAIM=$H/.claims
LOG=$H/watch.log
STOP=$H/WATCH_STOP
INTERVAL=${INTERVAL:-300}
STALL_MIN=${STALL_MIN:-30}
rm -f "$STOP"
last=0; last_change=$(date +%s)

while :; do
  [ -f "$STOP" ] && { echo "[$(date +%H:%M:%S)] stop file seen" | tee -a "$LOG"; exit 0; }
  now=$(date +%s)
  rows=$(grep -c . "$CSV" 2>/dev/null || echo 1); rows=$((rows - 1))
  arms=$((rows / 5)); total=$(wc -l < "$MAN")
  claimed=$(ls "$CLAIM" 2>/dev/null | wc -l)
  workers=$(ps -u "$USER" -o args= | grep -c "fast_dr[i]ver.*highspread" || true)
  [ "$rows" -ne "$last" ] && { last=$rows; last_change=$now; }
  stalled=$(( (now - last_change) / 60 ))

  msg="[$(date '+%m-%d %H:%M:%S')] $arms/$total arms | claimed $claimed | workers $workers"
  [ "$stalled" -ge "$STALL_MIN" ] && msg="$msg | !! NO NEW ROWS ${stalled}min"
  tb=$(grep -l Traceback "$H"/gpu*.log 2>/dev/null | tr '\n' ' ')
  [ -n "$tb" ] && msg="$msg | !! Traceback in $tb"
  # orphaned claims: claimed but not scored, with no worker to finish them
  if [ "$workers" -eq 0 ] && [ "$claimed" -gt "$arms" ]; then
    msg="$msg | !! $((claimed - arms)) orphaned claims"
  fi
  echo "$msg" | tee -a "$LOG"

  # RESULTS TABLE: per-seed, does the templated ipTM order the poses it was shown?
  $PY "$DC/highspread_report.py" 2>&1 | tee -a "$LOG" | tail -40

  # remediation: one worker per card, only if that card has no worker of ours
  if [ "$arms" -lt "$total" ]; then
    for G in 5 6; do
      if ! ps -u "$USER" -o args= | grep -q "CUDA_VISIBLE_DEVICES=$G.*highspread"; then
        running_here=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$G" 2>/dev/null | wc -l)
        if [ "$running_here" -le 1 ]; then          # holder only
          echo "[$(date +%H:%M:%S)] !! relaunching worker on gpu$G" | tee -a "$LOG"
          CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold nohup $PY "$DC/fast_driver.py" \
            --mode manifest --dtype fp32 --ncycle 10 --arm-nsample 5 --manifest "$MAN" \
            --claim-dir "$CLAIM" --csv "$CSV" --timing-csv "$REPO/artifacts/stage_timings.csv" \
            --out "$H/out" >> "$H/gpu$G.log" 2>&1 &
        fi
      fi
    done
  else
    echo "[$(date +%H:%M:%S)] ALL $total ARMS COMPLETE" | tee -a "$LOG"; exit 0
  fi
  sleep "$INTERVAL"
done
