#!/usr/bin/env bash
# Keep all four GPUs busy for the overnight 50-seed chain run. Checks every 5 minutes.
#
# THREE FAILURE MODES IT REPAIRS, all of which have wasted GPU-hours in this project before:
#   1. a worker dies mid-seed -> its claim is left behind and NO other worker will ever take that
#      seed. Claims record their owning GPU; a claim whose owner has no live session and no .done
#      marker is released.
#   2. a worker exits (crash, or genuinely out of seeds) leaving its GPU idle -> restart it if seeds
#      remain, otherwise hand the GPU to the PXM sweep so the card is never doing nothing.
#   3. a GPU sits at 0% with a live session (hung process) -> reported, not killed, because killing
#      a healthy-but-slow job is worse than logging it.
#
# It never touches GPUs 3,4,5,7 -- 4 and 5 are john's and 7 is shawn's.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
M=$REPO/artifacts/mchain50
LOG=$M/watch.log
GPUS="0 1 2 6"
cd "$REPO" || exit 1
mkdir -p "$M"

while true; do
  ts=$(date +%F\ %H:%M:%S)
  done_n=$(ls "$M/.done" 2>/dev/null | wc -l)
  claim_n=$(ls "$M/.claims" 2>/dev/null | wc -l)
  rows=$(( $(wc -l < "$M/results.csv" 2>/dev/null || echo 1) - 1 ))

  # --- 1. release claims held by dead workers
  for c in $(ls "$M/.claims" 2>/dev/null); do
    [ -e "$M/.done/$c" ] && continue
    own=$(cat "$M/.claims/$c/owner" 2>/dev/null)
    [ -n "$own" ] || continue
    if ! tmux has-session -t "mchw$own" 2>/dev/null; then
      rm -rf "$M/.claims/$c" && echo "[$ts] released stranded claim seed $c (owner GPU$own dead)" >> "$LOG"
    fi
  done

  # --- 2. every GPU must be doing something
  for g in $GPUS; do
    if tmux has-session -t "mchw$g" 2>/dev/null; then
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$g" 2>/dev/null | tr -dc '0-9')
      [ "${util:-0}" -eq 0 ] && echo "[$ts] NOTE GPU$g worker alive but 0% util (may be scoring)" >> "$LOG"
      continue
    fi
    # worker gone -- any seeds left?
    left=0
    for S in $(seq 300000 300049); do
      [ -e "$M/.done/$S" ] && continue
      [ -d "$M/.claims/$S" ] && continue
      left=1; break
    done
    if [ "$left" = "1" ]; then
      tmux kill-session -t "pxm$g" 2>/dev/null
      sleep 3
      tmux new-session -d -s "mchw$g" -c "$REPO" "GPU=$g bash ops/mchain50_worker.sh >> $M/logs/gpu$g.log 2>&1"
      echo "[$ts] restarted chain worker on GPU$g (seeds remain)" >> "$LOG"
    elif ! tmux has-session -t "pxm$g" 2>/dev/null; then
      tmux new-session -d -s "pxm$g" -c "$REPO" "GPU=$g bash ops/run_pxm2025_chunked.sh"
      echo "[$ts] chain work exhausted -> GPU$g handed to the PXM sweep" >> "$LOG"
    fi
  done

  echo "[$ts] seeds done $done_n/50  claimed $claim_n  csv rows $rows  workers $(tmux ls 2>/dev/null | grep -cE '^mchw[0-9]:')  pxm $(tmux ls 2>/dev/null | grep -cE '^pxm[0-9]:')" >> "$LOG"
  [ "$done_n" -ge 50 ] && { echo "[$ts] ALL 50 SEEDS DONE" >> "$LOG"; break; }
  sleep 300
done
