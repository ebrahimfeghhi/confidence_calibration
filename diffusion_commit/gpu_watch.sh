#!/usr/bin/env bash
# Keep a claim on our GPUs: every 60s, make sure each watched card has one of OUR processes on it --
# either real inference or a holder -- and RESTART THE HOLDER if it has none.
#
# WHY IT REMEDIATES RATHER THAN REPORTS. A watcher that only prints is not enough: this project has a
# case where the health log correctly flagged a wedged worker for 16 hours while it burned a GPU, and
# nobody was reading. So this one acts.
#
# WHY IT IS NEEDED AT ALL. Our runs re-invoke inference once per seed. Between seeds there is a
# CPU-only gap (template building, scoring) where the card reads 0 MiB and looks free to a neighbour.
# batch.sh now starts a holder itself, but a holder can still die -- tmux session killed, OOM, the run
# switching over, a smoke test between sweeps. This is the backstop for all of those.
#
# IT NEVER FIGHTS FOR A CARD. If another user has a process on a GPU, it logs and stands down. Their
# job got there first; taking the card back is not ours to do.
#
# Usage:  GPUS="5 6" bash gpu_watch.sh          (stop: touch the stop file it prints)
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS=${GPUS:-"5 6"}
GIB=${GIB:-10}
PERIOD=${PERIOD:-60}
LOG=${LOG:-$REPO/artifacts/gpu_watch.log}
STOP=${STOP:-$REPO/artifacts/GPU_WATCH_STOP}
ME=$(id -un)
rm -f "$STOP"
echo "[gpu_watch] watching GPUs [$GPUS] every ${PERIOD}s; log $LOG; stop: touch $STOP"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "START watching [$GPUS] period ${PERIOD}s holder ${GIB}GiB"

while true; do
  [ -f "$STOP" ] && { log "stop file seen, exiting (holders left running)"; exit 0; }

  for g in $GPUS; do
    ours=0; other=0; otheru=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      [ -z "$u" ] && continue
      if [ "$u" = "$ME" ]; then ours=1; else other=1; otheru="$otheru $u"; fi
    done

    # Something of ours is on the card -- inference or holder, either counts as a claim.
    [ "$ours" = "1" ] && continue

    # Someone else is computing here. Stand down, and drop our own stale holder session if one lingers.
    if [ "$other" = "1" ]; then
      log "GPU$g now used by$otheru and none of ours -- standing down (not reclaiming)"
      tmux has-session -t "hhold$g" 2>/dev/null && { tmux kill-session -t "hhold$g" 2>/dev/null; log "GPU$g killed our stale holder session"; }
      continue
    fi

    # Card is empty and unclaimed: this is the gap a neighbour launches into. Re-hold it.
    tmux kill-session -t "hhold$g" 2>/dev/null          # clear a dead/zombie session first
    rm -f "$REPO/artifacts/.hold_stop_gpu$g"            # a leftover stop file would kill the new holder
    tmux new-session -d -s "hhold$g" -c "$REPO" \
      "CUDA_VISIBLE_DEVICES=$g $PY $REPO/scripts/gpu_hold.py --gib $GIB --stop $REPO/artifacts/.hold_stop_gpu$g"
    sleep 5
    if tmux has-session -t "hhold$g" 2>/dev/null; then
      log "GPU$g was UNHELD and empty -> restarted holder (${GIB}GiB)"
    else
      log "GPU$g was UNHELD and empty -> holder FAILED TO START, card is exposed"
    fi
  done

  sleep "$PERIOD"
done
