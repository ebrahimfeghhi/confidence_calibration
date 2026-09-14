#!/usr/bin/env bash
# Borrow a GPU from a running sweep, run CMD on it, and ALWAYS give it back.
#
# Generic version of ops/borrow_gpu.sh (which was hardcoded to hybrid25 + tmplctl). The worker is
# restored by an EXIT trap, so the card comes back whether CMD succeeds, fails, OOMs, or this script
# is interrupted -- losing a card to a half-finished borrow is worse than not running at all.
#
# WHY KILLING A WORKER IS SAFE: hybrid25b_worker.sh claims a unit by creating .claims/<unit>/ and only
# touches .done/<unit> AFTER rows are written. A unit killed mid-flight is therefore not marked done
# -- but its CLAIM dir survives and silently blocks every future retry, so it is released explicitly.
#
# Usage: GPU=5 WORKER=hybrid25b_worker.sh ROOT=artifacts/hybrid25b CMD="..." bash ops/borrow.sh
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPU=${GPU:-5}
NS=${NS:-20}
WORKER=${WORKER:-hybrid25b_worker.sh}
ROOT=${ROOT:-$REPO/artifacts/hybrid25b}
SESS=${SESS:-hyw$GPU}
CMD=${CMD:?set CMD}
cd "$REPO" || exit 1
LOG=ops/borrow.log

say () { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

restore () {
  say "RESTORING $WORKER on GPU$GPU"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    tmux send-keys -t "$SESS" "cd $REPO && GPU=$GPU NS=$NS bash ops/$WORKER" C-m
  else
    tmux new-session -d -s "$SESS" "cd $REPO && GPU=$GPU NS=$NS bash ops/$WORKER"
  fi
  sleep 5
  pgrep -u "$(id -u)" -f "$WORKER" >/dev/null \
    && say "  worker is back (GPU$GPU, NS=$NS, session $SESS)" \
    || say "  !! WORKER DID NOT RETURN -- run by hand: tmux new-session -d -s $SESS 'cd $REPO && GPU=$GPU NS=$NS bash ops/$WORKER'"
}
trap restore EXIT INT TERM

INFLIGHT=""
for c in "$ROOT"/.claims/*/; do
  [ -f "$c/owner" ] || continue
  [ "$(cat "$c/owner" 2>/dev/null)" = "$GPU" ] || continue
  u=$(basename "$c"); [ -e "$ROOT/.done/$u" ] && continue
  INFLIGHT="$u"
done
say "GPU$GPU in-flight unit: ${INFLIGHT:-none}"

for p in $(pgrep -u "$(id -u)" -f "$WORKER"); do
  g=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^GPU=' | cut -d= -f2)
  [ "$g" = "$GPU" ] && { say "  killing worker pid $p"; kill "$p" 2>/dev/null; }
done
for p in $(pgrep -u "$(id -u)" -f 'runner/inference.py'); do
  g=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
  [ "$g" = "$GPU" ] && { say "  killing inference pid $p"; kill "$p" 2>/dev/null; }
done
[ -n "$INFLIGHT" ] && { rm -rf "${ROOT:?}/.claims/${INFLIGHT:?}"; say "  released claim $INFLIGHT"; }

for i in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
  [ "${free:-0}" -ge 60000 ] && { say "  GPU$GPU free ${free}MiB"; break; }
  sleep 5
done

say "running: $CMD"
CUDA_VISIBLE_DEVICES=$GPU bash -c "$CMD"
say "command finished with status $?"
