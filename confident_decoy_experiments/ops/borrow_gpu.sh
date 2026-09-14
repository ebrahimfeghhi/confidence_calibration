#!/usr/bin/env bash
# Borrow a GPU from the hybrid25 sweep, run tmplctl on it, and ALWAYS give it back.
#
# THE GUARANTEE. The sweep worker is restored by an EXIT trap, so it comes back whether tmplctl
# succeeds, fails its gate (exit 2), errors, or this script is interrupted. Losing a card to a
# half-finished borrow would be worse than not running the experiment at all.
#
# WHY KILLING THE WORKER IS SAFE. hybrid25_worker.sh claims a unit by creating .claims/<unit>/ and
# only touches .done/<unit> AFTER rows are written to the CSV. So a unit killed mid-flight is not
# marked done -- but its CLAIM dir survives, and a stale claim silently blocks every future worker
# from retrying it. This script therefore releases the in-flight claim explicitly. (Stale claims have
# already cost this project a silent stall once.)
#
# The borrowed card must be idle before tmplctl starts, or the first run OOMs and the whole borrow is
# wasted; CUDA frees VRAM asynchronously after the process dies, so we poll rather than assume.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
M=$REPO/artifacts/hybrid25
GPU=${GPU:-5}
NS=${NS:-20}                      # must match how the worker was launched, or it changes the sweep
SESS=${SESS:-hyw$GPU}
cd "$REPO" || exit 1

restore () {
  echo "=== $(date +%H:%M:%S) RESTORING sweep worker on GPU$GPU"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    tmux send-keys -t "$SESS" "GPU=$GPU NS=$NS bash ops/hybrid25_worker.sh" C-m
  else
    tmux new-session -d -s "$SESS" "GPU=$GPU NS=$NS bash ops/hybrid25_worker.sh"
  fi
  sleep 5
  if pgrep -f "hybrid25_worker" >/dev/null; then
    echo "    worker process is back (GPU$GPU, NS=$NS, session $SESS)"
  else
    echo "    !! WORKER DID NOT COME BACK -- run this by hand NOW:"
    echo "       tmux new-session -d -s $SESS 'cd $REPO && GPU=$GPU NS=$NS bash ops/hybrid25_worker.sh'"
  fi
}
trap restore EXIT INT TERM

# --- 1. what is GPU$GPU holding right now? -------------------------------------------------------
INFLIGHT=""
for c in "$M"/.claims/*/; do
  [ -f "$c/owner" ] || continue
  [ "$(cat "$c/owner" 2>/dev/null)" = "$GPU" ] || continue
  u=$(basename "$c")
  [ -e "$M/.done/$u" ] && continue        # finished, claim is legitimately spent
  INFLIGHT="$u"
done
echo "=== $(date +%H:%M:%S) GPU$GPU in-flight unit: ${INFLIGHT:-none}"

# --- 2. stop the worker and its inference child --------------------------------------------------
for p in $(pgrep -f "hybrid25_worker"); do
  env_gpu=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^GPU=' | cut -d= -f2)
  [ "$env_gpu" = "$GPU" ] && { echo "  killing worker pid $p"; kill "$p" 2>/dev/null; }
done
for p in $(pgrep -f 'runner/inference.py'); do
  env_gpu=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
  [ "$env_gpu" = "$GPU" ] && { echo "  killing inference pid $p"; kill "$p" 2>/dev/null; }
done

# --- 3. release the in-flight claim so the unit is retried, not silently skipped ------------------
if [ -n "$INFLIGHT" ]; then
  rm -rf "${M:?}/.claims/${INFLIGHT:?}"
  echo "  released claim $INFLIGHT (not in .done, so it will be re-run)"
fi

# --- 4. wait for the card to actually be free ----------------------------------------------------
for i in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
  [ "${free:-0}" -ge 60000 ] && { echo "  GPU$GPU free: ${free}MiB"; break; }
  sleep 5
done

# --- 5. the experiment (trap restores the worker no matter how this ends) -------------------------
echo "=== $(date +%H:%M:%S) starting tmplctl on GPU$GPU"
GPU=$GPU bash ops/tmplctl.sh
echo "=== $(date +%H:%M:%S) tmplctl finished with status $?"
