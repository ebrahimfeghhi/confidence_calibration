#!/usr/bin/env bash
# Stop the other hunt worker when either one hits.
#
# WHY THIS EXISTS. Worker 1 (ops/hunt_9x1h.sh) was launched before the second worker was conceived,
# so it knows nothing about the STOP sentinel. It cannot be patched: bash reads a running script
# incrementally from a byte offset, and editing one in place makes the live interpreter resume in
# shifted code -- that already cost this project three silently re-run arms (ops/chain3.sh).
#
# So the coordination lives outside both scripts:
#   worker 2 hits -> it touches STOP itself; this referee kills worker 1's session and restarts pxm6
#   worker 1 hits -> this referee touches STOP; worker 2 exits at its next check and restarts pxm2
#                    on its own
# Either way both GPUs return to the PXM sweep and the winning dump is left in place.
#
# Detection is on the "HIT on seed" line that both workers print, so it fires on the same event the
# logs record, not on a separate re-derivation of who won.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
OUT=$REPO/artifacts/hunt
STOP=$OUT/STOP
L1=$REPO/artifacts/hunt_driver.log
L2=$REPO/artifacts/hunt_w2_driver.log
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) referee watching both workers"
while :; do
  # neither worker alive -> nothing left to referee
  if ! tmux has-session -t hunt 2>/dev/null && ! tmux has-session -t huntw2 2>/dev/null; then
    echo "=== $(date +%F\ %H:%M:%S) both workers gone; referee exiting"; break
  fi
  if grep -q 'HIT on seed' "$L1" 2>/dev/null; then
    s=$(grep -oE 'HIT on seed [0-9]+' "$L1" | tail -1 | grep -oE '[0-9]+')
    echo "=== $(date +%F\ %H:%M:%S) worker 1 hit seed $s -> setting STOP for worker 2"
    touch "$STOP"
    break
  fi
  if grep -q 'HIT on seed' "$L2" 2>/dev/null; then
    s=$(grep -oE 'HIT on seed [0-9]+' "$L2" | tail -1 | grep -oE '[0-9]+')
    echo "=== $(date +%F\ %H:%M:%S) worker 2 hit seed $s -> stopping worker 1 and restoring pxm6"
    touch "$STOP"
    tmux kill-session -t hunt 2>/dev/null
    sleep 5
    pkill -f 'DUMP_FULL_DIR.*hunt/reps' 2>/dev/null
    sleep 3
    tmux new-session -d -s pxm6 -c "$REPO" "GPU=6 bash ops/run_pxm2025_chunked.sh"
    sleep 5
    tmux has-session -t pxm6 2>/dev/null && echo "    pxm6 restarted" || echo "    FAILED to restart pxm6"
    break
  fi
  sleep 20
done
echo "=== $(date +%F\ %H:%M:%S) referee done"
