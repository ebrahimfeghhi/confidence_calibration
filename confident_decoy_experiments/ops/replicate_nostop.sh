#!/usr/bin/env bash
# Keep all four replications running past the first hit.
#
# ops/replicate_sweep_9x1h.sh was written to stop everything on the first hit: the winning worker
# touches artifacts/replicate/STOP and breaks, and the other three exit at their next check. The
# request now is to let them keep going, so this guard polls for STOP and removes it before the other
# workers notice.
#
# WHY A GUARD RATHER THAN AN EDIT: all four scripts are executing, and bash reads a running script
# incrementally from a byte offset -- editing one in place makes the live interpreter resume in
# shifted code. That has already cost this project three silently re-run arms (ops/chain3.sh).
#
# WHAT THIS CANNOT SAVE: the worker that actually hits breaks its own loop and kills its own
# generator in the same step that touches STOP, so THAT worker stops regardless. Only the other three
# are preserved. The hit itself is recorded in its driver log before it exits, so nothing is lost.
#
# Polls every 2s: the other workers check STOP between seeds (~3.4 min apart) and in their 20s
# generation wait, so a 2s window closes it long before any of them look.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
STOP=$REPO/artifacts/replicate/STOP
LOG=$REPO/artifacts/replicate/nostop.log
echo "=== $(date +%F\ %H:%M:%S) STOP-guard armed" >> "$LOG"
while :; do
  if [ -e "$STOP" ]; then
    rm -f "$STOP"
    echo "=== $(date +%F\ %H:%M:%S) removed STOP (a worker hit; others continue)" >> "$LOG"
  fi
  # exit once no replication sessions remain
  tmux ls 2>/dev/null | grep -qE '^rep[0126]:' || {
    echo "=== $(date +%F\ %H:%M:%S) no replications left; guard exiting" >> "$LOG"; break; }
  sleep 2
done
