#!/usr/bin/env bash
# Return ONE GPU to the PXM-2025 sweep the moment the borrowed job on it exits.
#
# One watcher per GPU rather than one for all four, so each card goes back as soon as ITS job is done
# instead of waiting for the slowest. GPUs are handed back within ~15 s of the job ending.
#
# WATCH is the tmux session to wait on and GPU the card to restart pxm<GPU> on. Both are required:
# a hardcoded session name here previously left a watcher waiting on a finished run, which would have
# restarted a worker onto a GPU the next arm was already using.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
WATCH=${WATCH:?set WATCH to the tmux session to wait on}
GPU=${GPU:?set GPU}
while tmux has-session -t "$WATCH" 2>/dev/null; do sleep 15; done
echo "$(date +%F\ %H:%M:%S) $WATCH done -> restarting pxm$GPU on GPU$GPU"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025.sh"
sleep 4
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "  pxm$GPU up" || echo "  FAILED pxm$GPU"
