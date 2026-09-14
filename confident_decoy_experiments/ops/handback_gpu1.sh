#!/usr/bin/env bash
# Return GPU1 to the PXM-2025 sweep the moment the borrowed run finishes.
#
# GPU1 was taken from the pxm1 worker for the mt8vyl self-templating arm, keeping our total at 4 GPUs.
# The requirement is that it goes straight back with no idle window, so this waits on the tmux session
# rather than on a fixed timer.
#
# 9goe's claim was already released when pxm1 was stopped (its 20/250 poses are kept; run_pxm2025.sh
# resumes per seed), so the restarted worker re-claims work normally.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
# WATCH is the session to wait on -- pass it in. Hardcoding a name here already caused a
# near-miss: the watcher was left waiting on a finished run and would have restarted pxm1
# onto a GPU the next arm was already using.
WATCH=${WATCH:?set WATCH to the tmux session to wait on}
while tmux has-session -t "$WATCH" 2>/dev/null; do sleep 30; done
echo "$(date +%F\ %H:%M) $WATCH finished; restarting pxm1 on GPU1"
tmux new-session -d -s pxm1 -c "$REPO" 'GPU=1 bash ops/run_pxm2025.sh'
sleep 5
tmux has-session -t pxm1 2>/dev/null && echo "pxm1 back up on GPU1" || echo "FAILED to restart pxm1"
