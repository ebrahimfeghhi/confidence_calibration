#!/usr/bin/env bash
# Swap the score watcher onto a 10-minute cycle WITHOUT truncating a pass in flight.
# score_msaoff5.py rewrites msaoff5_scores.csv wholesale ("w" mode), so killing the watcher mid-pass
# leaves a partial CSV until the next cycle. This waits for the current pass to exit, then restarts.
EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$EXP" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/retick_msascore.log; }
say "waiting for the in-flight scorer pass to finish"
while pgrep -f score_msaoff5.py >/dev/null; do sleep 20; done
say "pass done; restarting msascore at EVERY=600"
tmux kill-session -t msascore 2>/dev/null
tmux new-session -d -s msascore "cd $EXP && EVERY=600 bash ops/score_watch_msaoff5.sh"
sleep 5
tmux has-session -t msascore 2>/dev/null && say "msascore up on 10-min cycle" || say "FAILED to restart msascore"
