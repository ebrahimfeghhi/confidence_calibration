#!/usr/bin/env bash
# Hand each GPU from phase30 to the deep-adaptive run as it drains, with no idle gap.
# Nothing is killed: phase30 workers exit on their own once nothing is left to claim, and a
# POCKET claimer (priority 1) starts in the same breath so the card is never observably free.
# The two runs use SEPARATE lock dirs, so a GPU that has flipped cannot take phase30 work and
# vice versa; the POCKET claimer (priority 1)s coordinate among themselves through their own lock dir.
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/join_deepadapt.log; }
say "armed: GPUs 0/3/4 will switch to pocket-constrained run as phase30 finishes"
declare -A flipped
while true; do
  pending=0
  for g in 0 3 4; do
    if [ "${flipped[$g]}" = 1 ]; then
      tmux has-session -t "da_$g" 2>/dev/null && pending=1
      continue
    fi
    # phase30 still on this card?
    busy=0
    for p in $(pgrep -f "runner/inference.py" 2>/dev/null); do
      [ -r /proc/$p/environ ] || continue
      gg=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
      [ "$gg" = "$g" ] && busy=1
    done
    if [ "$busy" = 1 ] || tmux has-session -t "p30_$g" 2>/dev/null; then pending=1; continue; fi
    say "GPU$g free -> starting POCKET claimer (priority 1)"
    tmux new-session -d -s "da_$g" "cd $REPO && bash ops/run_pocket.sh $g 2>&1 | tee -a ops/deepadapt_g${g}.log"
    sleep 3
    if tmux has-session -t "da_$g" 2>/dev/null; then say "GPU$g deep-adaptive up"; flipped[$g]=1; pending=1
    else say "GPU$g FAILED to start"; pending=1; fi
  done
  [ "$pending" = 0 ] && { say "all GPUs handed over and all POCKET claimer (priority 1)s exited"; break; }
  sleep 20
done
