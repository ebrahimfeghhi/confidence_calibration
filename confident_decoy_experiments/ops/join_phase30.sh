#!/usr/bin/env bash
# Add GPUs 0 and 4 to the phase30 run the moment their current work ends, with no idle window.
# GPU0 is finishing the paired-fraction measurement; GPU4 is finishing msaoff5's last complex (8xnh).
# Nothing is killed -- each holds its card until its own job exits, then a claimer starts immediately.
# The claimers coordinate through the shared phase30 lock dir, so no complex can be run twice.
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/join_phase30.log; }
say "waiting to add GPUs 0 and 4 to phase30"
declare -A done_g
while true; do
  left=0
  for g in 0 4; do
    [ "${done_g[$g]}" = 1 ] && continue
    busy=0
    for p in $(pgrep -f "runner/inference.py" 2>/dev/null); do
      [ -r /proc/$p/environ ] || continue
      gg=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
      [ "$gg" = "$g" ] && busy=1
    done
    # GPU0 additionally must have finished the paired-fraction loop (it cycles processes)
    if [ "$g" = 0 ] && ! grep -q "### done" ops/measure_paired.log 2>/dev/null; then busy=1; fi
    if [ "$busy" = 1 ]; then left=1; continue; fi
    if tmux has-session -t "p30_$g" 2>/dev/null; then done_g[$g]=1; continue; fi
    say "GPU$g free -> starting phase30 claimer"
    tmux new-session -d -s "p30_$g" "cd $REPO && bash ops/run_phase30.sh $g 2>&1 | tee -a ops/phase30_g${g}.log"
    sleep 3
    tmux has-session -t "p30_$g" 2>/dev/null && { say "GPU$g phase30 up"; done_g[$g]=1; } || { say "GPU$g FAILED to start"; left=1; }
  done
  [ "$left" = 0 ] && { say "GPUs 0 and 4 both added; watcher exiting"; break; }
  sleep 20
done
