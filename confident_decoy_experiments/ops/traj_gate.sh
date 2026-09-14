#!/usr/bin/env bash
# Start the trajectory run once the 43-target assembly rebuild drains, so neither waits on the other.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/traj_gate.log; }
say "gate armed: waiting for the assembly rebuild to finish"
while true; do
  left=0
  while read -r p; do
    [ -z "$p" ] && continue
    n=$(find artifacts/pp/out_asm/$p -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -lt 250 ] && left=$((left+1))
  done < artifacts/pp/asm_queue.txt
  if [ "$left" -eq 0 ]; then
    say "rebuild drained. starting trajectory run on GPUs 0/3/4"
    for g in 0 3 4; do
      tmux kill-session -t traj_$g 2>/dev/null
      tmux new-session -d -s traj_$g "cd $REPO && ./ops/run_traj.sh $g 2>&1 | tee -a ops/traj_g$g.log"
      say "  launched traj_$g"
    done
    break
  fi
  sleep 180
done
