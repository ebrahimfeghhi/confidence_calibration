#!/usr/bin/env bash
# Restart the 43-target assembly rebuild once the trajectory run finishes.
# The rebuild was paused at 29/43 to free the GPUs; run_pp_asm.sh resumes from completed seeds, so
# the three in-flight complexes only lose the seed that was mid-flight.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/asm_resume_gate.log; }
say "gate armed: waiting for the trajectory run to finish"
while true; do
  left=0
  for p in $(/home/ebrahim/Protenix/.venv/bin/python -c \
      "import json;print(' '.join(json.load(open('artifacts/traj20_seeds.json'))))"); do
    n=$(ls artifacts/traj_cycles/$p/*_coords.npz 2>/dev/null | wc -l)
    [ "$n" -lt 80 ] && left=$((left+1))
  done
  if [ "$left" -eq 0 ]; then
    say "trajectory run complete. resuming the assembly rebuild on GPUs 0/3/4"
    for g in 0 3 4; do
      tmux kill-session -t ppasm_$g 2>/dev/null
      tmux new-session -d -s ppasm_$g "cd $REPO && bash ops/run_pp_asm.sh $g 2>&1 | tee -a ops/pp_asm_g$g.log"
      say "  launched ppasm_$g"
    done
    break
  fi
  sleep 180
done
