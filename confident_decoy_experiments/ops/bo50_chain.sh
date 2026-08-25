#!/bin/bash
# Start the both-OFF 50-seed extension once d50 (sigma=1e-4, seeds 111-150) is finished.
# Same 49 complexes, same seeds, so the two arms end up matched at 50 seeds on the same interfaces.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/bo50_chain.log"; }
say "waiting for d50 to finish"
while true; do
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 200 ] && left=$((left+1))
    done < "$A/d50_queue.txt"
    [ "$left" -eq 0 ] && break
    say "d50 still running ($left arms incomplete)"
    sleep 600
done
say "d50 done -- starting both-off 50-seed extension"
for g in 0 1 5 6 7; do
    tmux has-session -t bo50w$g 2>/dev/null && continue
    tmux new-session -d -s bo50w$g -c $REPO
    tmux send-keys -t bo50w$g "bash $REPO/ops/bo50_worker.sh $g 2>&1 | tee -a $REPO/ops/bo50w$g.log" Enter
done
say "bo50 workers launched on gpus 1 5 6 7"
