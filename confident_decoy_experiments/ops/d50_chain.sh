#!/bin/bash
# Start the 1e-4 50-seed extension once the both-off gap fill (bo2) is done, then keep every card fed.
# TAIL PROBLEM: arms are per complex, so once fewer arms remain than GPUs, cards go idle. When that
# happens the remaining arms' seeds are split across the free cards instead of leaving them unused.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/d50_chain.log"; }
say "waiting for bo2 to finish"
while true; do
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 50 ] && left=$((left+1))
    done < "$A/bo2_queue.txt"
    [ "$left" -eq 0 ] && break
    say "bo2 still running ($left arms left)"
    sleep 300
done
say "bo2 done -- starting the 1e-4 50-seed extension"
for g in 1 5 6 7; do
    tmux has-session -t d50w$g 2>/dev/null && continue
    tmux new-session -d -s d50w$g -c $REPO
    tmux send-keys -t d50w$g "bash $REPO/ops/d50_worker.sh $g 2>&1 | tee -a $REPO/ops/d50w$g.log" Enter
done
say "d50 workers launched on gpus 1 5 6 7"
