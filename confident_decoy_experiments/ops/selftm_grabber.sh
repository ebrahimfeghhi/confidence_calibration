#!/bin/bash
# Reclaim capacity as it frees up. We were pushed off 0-3 when another user landed on them, so this
# watches for genuinely EMPTY cards and starts a worker there, up to MAX_GPUS total.
#
# "Empty" means no compute process at all AND under 2 GB -- not merely "no process of ours". Starting
# beside someone else is what caused the collision in the first place, and with 80-90 GB arms it
# OOMs both jobs.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
MAX_GPUS=3; ME=$(id -un); CAND="0 1 2 3 5 6 7"
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*" | tee -a "$REPO/ops/selftm_grabber.log"; }
say "grabber started (reclaim up to $MAX_GPUS cards; only fully-empty ones)"
while true; do
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 50 ] && left=$((left+1))
    done < "$A/selftm_queue.txt"
    if [ "$left" -eq 0 ]; then say "queue drained -- grabber exiting"; exit 0; fi

    held=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^selftmw' || true)
    if [ "$held" -ge "$MAX_GPUS" ]; then sleep 300; continue; fi

    for g in $CAND; do
        [ "$held" -ge "$MAX_GPUS" ] && break
        tmux has-session -t selftmw$g 2>/dev/null && continue
        # any process at all on this card, ours or not?
        uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g 2>/dev/null)
        busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -cF "$uuid" || true)
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
        if [ "$busy" -eq 0 ] && [ "$mem" -lt 2000 ]; then
            say "gpu$g is empty -> claiming it ($((held+1))/$MAX_GPUS), $left arms left"
            tmux new-session -d -s selftmw$g -c $REPO
            tmux send-keys -t selftmw$g "bash $REPO/ops/selftm_worker.sh $g 2>&1 | tee -a $REPO/ops/selftmw$g.log" Enter
            held=$((held+1))
        fi
    done
    sleep 300
done
