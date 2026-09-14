#!/bin/bash
# Keep every card busy. When a bo50 worker exits because its queue drained, that GPU would sit idle
# until someone noticed -- which is exactly how four cards idled for 5 h overnight. This backfills
# any card in LIST that has no worker session, provided unif still has unclaimed arms.
#
# GPU 4 is deliberately excluded: it is reserved for other users on this unscheduled box.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
LIST="0 1 2 3 5 6 7"
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*" | tee -a "$REPO/ops/gpu_filler.log"; }
say "filler started (backfill idle cards onto bo50; gpu4 reserved)"
while true; do
    # unclaimed unif work left?
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 200 ] && left=$((left+1))
    done < "$A/bo50_queue.txt"
    if [ "$left" -eq 0 ]; then say "bo50 queue drained -- filler exiting"; exit 0; fi

    for g in $LIST; do
        # any worker session already own this card?
        if tmux ls -F '#{session_name}' 2>/dev/null | grep -qE "^(bo50w|unifw|d50w)$g$"; then continue; fi
        # card must also be genuinely free of processes -- never start on top of someone else
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
        [ "$mem" -ge 2000 ] && { say "gpu$g has no session but ${mem}MiB used -- leaving alone"; continue; }
        say "gpu$g idle with $left bo50 arms left -> starting unif worker"
        tmux new-session -d -s bo50w$g -c $REPO
        tmux send-keys -t bo50w$g "bash $REPO/ops/bo50_worker.sh $g 2>&1 | tee -a $REPO/ops/bo50w$g.log" Enter
    done
    sleep 180
done
