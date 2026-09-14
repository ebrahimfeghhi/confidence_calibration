#!/bin/bash
# When bo50 drains, move every card back onto unif. Workers exit on their own once nothing is left
# to claim, so this waits for the queue rather than killing anything mid-arm.
# Also repoints the filler at unif, otherwise it would keep backfilling idle cards onto a dead queue.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*" | tee -a "$REPO/ops/bo50_to_unif.log"; }
say "waiting for bo50 to drain"
while true; do
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 200 ] && left=$((left+1))
    done < "$A/bo50_queue.txt"
    [ "$left" -eq 0 ] && break
    say "bo50 still running ($left arms left)"
    sleep 300
done
say "bo50 complete -- switching everything to unif"

# filler back to unif
sed -i -e 's#bo50_queue.txt#unif_queue.txt#' -e 's#-lt 200#-lt 250#' \
       -e 's#bo50_worker.sh#unif_worker.sh#' -e 's#bo50w\$g#unifw$g#g' \
       -e 's#bo50 arms left#unif arms left#' -e 's#onto bo50#onto unif#' \
       -e 's#bo50 queue drained#unif queue drained#' $REPO/ops/gpu_filler.sh
tmux kill-session -t gpufill 2>/dev/null || true
sleep 2
tmux new-session -d -s gpufill -c $REPO
tmux send-keys -t gpufill "bash $REPO/ops/gpu_filler.sh" Enter

for g in 0 1 2 3 5 6 7; do
    tmux has-session -t unifw$g 2>/dev/null && continue
    tmux new-session -d -s unifw$g -c $REPO
    tmux send-keys -t unifw$g "bash $REPO/ops/unif_worker.sh $g 2>&1 | tee -a $REPO/ops/unifw$g.log" Enter
done
say "unif workers launched on 0 1 2 3 5 6 7"
