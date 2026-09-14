#!/bin/bash
# Full pipeline for the remaining 29 complexes, run after wave 1's stage 2 drains:
#   stage 1 (MSA-free, 1 seed)  ->  build self-templates  ->  stage 2 (50 seeds, stock)
# Waits on queues rather than killing anything, and keeps all 4 cards fed at every step.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*" | tee -a "$REPO/ops/selftm_chain.log"; }

drained(){ # $1=queue file, $2=poses required per arm
    local left=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt "$2" ] && left=$((left+1))
    done < "$1"
    echo "$left"
}

say "waiting for wave-1 stage 2 to finish"
while [ "$(drained $A/selftm_queue.txt 250)" -gt 0 ]; do
    say "wave 1 stage 2: $(drained $A/selftm_queue.txt 250) arms left"; sleep 600
done
say "wave 1 done -- starting stage 1 for the remaining 29"

sed 's#s1_queue.txt#s1b_queue.txt#; s#s1_locks#s1b_locks#' $REPO/ops/s1_worker.sh > $REPO/ops/s1b_worker.sh
for g in 0 1 2 3; do
    tmux new-session -d -s s1bw$g -c $REPO
    tmux send-keys -t s1bw$g "bash $REPO/ops/s1b_worker.sh $g 2>&1 | tee -a $REPO/ops/s1bw$g.log" Enter
done
while [ "$(drained $A/s1b_queue.txt 5)" -gt 0 ]; do
    say "stage 1 wave 2: $(drained $A/s1b_queue.txt 5) complexes left"; sleep 120
done
say "stage 1 wave 2 done -- building self-templates"

# tid prefix must not collide with wave 1's z000-z099
$PY $REPO/scripts/build_selfpose_tmpl2.py --run selftm2 --tid-prefix 8 --tid-start 300 --seeds 50 2>&1 | tee -a "$REPO/ops/selftm_chain.log"

sed 's#selftm_queue.txt#selftm2_queue.txt#; s#selftm_locks#selftm2_locks#' \
    $REPO/ops/selftm_worker.sh > $REPO/ops/selftm2_worker.sh
for g in 0 1 2 3; do
    tmux new-session -d -s selftm2w$g -c $REPO
    tmux send-keys -t selftm2w$g "bash $REPO/ops/selftm2_worker.sh $g 2>&1 | tee -a $REPO/ops/selftm2w$g.log" Enter
done
say "wave 2 stage 2 launched on 4 gpus"
