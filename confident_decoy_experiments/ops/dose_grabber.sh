#!/bin/bash
# Every 5 min: if we hold fewer than MAX_GPUS and a card has NO other user's process, start a worker
# on it. Never touches a GPU anyone else is using, and never exceeds MAX_GPUS.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
LOG=$REPO/ops/dose_grabber.log
ME=$(id -un); MAX_GPUS=4
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
gpu_owner(){ # users with a process on gpu $1, excluding us
    local uu procs owners=""
    uu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" 2>/dev/null) || return
    procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$uu" | cut -d, -f2)
    for p in $procs; do
        local o; o=$(ps -o user= -p "$(echo $p|tr -d ' ')" 2>/dev/null | tr -d ' ')
        [ -n "$o" ] && [ "$o" != "$ME" ] && owners="$owners $o"
    done
    echo "$owners"
}
# Pinned to the cards we already hold. We do NOT acquire beyond these; the grabber's only job now is
# to RESTART a worker on one of them if its worker exits while arms remain, so a freed card in this
# set gets reused instead of idling. Never exceeds MAX_GPUS and never touches anyone else's card.
ALLOWED="${DOSE_GPUS:-1 5 6 7}"
say "grabber started (pinned to gpus: $ALLOWED; max $MAX_GPUS; every 5 min; only fully-free cards)"
while true; do
    # done?
    left=0
    for f in $(sed "s|^|$REPO/artifacts/|;s|$|_jobs.json|" $REPO/artifacts/dose_queue.txt); do
        r=$(basename "$f" _jobs.json)
        want=$(/home/ebrahim/Protenix/.venv/bin/python -c \
            "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $REPO/artifacts/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -lt "$want" ] && left=$((left+1))
    done
    # NOT an exit: the extender appends new rungs to dose_queue.txt, so "nothing left" is often just
    # a lull between ladder extensions. Exiting here would strand the sweep with no workers.
    if [ "$left" -eq 0 ]; then say "all queued arms complete -- idling in case the ladder extends"; sleep 600; continue; fi
    held=$(tmux ls 2>/dev/null | grep -c '^dosew[0-9]*:' || true)
    if [ "$held" -ge "$MAX_GPUS" ]; then
        say "holding $held gpu(s), at cap; $left arms left"; sleep 300; continue
    fi
    for g in $ALLOWED; do
        tmux has-session -t dosew$g 2>/dev/null && continue
        o=$(gpu_owner $g)
        # Also require the card to be genuinely IDLE, not just free of other users. The retiring
        # 1e-2 workers are OUR user, so gpu_owner ignores them -- starting here on that basis alone
        # would double-book the card and OOM both jobs.
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
        if [ -z "$o" ] && [ "$mem" -lt 2000 ]; then
            say "gpu$g is free -> starting worker (held $held -> $((held+1)))"
            tmux new-session -d -s dosew$g -c $REPO
            tmux send-keys -t dosew$g "bash $REPO/ops/dose_worker.sh $g 2>&1 | tee -a $REPO/ops/dosew$g.log" Enter
            held=$((held+1))
            [ "$held" -ge "$MAX_GPUS" ] && break
        fi
    done
    say "holding $held gpu(s); $left arms left"
    sleep 300
done
