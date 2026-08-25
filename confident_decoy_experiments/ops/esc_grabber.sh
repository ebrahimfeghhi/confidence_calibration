#!/bin/bash
# Every 5 min: if we hold fewer than MAX_GPUS and a card has NO other user's process, start a worker
# on it. Never touches a GPU anyone else is using, and never exceeds MAX_GPUS.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
LOG=$REPO/ops/esc_grabber.log
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
ALLOWED="${ESC_GPUS:-1 5 6 7}"
say "grabber started (pinned to gpus: $ALLOWED; max $MAX_GPUS; every 5 min; only fully-free cards)"
while true; do
    # done?
    left=0
    for f in $REPO/artifacts/esc_*_jobs.json; do
        r=$(basename "$f" _jobs.json)
        want=$(/home/ebrahim/Protenix/.venv/bin/python -c \
            "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $REPO/artifacts/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -lt "$want" ] && left=$((left+1))
    done
    if [ "$left" -eq 0 ]; then say "all arms complete -- grabber exiting"; exit 0; fi
    held=$(tmux ls 2>/dev/null | grep -c '^escw[0-9]*:' || true)
    if [ "$held" -ge "$MAX_GPUS" ]; then
        say "holding $held gpu(s), at cap; $left arms left"; sleep 300; continue
    fi
    for g in $ALLOWED; do
        tmux has-session -t escw$g 2>/dev/null && continue
        o=$(gpu_owner $g)
        if [ -z "$o" ]; then
            say "gpu$g is free -> starting worker (held $held -> $((held+1)))"
            tmux new-session -d -s escw$g -c $REPO
            tmux send-keys -t escw$g "bash $REPO/ops/esc_worker.sh $g 2>&1 | tee -a $REPO/ops/escw$g.log" Enter
            held=$((held+1))
            [ "$held" -ge "$MAX_GPUS" ] && break
        fi
    done
    say "holding $held gpu(s); $left arms left"
    sleep 300
done
