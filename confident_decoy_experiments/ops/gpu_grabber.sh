#!/bin/bash
# Watch for GPUs that become fully free and expand the allpose sweep onto them, up to MAX_GPUS.
#
# "Open" = NO process from any other user on the card. Free memory alone does not qualify --
# we do not run on top of john/kao.
#
# Restarting the scheduler is cheap: sweep_sched completion is per-seed and requires the final
# *_summary_confidence_sample_4.json, so only in-flight seeds are redone.

set -uo pipefail

ME=$(id -un)
MAX_GPUS=4
INTERVAL=300              # 5 min
COOLDOWN=600              # do not restart more than once per 10 min
PROJ=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
SESSION=gpu0
# Arms come from an APPENDABLE queue file, not a hardcoded list: score_watcher.sh appends a
# <run>nomsa arm whenever an arm's pooled ipTM selection fails, and that must be picked up without
# restarting a chain that is mid-run. Completed arms are re-listed harmlessly -- the scheduler skips
# finished jobs for free.
QUEUE=/home/ebrahim/tts_abag/confident_decoy_experiments/ops/run_queue.txt
arms() { grep -vE '^\s*(#|$)' "$QUEUE" | tr '\n' ' '; }
# Repo rule: everything this project produces lives under tts_abag, never $HOME.
LOG=/home/ebrahim/tts_abag/confident_decoy_experiments/ops/gpu_grabber.log
# PINNED, not resolved via paths.py. The naboo transfer is creating
# <repo>/artifacts, and paths.py prefers repo-local the moment it exists -- so a
# relaunch that recomputed EXP would point at the half-copied tree, where these
# arms' *_jobs.json do not exist, and the sweep would die silently.
A=/home/ebrahim/confident_decoy_experiments/artifacts

last_restart=0

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# GPUs that have no process belonging to another user.
open_gpus() {
    local out=()
    for i in 0 1 2 3 4 5 6 7; do
        local uu procs foreign=0
        uu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$i" 2>/dev/null) || continue
        procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -F "$uu")
        while IFS=, read -r _u p; do
            p=$(echo "$p" | tr -d ' ')
            [ -z "$p" ] && continue
            local owner
            owner=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
            [ -n "$owner" ] && [ "$owner" != "$ME" ] && foreign=1
        done <<< "$procs"
        [ "$foreign" -eq 0 ] && out+=("$i")
    done
    echo "${out[*]:-}"
}

# GPU list the running scheduler was launched with.
current_gpus() {
    ps -u "$ME" -o args= 2>/dev/null | grep '[s]weep_sched' | grep -o -- '--gpus [0-9,]*' | head -1 | awk '{print $2}'
}

relaunch() {
    local list="$1"
    say "RESTART -> --gpus $list"
    tmux kill-session -t "$SESSION" 2>/dev/null
    local waited=0
    while ps -u "$ME" -o args= 2>/dev/null | grep -q '[i]nference.py'; do
        sleep 5; waited=$((waited+5))
        [ "$waited" -ge 180 ] && { say "WARN: workers still alive after 180s; continuing anyway"; break; }
    done
    tmux new-session -d -s "$SESSION" -c "$PROJ"
    tmux send-keys -t "$SESSION" "GPUS=$1 bash /home/ebrahim/tts_abag/confident_decoy_experiments/ops/run_chain.sh" Enter
    last_restart=$(date +%s)
}

say "watcher started (max ${MAX_GPUS} GPUs, every $((INTERVAL/60)) min, open = no other user's process)"

while true; do
    # A missing tmux session no longer means "finished" -- the runs must stay up, so recreate it.
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        say "tmux session $SESSION is GONE -- recreating and reviving on GPU(s) ${last_list:-5}"
        relaunch "${last_list:-5}"
        sleep "$INTERVAL"; continue
    fi

    # Disk guard: a full quota is how every job on this box died once before.
    # free = hard limit ($4) - blocks used ($2). NOT $4-$3: the soft quota column is 0 here.
    freeg=$(quota 2>/dev/null | awk 'NR==3{printf "%d", ($4-$2)/1048576}')
    if [ -n "${freeg:-}" ] && [ "$freeg" -lt 30 ] 2>/dev/null; then
        say "*** ALERT: only ${freeg} GB quota headroom left -- runs will fail when it hits 0 ***"
    fi

    # Finished only when EVERY queued arm has all its seeds. Grepping the pane for a specific arm's
    # DONE line would exit early now that arms can be appended after that arm completes.
    pending=""
    for R in $(arms); do
        want=$(python3 -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${R}_jobs.json'))))" 2>/dev/null)
        [ -z "$want" ] && { pending="$pending $R(nojobs)"; continue; }
        have=$(find "$A/sweep_out/$R" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l)
        [ "$have" -lt "$want" ] && pending="$pending $R($have/$want)"
    done
    if [ -z "$pending" ]; then
        say "every queued arm is complete -- watcher exiting"
        exit 0
    fi

    cur=$(current_gpus)
    if [ -z "$cur" ]; then
        # No scheduler but the chain is not finished => it died, or is briefly between arms.
        # Confirm across two consecutive ticks before reviving, so we don't race an arm handoff.
        missing=$((${missing:-0} + 1))
        if [ "$missing" -ge 2 ]; then
            say "SCHEDULER DEAD (${missing} ticks) and chain unfinished -- reviving on GPU(s) ${last_list:-5}"
            relaunch "${last_list:-5}"
            missing=0
        else
            say "no scheduler this tick (arm handoff?) -- rechecking next tick before reviving"
        fi
        sleep "$INTERVAL"; continue
    fi
    missing=0
    last_list="$cur"
    ncur=$(echo "$cur" | tr ',' '\n' | grep -c .)

    # Health flags -- cheap, and these are the failures that hide in plain sight.
    active=$(tmux capture-pane -t "$SESSION" -p -S -400 2>/dev/null | grep -oE 'allpose[a-z0-9]+' | tail -1)
    if [ -n "$active" ] && [ -d "$A/${active}_logs" ]; then
        z=$(grep -h "Found 0 templates" "$A/${active}_logs"/*.log 2>/dev/null | wc -l)
        [ "$z" -gt 0 ] && say "*** ALERT $active: 'Found 0 templates' x$z -- UNTEMPLATED RUN ***"
        o=$(grep -il "out of memory" "$A/${active}_logs"/*.log 2>/dev/null | wc -l)
        [ "$o" -gt 0 ] && say "*** ALERT $active: CUDA OOM in $o log(s) ***"
    fi
    avail=$(quota -s 2>/dev/null | awk 'NR==3{print $1}')
    [ -n "${avail:-}" ] && say "quota: $avail used" >/dev/null

    if [ "$ncur" -ge "$MAX_GPUS" ]; then
        say "on $ncur GPU(s) ($cur) -- at cap, nothing to do"
        sleep "$INTERVAL"; continue
    fi

    opens=$(open_gpus)
    # Candidates = open GPUs we are not already using.
    add=()
    for g in $opens; do
        echo ",$cur," | grep -q ",$g," || add+=("$g")
    done

    if [ "${#add[@]}" -eq 0 ]; then
        say "on $ncur GPU(s) ($cur); no newly open GPU (open now: ${opens:-none})"
    else
        now=$(date +%s)
        if [ $((now - last_restart)) -lt "$COOLDOWN" ]; then
            say "would adopt ${add[*]} but in cooldown; waiting"
        else
            room=$((MAX_GPUS - ncur))
            take=$(printf '%s\n' "${add[@]}" | head -n "$room" | paste -sd, -)
            newlist="$cur,$take"
            say "adopting GPU(s) $take -> now $newlist"
            relaunch "$newlist"
        fi
    fi
    sleep "$INTERVAL"
done
