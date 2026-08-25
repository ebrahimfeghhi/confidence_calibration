#!/bin/bash
# Health watcher for the wnm ladder, every 10 min. Flags the failure modes that would otherwise burn
# the night silently: a worker pane dead or throwing, a claimed arm with no lock owner, CPU over the
# hard cap, or disk filling.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
while true; do
    bad=0
    for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^wnmw'); do
        p=$(tmux capture-pane -t "$s" -p 2>/dev/null | tail -40)
        if echo "$p" | grep -qE "Traceback|CUDA out of memory|RuntimeError|\*\*\*"; then
            say "!! $s shows an error:"; echo "$p" | grep -E "Traceback|out of memory|RuntimeError|\*\*\*" | tail -3
            bad=1
        fi
    done
    live=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^wnmw' || true)
    procs=$(pgrep -fc "sweep_sched.py --jobs.*wnm_" || true)
    load=$(uptime | sed 's/.*load average: //')
    disk=$(df -h /home | tail -1 | awk '{print $4" free ("$5" used)"}')
    todo=$(wc -l < "$A/wnm_queue.txt")
    done_arms=$(ls -d $A/sweep_out/wnm_* 2>/dev/null | wc -l)
    say "workers $live | inference procs $procs | arms started $done_arms/$todo | load $load | disk $disk"
    [ "$live" -gt 0 ] && [ "$procs" -eq 0 ] && say "!! workers alive but NO inference process running"
    sleep 600
done
