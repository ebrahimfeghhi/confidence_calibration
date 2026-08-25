#!/bin/bash
# Every 10 min: is generation actually progressing, and do the outputs look sane?
# Checks, in order of how badly each has burned this project before:
#   1. a lane's tmux session died while work remains
#   2. no new pose written anywhere for >20 min (stall)
#   3. NaN coordinates in a finished pose (the LAYERNORM_TYPE class of failure)
#   4. the per-cycle weight-noise line missing from a running arm (noise silently not applied)
#   5. CUDA OOM in any arm log
#   6. disk quota headroom
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
A=$REPO/artifacts
LOG=$REPO/ops/esc_health_watcher.log
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
say "escalation health watcher started (every 10 min; stall threshold 20 min)"
while true; do
    done_n=0; tot_n=0
    for f in $A/esc_*_jobs.json; do
        r=$(basename "$f" _jobs.json)
        want=$(/home/ebrahim/Protenix/.venv/bin/python -c \
              "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        tot_n=$((tot_n+want)); done_n=$((done_n+have))
    done
    # 1. lanes alive?
    # workers are escw<gpu> sessions, created/removed dynamically by the grabber
    live=$(tmux ls 2>/dev/null | grep -c '^escw[0-9]*:' || true)
    [ "$done_n" -lt "$tot_n" ] && [ "$live" -eq 0 ] && \
        say "*** NO WORKERS ALIVE while $done_n/$tot_n poses done ***"
    # 2. stall
    newest=$(find $A/sweep_out -path '*esc_*' -name '*_sample_*.cif' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    age=$(/home/ebrahim/Protenix/.venv/bin/python -c "
import time; t=${newest:-0}
print('never' if t==0 else f'{(time.time()-t)/60:.0f}')")
    [ "$age" != "never" ] && [ "$age" -gt 20 ] 2>/dev/null && \
        say "*** STALL: no new pose for ${age} min ($done_n/$tot_n) ***"
    # 3. NaN coordinates
    nan=0
    for f in $(find $A/sweep_out -path '*esc_*' -name '*_sample_0.cif' 2>/dev/null | head -40); do
        grep -q 'nan nan nan' "$f" && { say "*** NaN COORDS in $f ***"; nan=$((nan+1)); break; }
    done
    # 4. noise actually applied on arms that have logs
    for d in $A/esc_*_logs; do
        r=$(basename "$d" _logs)
        [ -n "$(ls $d/*.log 2>/dev/null)" ] || continue
        [ "$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)" -gt 0 ] || continue
        grep -q "wnoise-percycle" $d/*.log 2>/dev/null || say "*** $r: NO per-cycle noise line ***"
    done
    # 5. OOM
    oom=$(grep -lc "out of memory" $A/esc_*_logs/*.log 2>/dev/null | wc -l)
    [ "$oom" -gt 0 ] && say "*** CUDA OOM in $oom log(s) ***"
    # 6. disk
    freeg=$(quota 2>/dev/null | awk 'NR==3{printf "%d", ($4-$2)/1048576}')
    [ -n "${freeg:-}" ] && [ "$freeg" -lt 30 ] 2>/dev/null && say "*** only ${freeg} GB quota left ***"
    # report only the cards we actually hold (a worker session exists for them)
    ours=$(tmux ls 2>/dev/null | grep -oE '^escw[0-9]+' | grep -oE '[0-9]+$' | tr '\n' ',' | sed 's/,$//')
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader ${ours:+-i $ours} | tr '\n' ' ')
    say "$done_n/$tot_n poses | lanes $live/4 | last write ${age} min | gpu: $gpu | ${freeg:-?} GB free"
    sleep 600
done
