#!/bin/bash
# 30-minute health check for the run queue. Complements the other two watchers:
#   ops/gpu_grabber.sh   (5 min) - adopts free GPUs, REVIVES a dead scheduler, exits when queue done
#   ops/score_watcher.sh (5 min) - scores each arm as it completes, verifies conditions first
# This one answers the question neither of those does: "is the run actually PROGRESSING?"
# A worker can be alive and burning GPU while writing nothing (hung CUDA op, stuck NFS read),
# which looks healthy to a process check and to the scheduler.
#
# Everything is reported per tick so the log doubles as a progress record.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
OLD=/home/ebrahim/confident_decoy_experiments/artifacts
LOG=$REPO/ops/health_watcher.log
INTERVAL=1800
STALL_MIN=25          # no new seed written in this many minutes => stall

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

active_arm() {  # the arm the scheduler is currently running
    ps -u "$(id -un)" -o args= | grep '[s]weep_sched' \
      | grep -oE '/[a-z0-9]+_jobs\.json' | head -1 | sed 's|/||;s|_jobs.json||'
}

say "health watcher started (every $((INTERVAL/60)) min; stall threshold ${STALL_MIN} min)"

while true; do
    R=$(active_arm)
    if [ -z "$R" ]; then
        if pgrep -u "$(id -un)" -f sweep_sched >/dev/null 2>&1; then
            say "scheduler present but arm unresolved -- skipping this tick"
        else
            say "NO SCHEDULER RUNNING (gpu_grabber should revive it within ~10 min)"
        fi
        sleep "$INTERVAL"; continue
    fi

    OUT=$OLD/sweep_out/$R
    n=$(find "$OUT" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l)
    want=$(python3 -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$OLD/${R}_jobs.json'))))" 2>/dev/null)
    newest=$(find "$OUT" -name '*_summary_confidence_sample_4.json' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    age=$(python3 -c "
import time
t=${newest:-0}
print('never' if t==0 else f'{(time.time()-t)/60:.0f}')" 2>/dev/null)

    workers=$(ps -u "$(id -un)" -o args= | grep -c '[i]nference.py')
    gpus=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F, '$2+0>50{c++} END{print c+0}')

    # --- STALL: workers alive but nothing written recently -------------------
    if [ "$age" != "never" ] && [ "$age" -gt "$STALL_MIN" ] 2>/dev/null; then
        say "*** STALL: $R has written nothing for ${age} min ($n/$want seeds, $workers workers, ${gpus} GPUs busy) ***"
    fi

    # --- condition checks on the ACTIVE arm ---------------------------------
    L=$OLD/${R}_logs
    nulls=$(grep -h "Found 0 templates" "$L"/*.log 2>/dev/null | wc -l)
    case "$R" in
        allpose*)
            [ "$nulls" -gt 0 ] && say "*** ALERT $R: 'Found 0 templates' x$nulls -- UNTEMPLATED ARM ***" ;;
    esac
    case "$R" in
        *nomsa)
            msa=$(grep -ho "N_msa [0-9]*" "$L"/*.log 2>/dev/null | sort -u | tr '\n' ' ')
            [ "$msa" = "N_msa 1 " ] || say "*** ALERT $R: expected 'N_msa 1', got '$msa' ***" ;;
    esac
    oom=$(grep -il "out of memory" "$L"/*.log 2>/dev/null | wc -l)
    [ "$oom" -gt 0 ] && say "*** ALERT $R: CUDA OOM in $oom log(s) ***"

    # --- disk (a full quota killed every job on this box once) ---------------
    freeg=$(quota 2>/dev/null | awk 'NR==3{printf "%d", ($4-$2)/1048576}')
    [ -n "${freeg:-}" ] && [ "$freeg" -lt 30 ] 2>/dev/null && \
        say "*** ALERT: only ${freeg} GB quota headroom ***"

    say "$R: $n/$want seeds | last write ${age} min ago | $workers workers | ${gpus} GPUs >50% | ${freeg:-?} GB free"
    sleep "$INTERVAL"
done
