#!/bin/bash
# Run the arms in ops/run_queue.txt, IN ORDER, re-reading the file every iteration.
#
# WHY NOT `for R in $(cat queue)`: that expands the list ONCE at launch, so any arm appended later is
# silently skipped. That happened twice -- allpose9jbq10nomsa was added 10 minutes after a launch and
# never ran, and allpose8av210nomsa/allpose8che10nomsa were queued the same way. This loop picks the
# first INCOMPLETE arm from the file on every pass, so edits take effect at the next arm boundary
# without a restart, and the file is always the source of truth for order.
#
# Completion is per-arm: an arm is done when it has as many *_summary_confidence_sample_4.json files
# as its jobs.json declares seeds. sweep_sched itself skips finished jobs, so re-entering a partially
# complete arm is free.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
A=/home/ebrahim/confident_decoy_experiments/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
QUEUE=$REPO/ops/run_queue.txt
GPUS=${GPUS:-5,7,4,3}          # pinned, never derived from ps (an empty value silently breaks every arm)
NSAMPLE=5

want() { $PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/$1_jobs.json'))))" 2>/dev/null; }
have() { find "$A/sweep_out/$1" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l; }

echo "### chain started $(date +%H:%M) on GPUs $GPUS (queue re-read every arm)"
while true; do
    NEXT=""
    while read -r R; do
        [ -z "$R" ] && continue
        w=$(want "$R"); [ -z "$w" ] && { echo "### SKIP $R (no jobs.json)"; continue; }
        h=$(have "$R")
        if [ "$h" -lt "$w" ]; then NEXT=$R; NW=$w; NH=$h; break; fi
    done < <(grep -vE '^\s*(#|$)' "$QUEUE")

    if [ -z "$NEXT" ]; then
        echo "### ALL ARMS COMPLETE $(date +%H:%M)"; exit 0
    fi

    echo "### START $NEXT $(date +%H:%M)  ($NH/$NW seeds already done)"
    $PY "$REPO/scripts/sweep_sched.py" --jobs "$A/${NEXT}_jobs.json" --out "$A/sweep_out/$NEXT" \
        --logs "$A/${NEXT}_logs" --status "$A/${NEXT}_status.json" --gpus "$GPUS" --n-sample "$NSAMPLE"
    echo "### DONE $NEXT $(date +%H:%M)"

    # guard against a spin loop if an arm can never reach its declared seed count
    if [ "$(have "$NEXT")" -lt "$(want "$NEXT")" ]; then
        echo "### WARNING $NEXT ended still incomplete ($(have "$NEXT")/$(want "$NEXT")) -- pausing 5 min"
        sleep 300
    fi
done
