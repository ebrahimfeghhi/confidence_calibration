#!/bin/bash
# Runs lane 1's arms on a working GPU once that GPU has finished its own lane.
# Lane 1 sits on GPU 4, which another user occupies; its scheduler retries every 10 min, so if the
# card frees it may also start these. The skip guard in run_wnoise_pc.sh plus sweep_sched's per-seed
# skip means the worst case is a little duplicated compute, never corruption.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
GPU=$1; OWNLANE=$2; shift 2
echo "### gpu$GPU catch-up armed; waiting for its own lane to finish"
while true; do
    left=0
    for r in $OWNLANE; do
        want=$(/home/ebrahim/Protenix/.venv/bin/python -c \
            "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${r}_jobs.json')))*5)")
        have=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -lt "$want" ] && left=$((left+1))
    done
    [ "$left" -eq 0 ] && break
    sleep 300
done
echo "### gpu$GPU own lane done -- taking lane-1 share $(date +%H:%M)"
bash $REPO/ops/run_wnoise_pc.sh $GPU "$@"
