#!/bin/bash
# One worker per GPU for the MSA-module weight-noise ladder. Claims arms in s1b_queue.txt order
# (complex-major, so partial progress yields complete ladders on whole complexes).
# mkdir claiming is atomic, so workers started later never double-claim.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; LOCK=$A/s1b_locks; mkdir -p "$LOCK"
GPU=$1
say(){ echo "[$(date +%H:%M:%S)] gpu$GPU $*"; }
say "d50 worker started"
while true; do
    claimed=""
    while read -r r; do
        [ -z "$r" ] && continue
        f="$A/${r}_jobs.json"; [ -f "$f" ] || continue
        want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -ge "$want" ] && continue
        mkdir "$LOCK/$r" 2>/dev/null || continue
        claimed=$r; break
    done < "$A/s1b_queue.txt"
    if [ -z "$claimed" ]; then say "no unclaimed both-off work left -- exiting"; exit 0; fi
    say "claimed $claimed"
    bash $REPO/ops/run_stage1.sh $GPU $claimed
    rmdir "$LOCK/$claimed" 2>/dev/null || true
done
