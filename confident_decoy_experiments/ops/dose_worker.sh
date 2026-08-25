#!/bin/bash
# One worker per GPU for the dose-response sweep. Claims arms in dose_queue.txt ORDER, which is
# interface-major (all 6 sigmas of interface 1, then interface 2, ...). That ordering matters: if the
# sweep is cut short, what exists on disk is a set of COMPLETE dose-response curves on a subsample,
# rather than one sigma measured everywhere and the rest empty.
# Claiming is mkdir (atomic) so workers added later never double-claim.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; LOCK=$A/dose_locks; mkdir -p "$LOCK"
GPU=$1
say(){ echo "[$(date +%H:%M:%S)] gpu$GPU $*"; }
say "dose worker started"
while true; do
    claimed=""
    while read -r r; do
        [ -z "$r" ] && continue
        f="$A/${r}_jobs.json"; [ -f "$f" ] || continue
        want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -ge "$want" ] && continue
        mkdir "$LOCK/$r" 2>/dev/null || continue        # someone else has it
        claimed=$r; break
    done < "$A/dose_queue.txt"
    if [ -z "$claimed" ]; then
        # the extender appends new rungs to the queue, so idle != finished; wait and re-check.
        say "no unclaimed work; sleeping 5 min in case the extender adds rungs"
        sleep 300; continue
    fi
    say "claimed $claimed"
    bash $REPO/ops/run_dose.sh $GPU $claimed
    rmdir "$LOCK/$claimed" 2>/dev/null || true
done
