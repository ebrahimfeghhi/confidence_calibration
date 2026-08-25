#!/bin/bash
# Same as run_wnoise.sh but with PROTENIX_WNOISE_PERCYCLE=1: fresh weight noise every recycle,
# drawn from the original weights each time (not compounding).
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    sig=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['sigma'])")
    scope=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['scope'])")
    want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${run}_jobs.json')))*5)")
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    # Skip if another lane already finished it -- lane 1 is blocked behind another user on GPU 4 and
    # its arms are mirrored onto the free lanes, so both could reach the same arm.
    [ "$n" -ge "$want" ] && { echo "### SKIP $run (already $n/$want)"; continue; }
    echo "### START $run sigma=$sig scope=$scope PER-CYCLE GPU$GPU $(date +%H:%M)"
    PROTENIX_WNOISE=$sig PROTENIX_WNOISE_SCOPE=$scope PROTENIX_WNOISE_PERCYCLE=1 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
    w=$(grep -h "wnoise-percycle" $A/${run}_logs/*.log 2>/dev/null | tail -1)
    [ -z "$w" ] && echo "*** $run: NO [wnoise-percycle] line -- per-cycle noise NOT applied ***"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l) | $w"
done
