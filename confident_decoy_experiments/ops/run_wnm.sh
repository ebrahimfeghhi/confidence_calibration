#!/bin/bash
# Runner for the MSA-module weight-noise ladder on the train set.
#
# CONFIG (differs from the dose sweep -- read before reusing):
#   MSA subsampling  OFF  (--data.msa.min_size.test 16384 pins sample_size to the full depth)
#   MC dropout       ON   (stock 0.4 -- deliberately NOT disabled here)
#   weight noise     MSA module only, resampled every recycle from the ORIGINAL weights
#
# The sigma=0 arm is the matched control: same dropout-on / subsampling-off config, no noise.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    sig=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['sigma'])")
    scope=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['scope'])")
    want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${run}_jobs.json')))*5)")
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge "$want" ] && { echo "### SKIP $run (already $n/$want)"; continue; }
    echo "### START $run sigma=$sig scope=$scope dropout=ON subsampling=OFF GPU$GPU $(date +%H:%M)"
    PROTENIX_WNOISE=$sig PROTENIX_WNOISE_SCOPE=$scope PROTENIX_WNOISE_PERCYCLE=1 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--data.msa.min_size.test 16384"
    w=$(grep -h "wnoise-percycle" $A/${run}_logs/*.log 2>/dev/null | tail -1)
    if [ "$sig" = "0.0" ]; then
        [ -n "$w" ] && echo "*** $run: sigma=0 control but noise WAS applied -- control invalid ***"
    else
        [ -z "$w" ] && echo "*** $run: NO [wnoise-percycle] line -- noise NOT applied ***"
    fi
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l)"
done
