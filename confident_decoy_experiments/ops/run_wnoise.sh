#!/bin/bash
# Weight-noise sweep: can perturbing weights substitute for the trunk diversity that MSA subsampling
# supplies? MSA subsampling OFF (full MSA) and dropout OFF throughout, donor 137_3 template, mask ON.
# Reference points for this donor/seed set: subsampled 0.942 | full MSA 0.089.
# Each condition gets its own noise draw per seed (PROTENIX_WNOISE_SEED = the run seed).
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    sig=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['sigma'])")
    scope=$($PY -c "import json;print(json.load(open('$A/wnoise_manifest.json'))['$run']['scope'])")
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 25 ] && { echo "### SKIP $run (done)"; continue; }
    echo "### START $run sigma=$sig scope=$scope GPU$GPU $(date +%H:%M)"
    PROTENIX_WNOISE=$sig PROTENIX_WNOISE_SCOPE=$scope \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
    w=$(grep -h "\[wnoise\]" $A/${run}_logs/*.log 2>/dev/null | tail -1)
    [ -z "$w" ] && echo "*** $run: NO [wnoise] line -- weights were NOT perturbed ***"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l) | $w"
done
echo "### SESSION DONE GPU$GPU $(date +%H:%M)"
