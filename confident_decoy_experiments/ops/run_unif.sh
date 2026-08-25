#!/bin/bash
# Varying-sigma weight noise: sigma ~ U[0, 5e-2] redrawn EVERY recycle, MSA module only.
# Rationale: MSA subsampling's perturbation magnitude is itself random per cycle (uniform on
# [1,n] rows), so some cycles are near-no-ops and others near-total ablation. Fixed-sigma noise
# has no such spread. This arm matches that structure; 5e-2 is the top because the fixed ladder
# degrades there but has not collapsed (1e-1 is dead and would poison cycles).
# Config otherwise identical to the 1e-4 arm: subsampling OFF, dropout ON.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${run}_jobs.json')))*5)")
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge "$want" ] && { echo "### SKIP $run ($n/$want)"; continue; }
    echo "### START $run sigma~U[0,0.05] per-cycle scope=msa dropout=ON subsampling=OFF GPU$GPU $(date +%H:%M)"
    PROTENIX_WNOISE=0.05 PROTENIX_WNOISE_UNIFORM=0.05 \
    PROTENIX_WNOISE_SCOPE=msa PROTENIX_WNOISE_PERCYCLE=1 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--data.msa.min_size.test 16384"
    w=$(grep -h "wnoise-percycle" $A/${run}_logs/*.log 2>/dev/null | tail -1)
    [ -z "$w" ] && echo "*** $run: NO [wnoise-percycle] line -- noise NOT applied ***"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l)"
done
