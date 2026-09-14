#!/bin/bash
# Stock config PLUS a static weight perturbation: MSA subsampling ON, MC dropout ON, and Gaussian
# noise sigma=1e-4 on ALL weights, drawn ONCE PER SEED and held fixed across all 10 recycles.
#
# Per-seed rather than per-process: every seed of a job runs in one process, so perturbing after
# model.eval() would give the whole 50-seed pool one identical perturbed model -- a constant offset
# instead of a diversity source. PROTENIX_WNOISE_PERSEED redraws at cycle 0 of each forward pass.
#
# No --extra-args on purpose: leaving subsampling and dropout at their defaults is the point.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 250 ] && { echo "### SKIP $run ($n/250)"; continue; }
    echo "### START $run sigma=1e-4 scope=all PER-SEED subsampling=ON dropout=ON GPU$GPU $(date +%H:%M)"
    PROTENIX_WNOISE=1e-4 PROTENIX_WNOISE_SCOPE=all PROTENIX_WNOISE_PERSEED=1 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5
    w=$(grep -h "wnoise-percycle" $A/${run}_logs/*.log 2>/dev/null | tail -1)
    [ -z "$w" ] && echo "*** $run: NO wnoise line -- noise NOT applied ***"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l) | $w"
done
