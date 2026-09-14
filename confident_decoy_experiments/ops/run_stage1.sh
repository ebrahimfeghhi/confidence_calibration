#!/bin/bash
# Stage 1: one seed per complex with the MSA cut to its FIRST ROW ONLY (the query sequence), so the
# model folds with zero co-evolutionary evidence. Dropout off. The 5 diffusion samples become the
# templates for stage 2.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && { echo "### SKIP $run ($n/5)"; continue; }
    echo "### START $run MSA=1row dropout=OFF GPU$GPU $(date +%H:%M)"
    PROTENIX_MSA_NROWS=1 PROTENIX_MSA_STRATEGY=topk \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--mc_dropout_apply_rate 0"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l)"
done
