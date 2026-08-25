#!/bin/bash
# 8bf4, both-off, 10 seeds, run twice: default row shuffle vs PROTENIX_MSA_STRATEGY=topk.
# With subsampling pinned off sample_size == n, so both arms use the IDENTICAL rows and differ
# only in order -- the clean test of whether the shuffle drives seed-to-seed divergence.
# Per-recycle cross-chain distogram is dumped so divergence can be localised to a cycle.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python; GPU=$1
for tag in perm topk; do
    run=msaperm_${tag}_8bf4
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] && { echo "### SKIP $run ($n/50)"; continue; }
    export DUMP_CYCLE_DIR=$A/${run}_cycles
    [ "$tag" = "topk" ] && export PROTENIX_MSA_STRATEGY=topk || unset PROTENIX_MSA_STRATEGY
    echo "### START $run  strategy=${PROTENIX_MSA_STRATEGY:-random}  GPU$GPU $(date +%H:%M)"
    DUMP_CYCLE_TAG=$run \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5 \
        --extra-args "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l) cycles=$(ls $A/${run}_cycles 2>/dev/null|wc -l)"
done
