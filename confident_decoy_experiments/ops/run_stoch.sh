#!/bin/bash
# Isolate which stochastic source rescues 8che donor 137_3: MC dropout or MSA subsampling.
# Same donor, same 10 seeds, mask ON throughout -- only the two knobs vary.
#   dropout 0.4 + subsampled  = stock, already run as rep137_3_maskon (selected 0.942)
#   dropout 0.4 + full MSA    -> drop_fullmsa
#   dropout 0   + subsampled  -> nodrop_subsample
#   dropout 0   + full MSA    -> nodrop_fullmsa10  (10-seed version of det137_3)
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=${GPU:-3}
run() {
    echo "### START $1  [$2]  $(date +%H:%M)"
    $PY $REPO/scripts/sweep_sched.py --jobs $A/$1_jobs.json --out $A/sweep_out/$1 \
        --logs $A/$1_logs --status $A/$1_status.json --gpus $GPU --n-sample 5 --extra-args "$2"
    m=$(grep -ho "N_msa [0-9]*" $A/$1_logs/*.log 2>/dev/null | sort -u | tr '\n' ' ')
    n=$(grep -h "Found 0 templates" $A/$1_logs/*.log 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "*** $1: UNTEMPLATED x$n ***"
    echo "### DONE $1 poses=$(find $A/sweep_out/$1 -name '*_sample_*.cif' 2>/dev/null|wc -l) [N_msa: $m] $(date +%H:%M)"
}
run drop_fullmsa      "--data.msa.min_size.test 16384"
run nodrop_subsample  "--mc_dropout_apply_rate 0"
run nodrop_fullmsa10  "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
echo "### ALL DONE $(date +%H:%M)"
