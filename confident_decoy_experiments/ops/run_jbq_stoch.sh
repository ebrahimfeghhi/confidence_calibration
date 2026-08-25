#!/bin/bash
# dropout x MSA-subsampling factorial for 9jbq donor 138_1 (donor DockQ 0.041 -> out_sel 0.739).
# The stock cell (dropout 0.4 + subsampled) already exists inside allpose9jbq10 for this donor and
# these exact 10 seeds, so only the other three cells are run here. Mask ON throughout.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python; A=$REPO/artifacts
GPU=$1; shift
while [ $# -gt 0 ]; do
    arm=$1; extra=$2; shift 2
    echo "### START $arm [$extra] on GPU $GPU $(date +%H:%M)"
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${arm}_jobs.json --out $A/sweep_out/$arm \
        --logs $A/${arm}_logs --status $A/${arm}_status.json --gpus $GPU --n-sample 5 --extra-args "$extra"
    m=$(grep -ho "N_msa [0-9]*" $A/${arm}_logs/*.log 2>/dev/null|sort -u|tr '\n' ' ')
    n=$(grep -h "Found 0 templates" $A/${arm}_logs/*.log 2>/dev/null|wc -l)
    [ "$n" -gt 0 ] && echo "*** $arm: UNTEMPLATED x$n ***"
    echo "### DONE $arm poses=$(find $A/sweep_out/$arm -name '*_sample_*.cif' 2>/dev/null|wc -l) [N_msa: $m] $(date +%H:%M)"
done
echo "### SESSION DONE $(date +%H:%M)"
