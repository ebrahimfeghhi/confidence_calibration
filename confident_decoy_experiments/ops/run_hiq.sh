#!/bin/bash
# Replicate allpose8che10's single best donor (137_3) with its ORIGINAL template json and ORIGINAL
# seed list, mask ON then mask OFF. Arm 1 is a pure replication of the run that produced the project's
# headline 8che number (out_max 0.958 / selected 0.938), so it also tests reproducibility.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=${GPU:-2}
run_arm() {
    echo "### START $1  MULTICHAIN_MASK=$2  $(date +%H:%M)"
    PROTENIX_MULTICHAIN_MASK=$2 $PY $REPO/scripts/sweep_sched.py \
        --jobs $A/$1_jobs.json --out $A/sweep_out/$1 --logs $A/$1_logs \
        --status $A/$1_status.json --gpus $GPU --n-sample 5 --extra-args ""
    x=$(grep -h "PROTENIX_MULTICHAIN_MASK=0" $A/$1_logs/*.log 2>/dev/null | wc -l)
    n=$(grep -h "Found 0 templates" $A/$1_logs/*.log 2>/dev/null | wc -l)
    [ "$2" = "0" ] && [ "$x" -eq 0 ] && echo "*** $1: mask-OFF but no xchain line ***"
    [ "$2" = "1" ] && [ "$x" -gt 0 ] && echo "*** $1: mask-ON but xchain line present ***"
    [ "$n" -gt 0 ] && echo "*** $1: 'Found 0 templates' x$n -- UNTEMPLATED ***"
    for f in $(find $A/sweep_out/$1 -name '*_sample_0.cif'); do
        c=$(grep -c 'nan nan nan' "$f"); [ "$c" -gt 0 ] && echo "*** $1: NaN in $(basename $f) ***"
    done
    echo "### DONE $1 poses=$(find $A/sweep_out/$1 -name '*_sample_*.cif' 2>/dev/null | wc -l) $(date +%H:%M)"
}
run_arm hiq_maskon  1
run_arm hiq_maskoff 0
echo "### COMPLETE $(date +%H:%M)"
