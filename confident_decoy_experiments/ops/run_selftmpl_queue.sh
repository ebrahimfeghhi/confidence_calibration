#!/bin/bash
# Queue on GPU 2, after the running d3x10 (mask ON) finishes:
#   1. d3 template, 10 seeds, mask OFF   -- the mask-off twin of the run in flight
#   2. seed-137 sample-3 template (DockQ 0.398 AE / 0.529 BE), 10 seeds, mask OFF
#
# Same seeds (48-57) and stock config throughout, so the only variables are the template and the mask.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=${GPU:-2}

echo "### waiting for d3x10 (mask ON) to finish ..."
while [ "$(find $A/sweep_out/selftmpl8che_d3x10 -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 50 ]; do sleep 30; done
echo "### d3x10 done $(date +%H:%M)"

run_arm() {   # $1=jobs stem  $2=out stem  $3=mask value
    echo "### START $2  MULTICHAIN_MASK=$3  $(date +%H:%M)"
    PROTENIX_MULTICHAIN_MASK=$3 $PY $REPO/scripts/sweep_sched.py \
        --jobs $A/$1_jobs.json --out $A/sweep_out/$2 \
        --logs $A/$2_logs --status $A/$2_status.json \
        --gpus $GPU --n-sample 5 --extra-args ""
    # condition checks -- a silently-wrong arm is this project's recurring failure mode
    x=$(grep -h "PROTENIX_MULTICHAIN_MASK=0" $A/$2_logs/*.log 2>/dev/null | wc -l)
    n=$(grep -h "Found 0 templates" $A/$2_logs/*.log 2>/dev/null | wc -l)
    [ "$3" = "0" ] && [ "$x" -eq 0 ] && echo "*** $2: mask-OFF arm but no xchain line -- patch did not apply ***"
    [ "$3" = "1" ] && [ "$x" -gt 0 ] && echo "*** $2: mask-ON arm but xchain line present -- env leaked ***"
    [ "$n" -gt 0 ] && echo "*** $2: 'Found 0 templates' x$n -- UNTEMPLATED ***"
    nan=0
    for f in $(find $A/sweep_out/$2 -name '*_sample_0.cif'); do
        c=$(grep -c 'nan nan nan' "$f"); [ "$c" -gt 0 ] && nan=$((nan+1))
    done
    [ "$nan" -gt 0 ] && echo "*** $2: $nan seed(s) produced NaN coordinates ***"
    echo "### DONE $2  poses=$(find $A/sweep_out/$2 -name '*_sample_*.cif' 2>/dev/null | wc -l)  $(date +%H:%M)"
}

run_arm selftmpl8che_d3x10_off selftmpl8che_d3x10_off 0
run_arm s137                   s137_off               0
echo "### QUEUE COMPLETE $(date +%H:%M)"
