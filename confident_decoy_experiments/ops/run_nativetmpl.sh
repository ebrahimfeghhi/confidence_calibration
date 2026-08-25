#!/bin/bash
# Native-template factorial: {multichain_mask ON, OFF} x {MSA, no MSA}, 16 training complexes.
#
# THE POINT: a native template fixes each chain's GROUND-TRUTH fold. multichain_mask normally zeroes
# every cross-chain template feature, so the template says nothing about how the chains are arranged.
# Turning it off lets the template carry the true inter-chain geometry -- i.e. the dock. Comparing the
# two tells us whether the trunk CAN be moved to the right pose by a template at all (mask off works)
# or whether templates are simply the wrong channel for docking information (mask off still fails).
#
# mask OFF is an ORACLE condition -- it shows the model the answer. It is a diagnostic ceiling, not a
# method. The mask ON arms are the label-free ceiling for iterative self-templating.
#
# All four arms: dropout 0, MSA subsampling off (full depth), seed 101, 5 diffusion samples.
# With dropout and subsampling off the trunk is deterministic; the 5 samples differ by diffusion noise
# only. That is deliberate -- those 5 poses are exactly what an iterative scheme would template on next.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPUS=${GPUS:-5,7,4,3}
EXTRA='--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384'

# arm | jobs file | PROTENIX_MULTICHAIN_MASK
ARMS=(
  "nat_msa_maskon|nativetmpl|1"
  "nat_msa_maskoff|nativetmpl|0"
  "nat_nomsa_maskon|nativetmplnomsa|1"
  "nat_nomsa_maskoff|nativetmplnomsa|0"
)

echo "### native-template factorial started $(date +%H:%M) on GPUs $GPUS"
for spec in "${ARMS[@]}"; do
    IFS='|' read -r ARM JOBS XMASK <<< "$spec"
    done_n=$(find "$A/sweep_out/$ARM" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l)
    want_n=$($PY -c "import json;print(len(json.load(open('$A/${JOBS}_jobs.json'))))")
    if [ "$done_n" -ge "$want_n" ]; then echo "### SKIP $ARM (already $done_n/$want_n)"; continue; fi

    echo "### START $ARM $(date +%H:%M)  jobs=$JOBS  MULTICHAIN_MASK=$XMASK  ($done_n/$want_n done)"
    PROTENIX_MULTICHAIN_MASK=$XMASK $PY "$REPO/scripts/sweep_sched.py" \
        --jobs "$A/${JOBS}_jobs.json" --out "$A/sweep_out/$ARM" \
        --logs "$A/${ARM}_logs" --status "$A/${ARM}_status.json" \
        --gpus "$GPUS" --n-sample 5 --extra-args "$EXTRA"

    # Condition checks. A silently-wrong arm is the failure mode this project keeps hitting, so verify
    # the two switches that cannot be inferred from the output structures themselves.
    warn=$(grep -h "PROTENIX_MULTICHAIN_MASK=0" "$A/${ARM}_logs"/*.log 2>/dev/null | wc -l)
    if [ "$XMASK" = "0" ] && [ "$warn" -eq 0 ]; then
        echo "*** $ARM: MASK-OFF ARM BUT NO 'PROTENIX_MULTICHAIN_MASK=0' LINE -- patch did not apply ***"
    fi
    if [ "$XMASK" = "1" ] && [ "$warn" -gt 0 ]; then
        echo "*** $ARM: MASK-ON ARM BUT FOUND THE MASK-OFF WARNING -- env leaked ***"
    fi
    nulls=$(grep -h "Found 0 templates" "$A/${ARM}_logs"/*.log 2>/dev/null | wc -l)
    [ "$nulls" -gt 0 ] && echo "*** $ARM: 'Found 0 templates' x$nulls -- UNTEMPLATED ***"
    msa=$(grep -ho "N_msa [0-9]*" "$A/${ARM}_logs"/*.log 2>/dev/null | sort -u | tr '\n' ' ')
    echo "### DONE $ARM $(date +%H:%M)  poses=$(find "$A/sweep_out/$ARM" -name '*_sample_*.cif' 2>/dev/null | wc -l)  [$msa]"
done
echo "### ALL FOUR ARMS COMPLETE $(date +%H:%M)"
