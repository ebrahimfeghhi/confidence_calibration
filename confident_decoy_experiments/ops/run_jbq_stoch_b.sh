#!/bin/bash
# GPU 3 half: waits for the 8che full-MSA arm to release the card, then runs the remaining
# 9jbq cells (dropout-off + subsampled, and the second half of dropout-off + full MSA).
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
echo "### waiting for nodrop_fullmsa10 to finish on GPU 3 ..."
while [ "$(find $A/sweep_out/nodrop_fullmsa10 -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 25 ]; do sleep 20; done
echo "### GPU 3 free $(date +%H:%M)"
bash $REPO/ops/run_jbq_stoch.sh 3 \
    jbq_nodrop_sub "--mc_dropout_apply_rate 0" \
    jbq_nodrop_fullmsa_b "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
