#!/bin/bash
# Second half of nodrop_fullmsa10, on GPU 4, started once nodrop_subsample frees that card.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python; A=$REPO/artifacts
echo "### waiting for nodrop_subsample to finish on GPU 4 ..."
while [ "$(find $A/sweep_out/nodrop_subsample -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 50 ]; do sleep 30; done
echo "### START nodrop_fullmsa10b $(date +%H:%M)"
$PY $REPO/scripts/sweep_sched.py --jobs $A/nodrop_fullmsa10b_jobs.json --out $A/sweep_out/nodrop_fullmsa10b \
    --logs $A/nodrop_fullmsa10b_logs --status $A/nodrop_fullmsa10b_status.json --gpus 4 --n-sample 5 \
    --extra-args "--mc_dropout_apply_rate 0 --data.msa.min_size.test 16384"
m=$(grep -ho "N_msa [0-9]*" $A/nodrop_fullmsa10b_logs/*.log 2>/dev/null|sort -u|tr '\n' ' ')
echo "### DONE nodrop_fullmsa10b poses=$(find $A/sweep_out/nodrop_fullmsa10b -name '*_sample_*.cif' 2>/dev/null|wc -l) [N_msa: $m] $(date +%H:%M)"
