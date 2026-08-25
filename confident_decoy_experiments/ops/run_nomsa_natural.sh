#!/bin/bash
# 8slb no-MSA WITH the natural hmmsearch templates -- the control for whether removing the MSA alone
# rescues 8slb, or whether the donor template was doing the work.
# Waits for the 9jbq full-MSA halves to release GPU 4.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python; A=$REPO/artifacts
echo "### waiting for jbq_nodrop_fullmsa_a to finish on GPU 4 ..."
while [ "$(find $A/sweep_out/jbq_nodrop_fullmsa_a -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 25 ]; do sleep 20; done
echo "### START nomsa_natural8slb $(date +%H:%M)"
$PY $REPO/scripts/sweep_sched.py --jobs $A/nomsa_natural8slb_jobs.json --out $A/sweep_out/nomsa_natural8slb \
    --logs $A/nomsa_natural8slb_logs --status $A/nomsa_natural8slb_status.json --gpus 4 --n-sample 5 --extra-args ""
m=$(grep -ho "N_msa [0-9]*" $A/nomsa_natural8slb_logs/*.log 2>/dev/null|sort -u|tr '\n' ' ')
t=$(grep -ho "Found [0-9]* templates" $A/nomsa_natural8slb_logs/*.log 2>/dev/null|sort|uniq -c|tr '\n' ' ')
[ "$m" = "N_msa 1 " ] || echo "*** nomsa_natural8slb: expected N_msa 1, got '$m' ***"
echo "### DONE nomsa_natural8slb poses=$(find $A/sweep_out/nomsa_natural8slb -name '*_sample_*.cif' 2>/dev/null|wc -l) [N_msa: $m | templates: $t] $(date +%H:%M)"
