#!/bin/bash
# Fill the two missing mask-OFF cells for 9jbq donor 104_3 (DockQ 0.584, ipTM 0.241).
# The mask-ON cells already exist inside allpose9jbq10 / allpose9jbq10nomsa for this exact donor and
# these exact 10 seeds -- both produced frac_acceptable 0.00 (out_max 0.122 / 0.104).
# ARM is passed in so the two cells can run on separate GPUs concurrently.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
ARM=$1; GPU=$2
echo "### START $ARM on GPU $GPU  MULTICHAIN_MASK=0  $(date +%H:%M)"
PROTENIX_MULTICHAIN_MASK=0 $PY $REPO/scripts/sweep_sched.py \
    --jobs $A/${ARM}_jobs.json --out $A/sweep_out/$ARM --logs $A/${ARM}_logs \
    --status $A/${ARM}_status.json --gpus $GPU --n-sample 5 --extra-args ""
x=$(grep -h "PROTENIX_MULTICHAIN_MASK=0" $A/${ARM}_logs/*.log 2>/dev/null | wc -l)
n=$(grep -h "Found 0 templates" $A/${ARM}_logs/*.log 2>/dev/null | wc -l)
m=$(grep -ho "N_msa [0-9]*" $A/${ARM}_logs/*.log 2>/dev/null | sort -u | tr '\n' ' ')
[ "$x" -eq 0 ] && echo "*** $ARM: mask-OFF but no xchain line ***"
[ "$n" -gt 0 ] && echo "*** $ARM: 'Found 0 templates' x$n -- UNTEMPLATED ***"
case "$ARM" in *nomsa*) [ "$m" = "N_msa 1 " ] || echo "*** $ARM: expected N_msa 1, got '$m' ***";; esac
for f in $(find $A/sweep_out/$ARM -name '*_sample_0.cif'); do
    c=$(grep -c 'nan nan nan' "$f"); [ "$c" -gt 0 ] && echo "*** $ARM: NaN in $(basename $f) ***"
done
echo "### DONE $ARM poses=$(find $A/sweep_out/$ARM -name '*_sample_*.cif' 2>/dev/null | wc -l) [$m] $(date +%H:%M)"
