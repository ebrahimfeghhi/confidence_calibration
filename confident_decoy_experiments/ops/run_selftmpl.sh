#!/bin/bash
# Diffusion-feedback loop on 8che, one GPU.
#
#   pass 1: 10 recycles -> diffusion -> 5 poses          (no self-template)
#   pass 2: for EACH of those 5 poses, template on it and run 10 more recycles -> diffusion -> 5 poses
#
# So the 5 poses of pass 1 become 5 independent continuations, 25 output poses total.
#
# CROSS-CHAIN MASK IS OFF (PROTENIX_MULTICHAIN_MASK=0) by request: the fed-back pose therefore carries
# its inter-chain geometry (the dock), not just the monomer folds. This is the whole point -- with the
# mask on, a self-template cannot transmit where the antigen sat.
#
# CAVEAT vs the in-graph version: this restarts the trunk (z=0) for pass 2 rather than continuing from
# the z reached at cycle 10. Same information reaches the trunk, but as a fresh prior rather than an
# accumulated state.
#
# N_cycle=10 is 2.5x the trained default of 4 (configs_base.py:249) -- deliberately off-distribution,
# matching the requested design.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=${GPU:-2}
SEED=${SEED:-48}
NCYCLE=${NCYCLE:-10}
RUN=selftmpl8che

P1=$A/selftmpl/pass1
echo "### waiting for pass 1 (5 poses) ..."
while [ "$(find $P1 -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 5 ]; do sleep 20; done
CIFS=$(find $P1 -name '*_sample_*.cif' | sort)
echo "### pass 1 done:"; echo "$CIFS" | sed 's|.*/|    |'

echo "### building 5 self-templates"
$PY $REPO/scripts/build_selftmpl.py --pdb 8che --run $RUN --tid-prefix w \
    --cifs $CIFS --seeds $SEED || exit 1

echo "### pass 2: 5 branches, cross-chain mask OFF, N_cycle=$NCYCLE"
PROTENIX_MULTICHAIN_MASK=0 $PY $REPO/scripts/sweep_sched.py \
    --jobs $A/${RUN}_jobs.json --out $A/sweep_out/$RUN \
    --logs $A/${RUN}_logs --status $A/${RUN}_status.json \
    --gpus $GPU --n-sample 5 --extra-args ""
# STOCK config on purpose (dropout 0.4, MSA subsampling on). An earlier attempt disabled both to make
# the trunk deterministic, and that killed 8che's good-pose lottery outright -- all 5 pass-1 samples
# came back 0.004-0.069. Those two knobs are what generate the good basin here, so the feedback test
# has to run in the regime where good poses actually occur.

echo "### condition checks"
x=$(grep -h "PROTENIX_MULTICHAIN_MASK=0" $A/${RUN}_logs/*.log 2>/dev/null | wc -l)
n=$(grep -h "Found 0 templates" $A/${RUN}_logs/*.log 2>/dev/null | wc -l)
[ "$x" -eq 0 ] && echo "*** MASK-OFF NOT APPLIED -- patch did not take ***"
[ "$n" -gt 0 ] && echo "*** 'Found 0 templates' x$n -- UNTEMPLATED ***"
echo "### poses: $(find $A/sweep_out/$RUN -name '*_sample_*.cif' 2>/dev/null | wc -l) (expect 25)"
echo "### DONE $(date +%H:%M)"
