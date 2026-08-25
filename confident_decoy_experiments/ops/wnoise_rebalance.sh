#!/bin/bash
# Hand one of GPU 3's queued conditions to another card once that card's lane is done.
# Safe against a race: run_wnoise.sh skips any condition already at 25 poses, so whichever lane gets
# there second just skips it. Worst case is a few minutes of duplicated compute, never corruption.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
GPU=$1; WAITFOR=$2; TAKE=$3
echo "### gpu$GPU: waiting for $WAITFOR to finish before taking $TAKE"
while [ "$(find $A/sweep_out/$WAITFOR -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 25 ]; do sleep 20; done
n=$(find $A/sweep_out/$TAKE -name '*_sample_*.cif' 2>/dev/null | wc -l)
if [ "$n" -ge 25 ]; then echo "### $TAKE already complete -- nothing to do"; exit 0; fi
echo "### gpu$GPU taking $TAKE (currently $n/25) $(date +%H:%M)"
bash $REPO/ops/run_wnoise.sh $GPU $TAKE
