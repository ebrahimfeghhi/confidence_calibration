#!/usr/bin/env bash
# Add seed 101 to the mt8vyl_good arm on GPU0, then hand GPU0 back to PXM.
#
# The arm evaluates seeds 102-150 MINUS 125 (125 is the donor, so it cannot also be a test seed).
# That is 48 seeds; seed 101 is a legitimate extra here because the donor is 125, not 101 -- adding it
# restores the 49-seed count and makes this arm exactly comparable to the bad-donor arm.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
J=$REPO/artifacts/selftmpl_tmpl/mt8vyl_good/8vyl_mt8vyl_good.json
while tmux has-session -t mtgood0 2>/dev/null; do sleep 15; done
echo "$(date +%H:%M:%S) mtgood0 done -> running seed 101 on GPU0"
RUN=mt8vyl_good PDB=8vyl GPU=0 NCYCLE=10 NTMPL=5 SEEDS=101 JSON=$J bash "$REPO/ops/run_multitmpl.sh"
echo "$(date +%H:%M:%S) seed 101 done -> restarting pxm0"
tmux new-session -d -s pxm0 -c "$REPO" 'GPU=0 bash ops/run_pxm2025.sh'
sleep 4
tmux has-session -t pxm0 2>/dev/null && echo "  pxm0 up" || echo "  FAILED pxm0"
