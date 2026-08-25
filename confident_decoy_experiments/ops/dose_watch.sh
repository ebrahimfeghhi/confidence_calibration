#!/bin/bash
# Score the dose sweep, then decide whether to extend the ladder. Every 30 min.
# Scoring first is required: the extender reads the curve this pass writes.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
while true; do
    say "scoring pass"
    $PY $REPO/scripts/score_dose.py 2>&1 | tee $REPO/ops/dose_score_latest.txt | tail -20
    say "extension check"
    $PY $REPO/scripts/dose_extend.py 2>&1
    say "sleeping 30 min"
    sleep 1800
done
