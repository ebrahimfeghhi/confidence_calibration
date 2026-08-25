#!/bin/bash
# Starts the sigma=5e-2 sweep once the 1e-2 sweep (esc_*, including escalations) is fully done.
# Workers pick up e52_* arms automatically -- esc_worker.sh globs esc_*_jobs.json, so this widens
# that glob to include e52_ by symlinking? No: instead we flip a flag file the workers already read.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/esc_chain_5e2.log"; }
say "waiting for the 1e-2 sweep to finish before starting 5e-2"
while true; do
    left=0
    for f in $A/esc_*_jobs.json; do
        r=$(basename "$f" _jobs.json)
        w=$(/home/ebrahim/Protenix/.venv/bin/python -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        h=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$h" -lt "$w" ] && left=$((left+1))
    done
    # also wait for the escalation driver to stop extending
    ext=$(tail -20 "$REPO/ops/esc_driver.log" 2>/dev/null | grep -c "EXTEND" || true)
    if [ "$left" -eq 0 ] && [ "$ext" -eq 0 ]; then break; fi
    say "1e-2 still running ($left arms incomplete, $ext recent extensions)"
    sleep 600
done
say "1e-2 complete -- enabling 5e-2 arms"
touch $A/.e52_enabled
say "5e-2 enabled; workers will pick up e52_* arms"
