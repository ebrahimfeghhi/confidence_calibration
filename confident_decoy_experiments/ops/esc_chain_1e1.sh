#!/bin/bash
# Enables the sigma=1e-1 tier once the 5e-2 tier (e52_*, including any escalations) is drained.
# Mirrors esc_chain_5e2.sh. Does NOT wait on the 1e-2 tier -- worker priority ordering already
# guarantees 1e-2 arms are claimed before 5e-2, and 5e-2 before 1e-1, so all three can be flagged
# live without them competing for GPUs.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/esc_chain_1e1.log"; }

# Don't even start counting until 5e-2 is live, otherwise "0 arms incomplete" is trivially true
# before any e52 work has been claimed and we would enable 1e-1 immediately.
while [ ! -f "$A/.e52_enabled" ]; do sleep 300; done
say "5e-2 is live; waiting for it to drain before enabling 1e-1"

while true; do
    left=0
    for f in $A/e52_*_jobs.json; do
        r=$(basename "$f" _jobs.json)
        w=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        h=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$h" -lt "$w" ] && left=$((left+1))
    done
    ext=$(tail -20 "$REPO/ops/esc_driver.log" 2>/dev/null | grep -c "EXTEND" || true)
    if [ "$left" -eq 0 ] && [ "$ext" -eq 0 ]; then break; fi
    say "5e-2 still running ($left arms incomplete, $ext recent extensions)"
    sleep 600
done
say "5e-2 complete -- enabling 1e-1 arms"
touch $A/.e1_enabled
say "1e-1 enabled; workers will pick up e1_* arms"
