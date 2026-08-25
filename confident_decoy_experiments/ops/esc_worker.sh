#!/bin/bash
# One worker per GPU. Repeatedly claims the next unfinished esc_* arm and runs it.
# Claiming uses mkdir (atomic) so two workers never take the same arm -- which matters because the
# grabber can add workers at any time.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; LOCK=$A/esc_locks; mkdir -p "$LOCK"
GPU=$1
say(){ echo "[$(date +%H:%M:%S)] gpu$GPU $*"; }
say "worker started"
while true; do
    claimed=""
    # e52_* (sigma 5e-2) arms only become eligible once the chain script creates the flag, so the
    # 1e-2 sweep finishes first instead of the two interleaving.
    pool="$A/esc_*_jobs.json"
    [ -f "$A/.e52_enabled" ] && pool="$pool $A/e52_*_jobs.json"
    [ -f "$A/.e25_enabled" ] && pool="$pool $A/e25_*_jobs.json"
    # sigma=1e-1, last tier: only after 5e-2 is drained. Order matters -- the loop takes the FIRST
    # incomplete arm it finds, so listing e1_ last keeps 1e-2 > 5e-2 > 1e-1 priority.
    [ -f "$A/.e1_enabled" ] && pool="$pool $A/e1_*_jobs.json"
    for f in $pool; do
        r=$(basename "$f" _jobs.json)
        want=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$f')))*5)")
        have=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$have" -ge "$want" ] && continue
        mkdir "$LOCK/$r" 2>/dev/null || continue        # someone else has it
        claimed=$r; break
    done
    if [ -z "$claimed" ]; then say "no unclaimed work left -- exiting"; exit 0; fi
    say "claimed $claimed"
    bash $REPO/ops/run_wnoise_pc.sh $GPU $claimed
    rmdir "$LOCK/$claimed" 2>/dev/null || true
done
