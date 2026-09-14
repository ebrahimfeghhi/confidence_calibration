#!/bin/bash
# Stage 2: 50 seeds at STOCK settings (MSA subsampling on, dropout on, no weight noise), templated
# ONLY on that complex's 5 MSA-free stage-1 poses. multichain_mask left ON -> intra-chain fold only.
# PROTENIX_MAX_TEMPLATES=1 so all five are used; the default cap of 4 would silently drop one.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts; GPU=$1; shift
for run in "$@"; do
    n=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 250 ] && { echo "### SKIP $run ($n/250)"; continue; }
    echo "### START $run selftmpl=1(per-donor arm) mask=ON stock GPU$GPU $(date +%H:%M)"
    PROTENIX_MAX_TEMPLATES=1 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/${run}_jobs.json --out $A/sweep_out/$run \
        --logs $A/${run}_logs --status $A/${run}_status.json --gpus $GPU --n-sample 5
    # SILENT-FAILURE GUARD: a bad template id makes Protenix log "Found 0 templates" and run
    # untemplated while still emitting plausible poses. Three arms were scored and believed for a
    # week that way. Assert the real count here rather than trusting the build.
    f=$(grep -ho "Found [0-9]* templates" $A/${run}_logs/*.log 2>/dev/null | sort -u | tr '\n' ' ')
    echo "### DONE $run poses=$(find $A/sweep_out/$run -name '*_sample_*.cif' 2>/dev/null|wc -l) | templates: ${f:-NONE}"
    # ONE template per arm now, so "Found 1 templates" is correct. "Found 0" means the id was
    # rejected and the run is silently untemplated -- that is the failure to catch.
    case "$f" in
        *"Found 1 templates"*) ;;
        *) echo "*** $run: expected 'Found 1 templates', got '${f:-NONE}' ***";;
    esac
done
