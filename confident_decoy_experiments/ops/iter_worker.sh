#!/bin/bash
# One GPU, one complex at a time, 50 iterations each. Each GPU owns a 50-id template block and
# RECYCLES it per complex -- ids are a scarce global resource (~555 left, no free first-char) and a
# finished complex's templates are no longer referenced.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=$1; START=$2
while read -r pdb; do
    [ -z "$pdb" ] && continue
    n=$(find $REPO/artifacts/sweep_out/iter_${pdb} -name '*_sample_*.cif' 2>/dev/null | wc -l)
    if [ "$n" -ge 50 ]; then echo "### SKIP $pdb ($n/50)"; continue; fi
    echo "### START $pdb on gpu$GPU tids 8$(printf %03d $START)+ $(date +%H:%M)"
    $PY $REPO/scripts/iter_selftmpl.py --pdb "$pdb" --gpu "$GPU" --iters 50 \
        --tid-prefix 8 --tid-start "$START" --run iter
    echo "### DONE $pdb poses=$(find $REPO/artifacts/sweep_out/iter_${pdb} -name '*_sample_*.cif' 2>/dev/null | wc -l)"
done < "$REPO/artifacts/iter_gpu${GPU}.txt"
echo "### ALL COMPLEXES DONE on gpu$GPU"
