#!/usr/bin/env bash
# Take every confident-decoy interface from 50 free seeds to 75, folding ONCE PER PDB.
#
# WHY PER PDB. 89 interfaces short of 75 seeds live on only 35 distinct structures -- 9bao alone carries
# 8 of them. A fold produces all of a structure's interfaces at once, so folding per interface would
# repeat the same 25 folds up to 8 times. 875 folds, not 2225.
#
# Interfaces beyond the first on a PDB are scored afterwards from the saved cifs (cheap, CPU only);
# this script scores one interface inline so progress is visible in the csv as it goes.
#
# Uses fast_driver --mode free, so the checkpoint is loaded once per PDB rather than once per fold.
# Claims are per PDB via atomic mkdir, so both GPUs can share this list without overlapping.
#
# Usage: GPU=5 bash free75_fill.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
TODO=${TODO:-$REPO/artifacts/free75_todo.tsv}
OUT=${OUT:-$REPO/artifacts/free75fill}
CSV=${CSV:-$REPO/artifacts/commit11/results.csv}
CLAIM=$OUT/.claims
SEEDS=${SEEDS:-300050-300074}
STOP=$REPO/artifacts/FREE75FILL_STOP
mkdir -p "$CLAIM"; cd "$REPO" || exit 1

# one line per PDB, carrying the first interface for inline scoring
awk -F'\t' '!seen[$2]++ {print $2"\t"$1}' "$TODO" | while IFS=$'\t' read -r PDB IFACE; do
  [ -f "$STOP" ] && { echo "[$(date +%H:%M:%S)] stop file seen"; exit 0; }
  mkdir "$CLAIM/$PDB" 2>/dev/null || continue
  echo "[$(date '+%m-%d %H:%M:%S')] gpu$GPU >>> $PDB ($IFACE) seeds $SEEDS"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  $PY "$DC/fast_driver.py" --mode free --arm-name free75 \
      --pdb "$PDB" --iface "$IFACE" --seeds "$SEEDS" \
      --csv "$CSV" --timing-csv "$REPO/artifacts/stage_timings.csv" \
      --out "$OUT/$PDB" 2>&1 | grep -vE "INFO|WARNING|^\s*$" | tail -3
done
echo "[$(date '+%m-%d %H:%M:%S')] gpu$GPU free75 fill done"
