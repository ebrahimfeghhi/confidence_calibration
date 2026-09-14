#!/usr/bin/env bash
# Run commit11 (fold + control + 5 pose-arms) over a list of seeds on one GPU.
#
# Uses the SHELL version, which is verified: every arm asserts that LOAD_TRUNK fired, that exactly
# one recycle ran, and that the expected number of templates was found. The single-process driver is
# faster but still being debugged; correctness first.
#
# Each seed keeps its trunk dump (z_trunk + s_trunk) so arms can be re-run or extended without
# refolding. At 1470 tokens that is ~1.1 GB per seed.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
GPU=${GPU:?}; PDB=${PDB:?}; IFACE=${IFACE:?}; SEEDS=${SEEDS:?}
for s in $SEEDS; do
  if [ -d "$REPO/artifacts/commit11/${IFACE}_s${s}/p4" ]; then
    echo "=== seed $s already complete, skipping"; continue
  fi
  echo "=== $(date +%H:%M:%S) GPU$GPU seed $s"
  GPU=$GPU PDB=$PDB IFACE=$IFACE SEED=$s bash ops/commit11.sh
done
echo "=== $(date +%H:%M:%S) GPU$GPU: all seeds done"
