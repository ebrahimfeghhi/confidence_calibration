#!/usr/bin/env bash
# Fold several seeds of one complex and report which produce a STRADDLING set of poses.
#
# Target seeds must be picked from our own folds: the original sweep ran all 50 seeds in a single
# process, so a standalone re-fold of the same seed draws a different MSA and yields different poses.
# Selecting from sweep tables sent us to a 9x1h seed that spanned 0.591 there and 0.155 here.
# Each fold caches its trunk dump, so whichever seed we choose runs its arms without refolding.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
GPU=${GPU:?}; PDB=${PDB:?}; IFACE=${IFACE:?}; SEEDS=${SEEDS:-"300000 300001 300002 300003"}
for s in $SEEDS; do
  echo "=== seed $s"
  GPU=$GPU PDB=$PDB IFACE=$IFACE SEED=$s FOLD_ONLY=1 bash ops/commit11.sh 2>&1 | grep -E "DONOR POSES|span|trunk |donor:"
done
