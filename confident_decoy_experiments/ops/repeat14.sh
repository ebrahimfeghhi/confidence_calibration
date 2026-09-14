#!/usr/bin/env bash
# Fold 9x1h seed 300014 REPEATEDLY, same config, to settle why it did not reproduce the sweep.
#
# The sweep ran all 50 seeds in ONE process and got poses spanning 0.044-0.634 for this seed; our
# standalone fold got 0.037-0.192. Two explanations: (a) CUDA/bf16 nondeterminism, in which case
# repeats should scatter and some may recover the good pose; (b) a different RNG history in a
# single-seed process, in which case every repeat is IDENTICAL and the sweep's per-seed poses are
# simply not reproducible standalone.
#
# Each repeat writes to its own directory via TAG and keeps its trunk dump, so any repeat that does
# produce a straddling set can go straight to the arms.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
GPU=${GPU:?}; REPS=${REPS:?}
for r in $REPS; do
  echo "=== repeat $r (GPU$GPU)"
  GPU=$GPU PDB=9x1h IFACE=9x1h_EF SEED=300014 TAG="r$r" FOLD_ONLY=1 \
    bash ops/commit11.sh 2>&1 | grep -E "DONOR POSES|span |trunk "
done
