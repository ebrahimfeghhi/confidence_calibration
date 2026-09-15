#!/usr/bin/env bash
# Run the full fold+commit unit over many interfaces on ONE GPU, via the single-process driver.
#
# Per interface: NSEEDS seeds, each seed = fold (10 recycles, trunk dumped) + control + p0..p4, then the
# trunk is discarded. Rows go to the PROJECT results.csv, keyed by iface+seed+arm.
#
# WHY fast_driver.py AND NOT batch.sh. The shell driver spawns a fresh runner/inference.py per stage,
# and MEASURED on 8fas: 48s of checkpoint load wrapped around 14s of compute, seven times per seed --
# 390s/seed, ~77% overhead. The single-process driver loads the 464M checkpoint ONCE per complex and
# reuses it: 104s/seed (fold 29s, control 11s, each p-arm 9s). Over the 1000 seeds of this sweep that is
# ~14h instead of ~54h on two GPUs, i.e. ~40 GPU-hours of loading the same file 7000 times.
#
# It is validated on mechanism, NOT on numeric agreement -- the model is not run-to-run deterministic on
# this box (the same 9x1h seed spanned DockQ 0.004-0.651 across nine identical repeats), so "same
# answers" is not a checkable claim. What was checked: all 7 stages produce 5 poses, each arm writes to
# its own directory (arms overwriting each other was the known trap -- verified no two arms share
# byte-identical rows), the trunk is discarded, and rows land in the project csv.
#
# DRIVER=shell falls back to batch.sh if the fast driver is ever in doubt.
#
# Usage: GPU=5 IFACES='9x1h_CD 9i20_AB' NSEEDS=50 bash pxm_sweep.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; IFACES=${IFACES:?}
NSEEDS=${NSEEDS:-50}
DRIVER=${DRIVER:-fast}
DATASET=${DATASET:-pxm2025}
OUTROOT=${OUTROOT:-$REPO/artifacts/commit11/fast}
REPORT=${REPORT:-$REPO/artifacts/pxm_reports.txt}
mkdir -p "$OUTROOT"

echo "=== $(date +%H:%M:%S) GPU$GPU pxm sweep [$DRIVER]: $(echo "$IFACES" | wc -w) interfaces x $NSEEDS seeds"
for iface in $IFACES; do
  pdb=${iface%%_*}
  echo "=== $(date +%H:%M:%S) GPU$GPU >>> $iface"
  if [ "$DRIVER" = "fast" ]; then
    CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      $PY "$DC/fast_driver.py" --pdb "$pdb" --iface "$iface" --seeds "1-$NSEEDS" \
        --out "$OUTROOT/$iface" --dataset "$DATASET" \
        || echo "!! $iface exited nonzero, moving on"
  else
    GPU=$GPU PDB=$pdb IFACE=$iface DATASET=$DATASET NS=5 KEEP_TRUNK=0 ZINIT=0 \
      SEEDS="$(seq 1 "$NSEEDS" | tr '\n' ' ')" bash "$DC/batch.sh" \
        || echo "!! $iface exited nonzero, moving on"
  fi
  # Report the moment this interface is done, rather than at the end of a multi-day sweep.
  { echo; echo "######## $iface finished $(date '+%m-%d %H:%M:%S') on GPU$GPU [$DRIVER] ########"
    $PY "$DC/report_complex.py" --iface "$iface"; } >> "$REPORT" 2>&1
  echo "    -> summary appended to $REPORT"
  # Structures are large and the scores are already in the csv; keep only the donor poses, which are
  # the templates and the x-axis. Arm cifs are pruned by the driver itself.
  du -sh "$OUTROOT/$iface" 2>/dev/null | sed 's/^/    kept: /'
done
echo "=== $(date +%H:%M:%S) GPU$GPU pxm sweep COMPLETE"
