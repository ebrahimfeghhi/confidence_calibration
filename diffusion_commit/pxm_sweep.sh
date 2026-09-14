#!/usr/bin/env bash
# Run the full fold+commit unit over many PXM-2025 confident-decoy interfaces on ONE GPU.
#
# Per interface: NSEEDS seeds, each seed = fold (10 recycles, trunk dumped) + control + p0..p4, then the
# trunk is DISCARDED. The trunk is only needed while that seed's own arms run, so keeping it across a
# sweep this size would cost ~100 GB at 462 tokens and several times that at 1474 -- on a quota this
# project has already filled once and crashed every job with.
#
# Scored rows land in artifacts/commit11/results.csv, keyed by iface+seed+arm, so nothing here has to be
# re-derived from the structures afterwards.
#
# Usage: GPU=5 IFACES='9x1h_CD 9i20_AB' NSEEDS=50 bash pxm_sweep.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU=${GPU:?}; IFACES=${IFACES:?}
NSEEDS=${NSEEDS:-50}
SEEDS=$(seq 1 "$NSEEDS" | tr '\n' ' ')

echo "=== $(date +%H:%M:%S) GPU$GPU pxm sweep: $(echo "$IFACES" | wc -w) interfaces x $NSEEDS seeds"
for iface in $IFACES; do
  pdb=${iface%%_*}
  echo "=== $(date +%H:%M:%S) GPU$GPU >>> $iface"
  # DATASET=pxm2025 is the default, but state it: it selects the input json, the scoring reference and
  # PXMeter as the scorer together, and mixing pools silently would make the numbers incomparable.
  GPU=$GPU PDB=$pdb IFACE=$iface DATASET=pxm2025 NS=5 KEEP_TRUNK=0 ZINIT=0 \
    SEEDS="$SEEDS" bash "$DC/batch.sh" || echo "!! $iface exited nonzero, moving on"
  # Report the moment this interface is done, rather than at the end of a multi-day sweep.
  REPORT=${REPORT:-/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/pxm_reports.txt}
  { echo; echo "######## $iface finished $(date '+%m-%d %H:%M:%S') on GPU$GPU ########"
    /home/ebrahim/Protenix/.venv/bin/python "$DC/report_complex.py" --iface "$iface"; } >> "$REPORT" 2>&1
  echo "    -> summary appended to $REPORT"
done
echo "=== $(date +%H:%M:%S) GPU$GPU pxm sweep COMPLETE"
