#!/usr/bin/env bash
# Keep artifacts/pxm2025/epitope_overlap_pxm.csv caught up with the PXM-2025 sweep.
#
# epitope_overlap_pxm.py is resumable (it skips poses already in the csv) and enumerates work from
# pxm_json/, which ops/pxm2025_scorewatch.sh fills as PXMeter scores each pose. So re-running it on a
# loop simply picks up whatever landed since last time. 24 workers: the scoring watcher also uses 24,
# and our CPU budget on this box is 64 cores total.
#
# Exits once the generation sweep is done AND a pass finds nothing new, so it does not idle forever.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/miniconda3/envs/foldbench/bin/python
LOG=$REPO/artifacts/pxm2025/epiwatch.log
JOBS=${JOBS:-24}

cd "$REPO" || exit 1
while :; do
  echo "=== $(date +%F\ %H:%M) pass start"
  $PY scripts/epitope_overlap_pxm.py --jobs "$JOBS" 2>&1 | tail -25
  n=$(($(wc -l < artifacts/pxm2025/epitope_overlap_pxm.csv) - 1))
  echo "=== $(date +%F\ %H:%M) pass done, $n poses total"

  # stop only when generation has finished everywhere and there is nothing left to pick up
  if grep -qs "ALL DONE" artifacts/pxm2025/out/*.log ops/pxm*.log 2>/dev/null; then
    left=$(ls artifacts/pxm2025/pxm_json | wc -l)
    [ "$n" -ge "$left" ] && { echo "=== caught up and sweep finished, exiting"; break; }
  fi
  sleep 1800
done
