#!/usr/bin/env bash
# Score PXM-2025 complexes with PXMeter's own metrics as soon as each one finishes generating.
#
# The scorer is resumable (a pose whose pxm JSON exists is skipped) and only picks up complexes with
# all 50 seeds present, so re-running it on a loop costs nothing and never scores a partial complex.
# 24 workers, each pinned to one thread -> ~24 cores, well inside the 64-core budget and leaving the
# two GPU workers unstarved.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=$REPO/envs/pxmeter/bin/python
cd "$REPO" || exit 1
while true; do
  n=$(ls artifacts/pxm2025/out/.claims 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] claimed $n/131; scoring any newly complete complexes"
  $PY scripts/score_pxm2025.py --jobs 24 2>&1 | tail -3
  # stop once every complex is claimed AND every claimed one is fully scored
  done_n=$(awk -F, 'NR>1{print $1"_"$2$3}' artifacts/pxm2025/dockq_pxm2025.csv 2>/dev/null | sort -u | wc -l)
  echo "[$(date +%H:%M)] interfaces scored so far: $done_n / 314"
  [ "$done_n" -ge 314 ] && { echo "ALL SCORED"; break; }
  sleep 900
done
