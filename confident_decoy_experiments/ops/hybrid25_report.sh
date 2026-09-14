#!/usr/bin/env bash
# Every 10 min: re-extract ALL ab-ag interfaces from the PXMeter JSONs, then compare Arm A vs Arm B.
# Re-extraction is a pure re-read of files the workers already wrote -- no scoring, no GPU.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
while true; do
  /home/ebrahim/Protenix/.venv/bin/python scripts/hybrid25_all_interfaces.py \
     > artifacts/hybrid25/extract.log 2>&1
  {
    echo "=== $(date '+%F %H:%M:%S')"
    echo "units $(ls artifacts/hybrid25/.done 2>/dev/null | wc -l)/3075   workers $(tmux ls 2>/dev/null | grep -cE '^hyw[0-9]:')/3"
    tail -4 artifacts/hybrid25/extract.log
    /home/ebrahim/Protenix/.venv/bin/python scripts/read_hybrid25.py
  } > artifacts/hybrid25/report.txt 2>&1
  sleep 600
done
