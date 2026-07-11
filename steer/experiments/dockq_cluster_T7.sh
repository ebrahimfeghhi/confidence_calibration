#!/usr/bin/env bash
# Run dockq_cluster.py on the remaining 6 of the T7 set, SEQUENTIALLY (one job at a
# time = ~105 GB peak; running them in parallel would be ~630 GB and rude on the shared box).
# Each run appends to results/csv_files/dockq_cluster_summary.csv and has its own memory watchdog.
set -u
cd /home/ebrahim/tts_abag
PY=.venv/bin/python
export NPROC="${NPROC:-64}"

# T7 minus the already-done 8ts0 A-C
TARGETS=("8ts0 B C" "8a44 A D" "7yxu A C" "8a44 A C" "7zoz A C" "8vui B C")

for t in "${TARGETS[@]}"; do
  set -- $t
  tag="${1}_${2}${3}"
  echo "======== $(date '+%H:%M:%S')  START $1 $2-$3 ========"
  NPROC=$NPROC $PY steer/experiments/dockq_cluster.py "$1" "$2" "$3" \
      > "steer/experiments/dockq_cluster_${tag}.out" 2>&1
  rc=$?
  echo "======== $(date '+%H:%M:%S')  DONE  $1 $2-$3  (exit $rc) ========"
  if [ $rc -eq 3 ]; then
    echo "!!! memory watchdog killed a run (exit 3) — stopping the whole sweep to be safe."
    break
  fi
done
echo "======== T7 sweep complete ========"
