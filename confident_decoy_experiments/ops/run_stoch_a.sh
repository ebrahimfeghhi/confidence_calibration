#!/bin/bash
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python; A=$REPO/artifacts
run() { echo "### START $1 [$2] $(date +%H:%M)"
  $PY $REPO/scripts/sweep_sched.py --jobs $A/$1_jobs.json --out $A/sweep_out/$1 --logs $A/$1_logs \
      --status $A/$1_status.json --gpus $3 --n-sample 5 --extra-args "$2"
  m=$(grep -ho "N_msa [0-9]*" $A/$1_logs/*.log 2>/dev/null|sort -u|tr '\n' ' ')
  echo "### DONE $1 poses=$(find $A/sweep_out/$1 -name '*_sample_*.cif' 2>/dev/null|wc -l) [N_msa: $m] $(date +%H:%M)"; }
run "$1" "$2" "$3"
[ -n "${4:-}" ] && run "$4" "$5" "$3"
echo "### SESSION DONE $(date +%H:%M)"
