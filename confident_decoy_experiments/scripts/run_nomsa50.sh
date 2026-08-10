#!/bin/bash
# Wait for the in-flight 10-seed no-MSA pass to finish, then run the full 50-seed sweep over all 25
# interfaces. The 10-seed outputs are REUSED: nomsa_donors.py skips seeds that already have output,
# so those 7 runs only produce their missing 40 seeds. Shortest-first ordering.
set -u
while pgrep -f "nomsa_donors.py --subset rescued" >/dev/null; do sleep 60; done
sleep 15
/home/ebrahim/Protenix/.venv/bin/python \
  /home/ebrahim/tts_abag/confident_decoy_experiments/scripts/nomsa_donors.py \
  --subset all --n-seeds 50 --gpu 7
