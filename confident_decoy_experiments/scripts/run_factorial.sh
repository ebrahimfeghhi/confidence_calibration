#!/bin/bash
# Wait for the in-flight unmask8che run to exit, then run the 6-cell factorial on GPU 7.
set -u
while pgrep -f "input_json_path.*8che_allpose_137_3.json --dump_dir.*unmask8che " >/dev/null; do sleep 60; done
sleep 10
/home/ebrahim/Protenix/.venv/bin/python /home/ebrahim/tts_abag/confident_decoy_experiments/scripts/unmask_factorial.py --gpu 7 --cells A,B,C,D,E,F
