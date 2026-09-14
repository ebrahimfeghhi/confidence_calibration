#!/usr/bin/env bash
# Analyse AND plot each complex as soon as its 8 seeds finish.
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
while true; do
  /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/traj_analyze.py 2>&1 \
    | grep -vE "^\s*$" | tee -a ops/traj_analyze.log
  /home/ebrahim/Protenix/.venv/bin/python scripts/plot_traj.py >/dev/null 2>&1
  n=$(/home/ebrahim/Protenix/.venv/bin/python -c "
import json,glob
d=json.load(open('artifacts/traj20_seeds.json'))
print(sum(1 for p in d if len(glob.glob(f'artifacts/traj_cycles/{p}/*_coords.npz'))>=80))")
  echo "[$(date +%H:%M)] complexes done: $n/16" | tee -a ops/traj_analyze.log
  [ "$n" -ge 16 ] && { echo "ALL DONE" >> ops/traj_analyze.log; break; }
  sleep 240
done
