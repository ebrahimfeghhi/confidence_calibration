#!/bin/bash
# Wait for the running scorer to exit, drop any 7z2m entries it wrote back from its stale in-memory
# copy, then re-score them under the protocol (full assembly + OST global chain mapping).
set -u
EXP=/home/ebrahim/confident_decoy_experiments; PY=/home/ebrahim/Protenix/.venv/bin/python
cd $EXP
while pgrep -f "scripts/score_sweep.py" >/dev/null; do sleep 60; done
$PY -c "
import json
p='artifacts/sweep_scores.json'; d=json.load(open(p)); n=0
for r in d:
    for k in [k for k in d[r] if k.startswith('7z2m')]: del d[r][k]; n+=1
json.dump(d, open(p,'w'), indent=1); print(f'dropped {n} stale 7z2m entries')"
rm -f $EXP/artifacts/sweep_ost/*_7z2m_*.json
echo "=== rescoring 7z2m under protocol ==="
$PY scripts/score_sweep.py > $EXP/artifacts/decoy_logs/score_sweep_7z2m.log 2>&1
grep 7z2m $EXP/artifacts/decoy_logs/score_sweep_7z2m.log
