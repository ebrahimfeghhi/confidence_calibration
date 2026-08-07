#!/bin/bash
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
while :; do
  b=$($PY -c "import json,os;p='$EXP/artifacts/minmaxdonor_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  $PY $EXP/scripts/score_arm.py --arm minmaxdonor --jobs $EXP/artifacts/minmax_arm_jobs.json \
      --out $EXP/artifacts/sweep_out/minmaxdonor --logs $EXP/artifacts/minmax_logs --workers 10 >/dev/null 2>&1
  a=$($PY -c "import json,os;p='$EXP/artifacts/minmaxdonor_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  if [ "$a" -gt "$b" ]; then
    $PY - <<'PYEOF'
import json
E='/home/ebrahim/confident_decoy_experiments'
m=json.load(open(f'{E}/artifacts/minmaxdonor_scores.json'))
dq=json.load(open(f'{E}/artifacts/dockqdonor_scores.json'))
pk=json.load(open(f'{E}/artifacts/minmax_donor_picks.json'))
print()
for k in sorted(m):
    v=m[k]; d=dq.get(k,{})
    print(f"  {k:11} worst {pk[k]['worst_side']:5.2f}A  {v['sel_dockq']:6.3f}"
          f"{'*' if v['rescued'] else ' '} (n_ok {v['n_acceptable']:3d}) | dq {d.get('sel_dockq','-'):>6}"
          f"{'*' if d.get('rescued') else ' '} (n_ok {d.get('n_acceptable','-')})")
print(f"  minmaxdonor RESCUED {sum(1 for v in m.values() if v['rescued'])}/{len(m)} of the newly run")
PYEOF
  fi
  s=0
  for pr in /proc/[0-9]*; do c=$(tr '\0' ' ' < "$pr/cmdline" 2>/dev/null)||continue
    case "$c" in *sweep_sched.py*minmax_arm_jobs*) s=1;break;; esac; done
  [ "$s" -eq 0 ] && [ "$a" -ge 14 ] && { echo "[$(date +%H:%M)] all 14 scored -- exiting"; break; }
  sleep 600
done
