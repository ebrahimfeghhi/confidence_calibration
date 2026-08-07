#!/bin/bash
# Score paratopedonor interfaces as each one finishes (CPU/OST only, no GPU contention).
# One template per interface, so an interface is scorable the moment its 50 seeds are done --
# unlike leader20, whose answer is pooled across 20 templates.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
while :; do
  before=$($PY -c "import json,os;p='$EXP/artifacts/paratopedonor_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  $PY $EXP/scripts/score_arm.py --arm paratopedonor \
      --jobs $EXP/artifacts/paratope_arm_jobs.json \
      --out $EXP/artifacts/sweep_out/paratopedonor \
      --logs $EXP/artifacts/paratope_logs --workers 10 >/dev/null 2>&1
  after=$($PY -c "import json,os;p='$EXP/artifacts/paratopedonor_scores.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  if [ "$after" -gt "$before" ]; then
    $PY - <<'PYEOF'
import json, os
EXP='/home/ebrahim/confident_decoy_experiments'
p=json.load(open(f'{EXP}/artifacts/paratopedonor_scores.json'))
dq=json.load(open(f'{EXP}/artifacts/dockqdonor_scores.json'))
picks=json.load(open(f'{EXP}/artifacts/paratope_donor_picks.json'))
print(f"\n{'interface':11} {'para RMSD':>9} {'paratopedonor':>21} | {'dockqdonor':>18}")
n=r=0
for k in sorted(p):
    v=p[k]; d=dq.get(k,{})
    n+=1; r+=v['rescued']
    print(f"  {k:11} {picks[k]['paratope_rmsd']:7.2f}A  {v['sel_dockq']:6.3f}"
          f"{'*' if v['rescued'] else ' '} (n_ok {v['n_acceptable']:3d}) | "
          f"{d.get('sel_dockq','-'):>6}{'*' if d.get('rescued') else ' '} (n_ok {d.get('n_acceptable','-')})")
print(f"  paratopedonor RESCUED {r}/{n} scored"
      f"   (dockqdonor on the same {n}: {sum(1 for k in p if dq.get(k,{}).get('rescued'))})")
PYEOF
  fi
  sched=0
  for pr in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$pr/cmdline" 2>/dev/null) || continue
    case "$c" in *sweep_sched.py*paratope_arm_jobs*) sched=1; break;; esac
  done
  [ "$sched" -eq 0 ] && [ "$after" -ge 22 ] && { echo "[$(date +%H:%M)] all 22 scored -- watcher exiting"; break; }
  sleep 600
done
