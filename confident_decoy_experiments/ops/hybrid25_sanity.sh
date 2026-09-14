#!/usr/bin/env bash
# Every 30 minutes, check the SCORES look sane -- not just that the run is alive.
#
# WHY. A run can look perfectly healthy while producing garbage. Tonight a stray carriage return from
# csv.writer's \r\n default split every results row in two, and it was only caught because a reader
# script choked on it. Earlier today a neighbour's job caused silent CUDA OOMs and 36 units were
# marked complete with no data. Both were invisible to "are the workers up".
#
# WHAT IS CHECKED, each an independent way for the run to be quietly wrong:
#   1. csv integrity   -- every row has 8 fields, no stray CR
#   2. value ranges    -- DockQ in [0,1], ipTM in [0,1], no NaN floods
#   3. rows per unit   -- exactly 5 (one per diffusion sample); anything else means a partial write
#   4. templating fired-- "Found 5 templates" in the run logs; an untemplated run still emits
#                         plausible poses, so this is the failure mode that hides best
#   5. solved controls -- the 15 already-solved interfaces should stay solved; if templating is
#                         breaking them, something is wrong with the chain assignment
#   6. progress        -- units completed since the last check; zero means stalled
# Anything that fails is written to the top of the report as ALERT lines.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
M=$REPO/artifacts/hybrid25
OUT=$M/sanity.log
cd "$REPO" || exit 1
last=0
while true; do
  {
    echo "=== $(date '+%F %H:%M:%S') sanity check"
    /home/ebrahim/Protenix/.venv/bin/python - "$last" <<'EOF'
import csv, collections, glob, os, sys
import numpy as np
M="artifacts/hybrid25"; last=int(sys.argv[1]); alerts=[]
raw=open(f"{M}/results.csv").read()
if "\r" in raw: alerts.append("stray carriage return in results.csv")
rows=list(csv.reader(raw.splitlines()))
bad=[i for i,r in enumerate(rows) if len(r)!=8]
if bad: alerts.append(f"{len(bad)} malformed rows (first at line {bad[0]+1})")
d=csv.DictReader(raw.splitlines())
dq,ip,byunit=[],[],collections.Counter()
for r in d:
    try: a,b=float(r["dockq"]),float(r["iptm"])
    except (TypeError,ValueError): continue
    dq.append(a); ip.append(b); byunit[(r["pdb"],r["donor_seed"])]+=1
dq,ip=np.array(dq),np.array(ip)
n=len(dq)
if n:
    if ((dq<0)|(dq>1)).any(): alerts.append("DockQ outside [0,1]")
    if np.nanmax(ip)>1.001 or np.nanmin(ip)<-0.001: alerts.append("ipTM outside [0,1]")
    nan=np.isnan(ip).mean()
    if nan>0.5: alerts.append(f"{100*nan:.0f}% of ipTM values are NaN")
    wrong=[u for u,c in byunit.items() if c!=5]
    if wrong: alerts.append(f"{len(wrong)} units without exactly 5 rows (e.g. {wrong[0]})")
# templating actually fired
logs=glob.glob(f"{M}/logs/gpu*.log")
txt="".join(open(f).read() for f in logs) if logs else ""
few=txt.count("FEWER THAN 5 TEMPLATES")
if few: alerts.append(f"{few} units templated with <5 templates (cumulative)")
# count units that are CURRENTLY broken, not every occurrence ever logged. The retry guard fixes
# these automatically, so a cumulative grep keeps alerting on resolved failures -- and worse, hides
# a new one inside an already non-zero count.
import re as _re
stuck=[]
for m in _re.finditer(r"(\S+) produced no rows", txt):
    u=m.group(1)
    if not os.path.exists(f"{M}/.done/{u}"): stuck.append(u)
if stuck: alerts.append(f"{len(set(stuck))} units still with no rows after retry: {sorted(set(stuck))[:3]}")
# the solved controls must stay solved
sol=collections.defaultdict(list)
for r in csv.DictReader(raw.splitlines()):
    if r["cls"]=="solved":
        try: sol[r["interface"]].append(float(r["dockq"]))
        except (TypeError,ValueError): pass
broke=[k for k,v in sol.items() if len(v)>=25 and max(v)<0.23]
if broke: alerts.append(f"solved controls collapsed: {broke}")
done=len(os.listdir(f"{M}/.done")) if os.path.isdir(f"{M}/.done") else 0
if last and done==last: alerts.append(f"no progress since last check (still {done} units)")
print(f"units {done}/1500   rows {n}   since last check: +{done-last}")
if n:
    print(f"DockQ mean {dq.mean():.3f} range {dq.min():.3f}-{dq.max():.3f}   "
          f"ipTM mean {np.nanmean(ip):.3f} range {np.nanmin(ip):.3f}-{np.nanmax(ip):.3f}")
    c=collections.Counter(r["cls"] for r in csv.DictReader(raw.splitlines()))
    print(f"rows by class: {dict(c)}")
for a in alerts: print(f"   ALERT: {a}")
print("   all checks passed" if not alerts else f"   {len(alerts)} ALERT(S)")
open(f"{M}/.sanity_done","w").write(str(done))
EOF
  } >> "$OUT" 2>&1
  last=$(cat "$M/.sanity_done" 2>/dev/null || echo 0)
  sleep 1800
done
