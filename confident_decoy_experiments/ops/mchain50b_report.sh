#!/usr/bin/env bash
# Rolling summary of the pinned-config chain run. Rewrites a single report every 5 minutes.
#
# Scoring is NOT done here -- each worker already runs PXMeter on its 5 poses and appends to
# results.csv before starting the next round. This only aggregates what is already on disk, so it
# costs nothing and cannot interfere with the run.
#
# THE HEADLINE METRIC is the ipTM rank of the first acceptable pose. On the original 250 sweep poses
# that rank was 159/250 -- worse than chance, with zero good poses in the entire top 50. After 10
# rounds of self-templating (stock config) it moved to 6/250, with 5 good poses in the top 10. This
# run asks whether that survives when MSA depth and dropout are pinned, i.e. whether the improvement
# was real or an artifact of seed-lottery averaging.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
M=$REPO/artifacts/mchain50b
cd "$REPO" || exit 1
while true; do
  {
    echo "=== $(date '+%F %H:%M:%S')  pinned config: full MSA (15828 rows), no MC dropout, seed 1"
    echo "seeds done $(ls $M/.done 2>/dev/null | wc -l)/50   rows $(( $(wc -l < $M/results.csv 2>/dev/null || echo 1) - 1 ))/750   workers $(tmux ls 2>/dev/null | grep -cE '^mcw[0-9]:')"
    /home/ebrahim/Protenix/.venv/bin/python - <<'EOF'
import csv, collections
import numpy as np
GOOD={300014,300015,300038,300002,300024}
try: rows=list(csv.DictReader(open("artifacts/mchain50b/results.csv")))
except OSError: raise SystemExit
by=collections.defaultdict(list); allr=collections.defaultdict(list)
for r in rows:
    try: d,p=float(r["dockq"]),float(r["iptm"])
    except (ValueError,KeyError): continue
    by[(int(r["seed"]) in GOOD,int(r["round"]))].append((d,p))
    allr[int(r["round"])].append((d,p))
if not allr: raise SystemExit
for lab,flag in (("good-seed chains",True),("decoy-seed chains",False)):
    line=[]
    for rd in sorted(allr):
        v=by.get((flag,rd),[])
        if not v: continue
        d=np.array([x[0] for x in v]); p=np.array([x[1] for x in v])
        line.append(f"r{rd}: DockQ {d.mean():.3f} acc {100*(d>=0.23).mean():.0f}% ipTM {np.nanmean(p):.3f} (n={len(d)})")
    if line: print(f"  {lab:<18} " + " | ".join(line))
# the metric that matters: where does ipTM put the first acceptable pose?
for rd in sorted(allr):
    v=allr[rd]; d=np.array([x[0] for x in v]); p=np.array([x[1] for x in v])
    ok=np.isfinite(p)
    if ok.sum()<10: continue
    d,p=d[ok],p[ok]; g=d>=0.23
    if not g.any():
        print(f"  round {rd}: no acceptable poses yet (n={len(d)})"); continue
    o=np.argsort(-p); rk=int(np.where(g[o])[0][0])+1
    print(f"  round {rd}: first good pose at ipTM rank {rk}/{len(d)}"
          f"   top-10 best DockQ {d[o[:10]].max():.3f}   acceptable {int(g.sum())}")
EOF
  } > "$M/report.txt" 2>&1
  sleep 300
done
