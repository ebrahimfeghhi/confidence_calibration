#!/usr/bin/env python
"""PROPER bias-variance decomposition of CDR-H3 / epitope error.

For coordinates y_sp (seed s, sample p) expressed in a COMMON frame (framework-superposed for H3,
non-epitope-superposed for epitope) and native y*:
    mean structure  ybar   = per-atom mean over all poses
    seed means      ybar_s = per-atom mean within seed s
    BIAS      = RMSD(ybar, y*)                     <- systematic, irreducible by sampling
    VAR_within= mean_s mean_p ||y_sp - ybar_s||^2  <- DIFFUSION (trunk fixed within a seed)
    VAR_between= mean_s ||ybar_s - ybar||^2        <- TRUNK (representation varies across seeds)
    identity: mean_sp ||y_sp - y*||^2 = BIAS^2 + VAR_within + VAR_between
Reports each as an RMSD-like quantity (sqrt).
"""
import json, statistics, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ARTIFACTS_DIR

def decomp(byseed):
    """byseed: {seed: [coords,...]}, coords = list of (x,y,z); plus '_nat' key for native."""
    nat = byseed.pop("_nat")
    n = len(nat)
    seeds = sorted(byseed)
    # per-seed mean structures
    ybar_s = {}
    for s in seeds:
        poses = byseed[s]
        ybar_s[s] = [tuple(sum(p[i][d] for p in poses)/len(poses) for d in range(3)) for i in range(n)]
    allposes = [p for s in seeds for p in byseed[s]]
    ybar = [tuple(sum(p[i][d] for p in allposes)/len(allposes) for d in range(3)) for i in range(n)]
    def msd(a,b): return sum((a[i][d]-b[i][d])**2 for i in range(n) for d in range(3))/n
    bias2 = msd(ybar, nat)
    var_w = statistics.mean([msd(p, ybar_s[s]) for s in seeds for p in byseed[s]])
    var_b = statistics.mean([msd(ybar_s[s], ybar) for s in seeds])
    tot   = statistics.mean([msd(p, nat) for p in allposes])
    return math.sqrt(bias2), math.sqrt(var_w), math.sqrt(var_b), math.sqrt(tot)

which = sys.argv[1]
data = json.load(open(f"{ARTIFACTS_DIR}/{which}/raw_coords.json"))
print(f"{'chain':14} {'n':>4} | {'BIAS(A)':>8} {'DIFFUSION(A)':>12} {'TRUNK(A)':>9} | {'total(A)':>8} | bias share")
print("-"*78)
rows=[]
for tag, d in data.items():
    byseed = {int(k): v for k,v in d["byseed"].items()}
    byseed["_nat"] = d["native"]
    b,w,bt,t = decomp(byseed)
    share = b*b/(t*t) if t>0 else 0
    rows.append({"chain":tag,"bias":round(b,2),"diffusion":round(w,2),"trunk":round(bt,2),
                 "total":round(t,2),"bias_share_of_MSD":round(share,3)})
    print(f"{tag:14} {d['n_poses']:4} | {b:8.2f} {w:12.2f} {bt:9.2f} | {t:8.2f} | {share*100:5.1f}%")
json.dump(rows, open(f"{ARTIFACTS_DIR}/{which}/bias_variance.json","w"), indent=1)
import statistics as st
print(f"\nMEDIAN bias share of total MSD: {st.median([r['bias_share_of_MSD'] for r in rows])*100:.1f}%")
print(f"MEDIAN diffusion {st.median([r['diffusion'] for r in rows]):.2f} A vs trunk {st.median([r['trunk'] for r in rows]):.2f} A")
