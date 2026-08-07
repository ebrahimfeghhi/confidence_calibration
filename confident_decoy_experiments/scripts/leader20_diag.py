#!/usr/bin/env python
"""Why does pooled ipTM miss? Two hypotheses, separated on the poses already generated.

(a) NO DISCRIMINATION -- templating lifts ipTM on the decoy as much as on the good pose, so the
    good pose never out-scores the decoy inside the pool.
(b) TOO FEW SEEDS -- the good pose is barely generated at all, so there is usually nothing for
    selection to find.

For every interface this measures, over its whole leader20 pool (~500 poses):
    n_acceptable            how many good poses exist       -> speaks to (b)
    best ipTM among good    vs best ipTM among decoys       -> speaks to (a)
    gap                     positive means the good pose WOULD have been selected

Usage: python scripts/leader20_diag.py
Writes: artifacts/leader20_diagnosis.json
"""
import json, os, sys, glob, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SAMPLES, THRESH

OUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"
CACHE = f"{EXP}/artifacts/leader20_ost"
jobs = json.load(open(f"{EXP}/artifacts/leader20_jobs.json"))
scores = json.load(open(f"{EXP}/artifacts/leader20_scores.json"))
free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
by = {}
for j in jobs: by.setdefault(j["interface"], []).append(j)

rows = []
for lab in sorted(scores):
    good, bad = [], []
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    for j in by[lab]:
        for s in j["seeds"]:
            for k in SAMPLES:
                o = f"{CACHE}/{lab}_L{j['leader_rank']:02d}_{s}_{k}.json"
                cf = (f"{OUT}/{j['name']}/seed_{s}/predictions/"
                      f"{j['name']}_summary_confidence_sample_{k}.json")
                if not (os.path.exists(o) and os.path.exists(cf)): continue
                try:
                    d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
                except Exception: continue
                qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
                for i, t in enumerate(ifs):
                    if {t[0], t[1]} == {c1, c2} and i < len(qs):
                        x, y = ord(t[2]) - 65, ord(t[3]) - 65
                        if 0 <= x < len(M) and 0 <= y < len(M):
                            (good if qs[i] >= THRESH else bad).append(M[x][y])
                        break
    if not bad: continue
    bg = max(good) if good else None
    rows.append({"interface": lab, "n_poses": len(good)+len(bad), "n_good": len(good),
                 "frac_good": round(len(good)/(len(good)+len(bad)), 4),
                 "best_iptm_good": round(bg,3) if bg else None,
                 "best_iptm_decoy": round(max(bad),3),
                 "gap": round(bg-max(bad),3) if bg else None,
                 "free_frac_good": round((free[lab]["free_n_acceptable"] or 0)/250, 4),
                 "rescued": scores[lab]["method_rescued"]})

def gaptxt(r):
    return f"{r['gap']:+.3f}" if r['gap'] is not None else '  --'


print(f"{'interface':11} {'good/total':>12} {'free frac':>9} | {'best ipTM good':>14} "
      f"{'best ipTM decoy':>15} {'gap':>7}")
for r in rows:
    print(f"{r['interface']:11} {r['n_good']:5d}/{r['n_poses']:<6} {r['free_frac_good']:9.1%} | "
          f"{str(r['best_iptm_good']):>14} {r['best_iptm_decoy']:15.3f} "
          f"{gaptxt(r):>7}  {'RESCUED' if r['rescued'] else ''}")
ng = [r for r in rows if r["n_good"] == 0]
hav = [r for r in rows if r["n_good"] > 0]
pos = [r for r in hav if r["gap"] is not None and r["gap"] > 0]
print(f"\n  interfaces with NO good pose generated at all : {len(ng)}/{len(rows)}  -> hypothesis (b)")
print(f"  interfaces where a good pose exists          : {len(hav)}/{len(rows)}")
print(f"    ... and it OUT-CONFIDENCES every decoy     : {len(pos)}/{len(hav)}  -> would be selected")
if hav:
    print(f"  median good-pose fraction in the pool: {st.median(r['frac_good'] for r in hav):.1%} "
          f"(free run: {st.median(r['free_frac_good'] for r in hav):.1%})")
    print(f"  median gap (best good ipTM - best decoy ipTM): "
          f"{st.median(r['gap'] for r in hav if r['gap'] is not None):+.3f}")
json.dump(rows, open(f"{EXP}/artifacts/leader20_diagnosis.json","w"), indent=1)
