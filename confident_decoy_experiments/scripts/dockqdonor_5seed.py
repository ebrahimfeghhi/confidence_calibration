#!/usr/bin/env python
"""dockqdonor at 5 seeds instead of 50: how many interfaces still get rescued?

The arm was run at 50 seeds (250 poses/interface) and rescues 9/25. A deployable version cannot
afford 50 seeds per donor, so this re-scores the SAME runs restricted to a 5-seed subset -- selection
rule unchanged (highest interface pair_iptm), just a smaller pool.

Two numbers, because they answer different questions:
  first5     seeds 101-105 specifically -- one concrete draw, what you would actually have got
  bootstrap  mean over random 5-seed subsets -- how much of `first5` is luck of the draw

Reporting only `first5` would present a single draw as the method's rate; reporting only the mean
would hide that any real run gets one draw. Both are shown.

Interface matched BY NAME in every OST json.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/dockqdonor_5seed.py [--trials 2000]
Writes: artifacts/dockqdonor_5seed.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=2000)
ap.add_argument("--k", type=int, default=5)
A = ap.parse_args()

jobs = {j["interface"]: j["name"] for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
full = json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
OUT = f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"


def poses(lab):
    """{seed: [(iptm, dockq), ...]} for one interface."""
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    name = jobs[lab]
    per = {}
    for s in SEEDS:
        v = []
        for k in SAMPLES:
            f = f"{EXP}/artifacts/dockqdonor_ost/{lab}_{s}_{k}.json"
            try:
                d = json.load(open(f))
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    p = (f"{OUT}/{name}/seed_{s}/predictions/"
                         f"{name}_summary_confidence_sample_{k}.json")
                    try:
                        M = json.load(open(p))["chain_pair_iptm"]
                    except Exception:
                        break
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        v.append((M[x][y], qs[i]))
                    break
        if v:
            per[s] = v
    return per


rng = np.random.default_rng(0)
rows = []
for lab in sorted(jobs):
    per = poses(lab)
    if len(per) < 40:
        print(f"{lab:11} only {len(per)}/50 seeds scored -- SKIPPED")
        continue
    first = [s for s in sorted(per)[:A.k]]
    fv = [x for s in first for x in per[s]]
    f_sel = max(fv, key=lambda x: x[0])[1]
    ss = list(per)
    hits, sels = 0, []
    for _ in range(A.trials):
        pick = rng.choice(ss, size=A.k, replace=False)
        v = [x for s in pick for x in per[s]]
        sd = max(v, key=lambda x: x[0])[1]
        sels.append(sd); hits += sd >= THRESH
    rows.append({"interface": lab, "n_seeds": len(per),
                 "full50_sel": full.get(lab, {}).get("sel_dockq"),
                 "full50_rescued": full.get(lab, {}).get("rescued"),
                 "first5_sel": round(f_sel, 3), "first5_rescued": f_sel >= THRESH,
                 "boot_p_rescue": round(hits / A.trials, 3),
                 "boot_median_sel": round(float(np.median(sels)), 3)})

n = len(rows)
print(f"{'interface':11} {'50-seed sel':>11} {'':>2} {'first-5 sel':>11} {'':>2} {'P(rescue|5 seeds)':>18}")
for r in rows:
    print(f"{r['interface']:11} {r['full50_sel']:11.3f} {'*' if r['full50_rescued'] else ' ':>2} "
          f"{r['first5_sel']:11.3f} {'*' if r['first5_rescued'] else ' ':>2} {r['boot_p_rescue']:18.2f}")
f50 = sum(1 for r in rows if r["full50_rescued"])
f5 = sum(1 for r in rows if r["first5_rescued"])
exp = sum(r["boot_p_rescue"] for r in rows)
print(f"\n  dockqdonor at 50 seeds: {f50}/{n} rescued")
print(f"  dockqdonor at first 5 seeds (101-105): {f5}/{n}")
print(f"  expected rescues over random 5-seed draws: {exp:.1f}/{n}  "
      f"(sum of per-interface P(rescue))")
print(f"  interfaces rescued at 50 that are near-certain at 5 (P>=0.8): "
      f"{sum(1 for r in rows if r['full50_rescued'] and r['boot_p_rescue'] >= 0.8)}/{f50}")
print(f"  interfaces rescued at 50 that are coin-flips at 5 (P<0.5): "
      f"{sum(1 for r in rows if r['full50_rescued'] and r['boot_p_rescue'] < 0.5)}/{f50}")
json.dump({"k": A.k, "n": n, "rescued_50": f50, "rescued_first5": f5,
           "expected_5": exp, "rows": rows},
          open(f"{EXP}/artifacts/dockqdonor_5seed.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/dockqdonor_5seed.json")
