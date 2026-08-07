#!/usr/bin/env python
"""Why is dockqdonor's expected rescue count flat in seeds-per-donor?

Rescue needs TWO things: the sampled pool must CONTAIN an acceptable pose, and ipTM must SELECT one.
Adding seeds can only help the first. This decomposes the flat curve into those two factors:

  coverage   P(pool of K seeds contains a pose with DockQ >= 0.23)
  conversion P(top-ipTM pose is acceptable | the pool contains one)
  rescue     coverage x conversion  (= the flat quantity)

If coverage rises with K while conversion falls by a matching amount, the flatness is the product of
two real and opposing effects rather than a coincidence -- and it localises the problem to selection,
since coverage is the part extra sampling actually buys.

Bootstrap over random K-seed subsets, rng seeded, same pools as scripts/dockqdonor_5seed.py.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/why_flat.py [--trials 2000]
Writes: artifacts/why_flat.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=2000)
A = ap.parse_args()

jobs = {j["interface"]: j["name"] for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
OUT = f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"


def poses(lab):
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    name = jobs[lab]; per = {}
    for s in SEEDS:
        v = []
        for k in SAMPLES:
            try:
                d = json.load(open(f"{EXP}/artifacts/dockqdonor_ost/{lab}_{s}_{k}.json"))
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    try:
                        M = json.load(open(f"{OUT}/{name}/seed_{s}/predictions/"
                                           f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
                    except Exception:
                        break
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        v.append((M[x][y], qs[i]))
                    break
        if v:
            per[s] = v
    return per


P = {}
for lab in sorted(jobs):
    per = poses(lab)
    if len(per) >= 40:
        P[lab] = per
rng = np.random.default_rng(0)
KS = (1, 2, 3, 5, 10, 20, 35, 50)
res = {}
print(f"{'K seeds':>8} {'poses':>6} {'coverage':>9} {'conversion':>11} {'rescue':>8} {'E[resc]/'+str(len(P)):>10}"
      f"   coverage = pool HAS a good pose; conversion = ipTM PICKS one given it exists")
for K in KS:
    cov = con = resc = 0.0
    ncon = 0
    for lab, per in P.items():
        ss = list(per)
        c = h = r = 0
        for _ in range(A.trials):
            pick = ss if K >= len(ss) else rng.choice(ss, size=K, replace=False)
            v = [x for s in pick for x in per[s]]
            has = any(x[1] >= THRESH for x in v)
            sel = max(v, key=lambda x: x[0])[1] >= THRESH
            c += has; h += sel and has; r += sel
        cov += c / A.trials
        if c:
            con += h / c; ncon += 1
        resc += r / A.trials
    res[K] = {"coverage": cov / len(P), "conversion": con / ncon if ncon else None,
              "rescue": resc / len(P), "expected": resc}
    v = res[K]
    print(f"{K:>8} {K*5:>6} {v['coverage']:>9.3f} {v['conversion']:>11.3f} {v['rescue']:>8.3f} "
          f"{v['expected']:>10.1f}")
json.dump({"n_interfaces": len(P), "trials": A.trials, "by_K": res},
          open(f"{EXP}/artifacts/why_flat.json", "w"), indent=1)
c1, c2 = res[KS[0]], res[KS[-1]]
print(f"\n  K=1 -> K=50:  coverage {c1['coverage']:.3f} -> {c2['coverage']:.3f} "
      f"({c2['coverage']-c1['coverage']:+.3f}),  conversion {c1['conversion']:.3f} -> "
      f"{c2['conversion']:.3f} ({c2['conversion']-c1['conversion']:+.3f})")
print(f"\nWROTE {EXP}/artifacts/why_flat.json")
