#!/usr/bin/env python
"""Seed budget for the dockqdonor arm over ALL 25 interfaces, not just the 9 it rescues.

seed_efficiency.py reports P(rescue) per interface; the "expected rescues" figure quoted from it
was summed over the 9 rescued arms only, which answers "how many of the achievable ones do we get"
but not "what does the whole benchmark look like at K seeds". This does the latter, and adds mean
selected DockQ so the answer is not purely a threshold count.

For each interface and each K: draw K of the arm's 50 seeds at random, pool their 5*K poses, take
the HIGHEST-pair_iptm pose (the actual selection rule -- no labels), and record
  (a) whether its DockQ >= 0.23   -> rescue count
  (b) its DockQ                   -> mean selected DockQ
averaged over --trials draws. Then aggregate across all 25 interfaces.

Per-pose (pair_iptm, DockQ) are re-read from the arm's OST cache, matching seed_efficiency.py, so
this re-derives from artifacts rather than trusting any aggregate file.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/seed_budget_all25.py [--arm dockqdonor]
Writes: artifacts/<arm>_seed_budget_all25.json
"""
import argparse, json, os, random, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="dockqdonor")
ap.add_argument("--trials", type=int, default=400)
A = ap.parse_args()

CACHE = f"{EXP}/artifacts/{A.arm}_ost"
OUT = f"{EXP}/artifacts/sweep_out/{A.arm}/{A.arm}"
jobs = {j["interface"]: j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
scores = json.load(open(f"{EXP}/artifacts/{A.arm}_scores.json"))
rng = random.Random(0)
KS = [1, 2, 3, 5, 10, 25, 50]


def per_pose(lab):
    """[(seed, sample, pair_iptm, dockq)] from the arm's OST cache + confidence JSONs."""
    job = jobs[lab]; name = job["name"]
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    out = []
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{CACHE}/{lab}_{s}_{k}.json"
            cf = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)):
                continue
            try:
                d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):        # BY NAME
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        out.append((s, k, M[x][y], qs[i]))
                    break
    return out


rows, skipped = [], []
for lab in sorted(scores):
    poses = per_pose(lab)
    if len(poses) < 200:
        skipped.append((lab, len(poses))); continue
    by_seed = {}
    for s, k, ip, dq in poses:
        by_seed.setdefault(s, []).append((ip, dq))
    seeds = sorted(by_seed)
    rec = {"interface": lab, "n_seeds": len(seeds), "rescued_at_50": scores[lab]["rescued"],
           "p_by_k": {}, "meandq_by_k": {}}
    for K in KS:
        if K > len(seeds):
            rec["p_by_k"][K] = None; rec["meandq_by_k"][K] = None; continue
        hits, dqs = 0, []
        for _ in range(A.trials):
            sub = rng.sample(seeds, K)
            pool = [p for s_ in sub for p in by_seed[s_]]
            sel = max(pool, key=lambda x: x[0])[1]          # highest pair_iptm -> its DockQ
            hits += sel >= THRESH
            dqs.append(sel)
        rec["p_by_k"][K] = round(hits / A.trials, 3)
        rec["meandq_by_k"][K] = round(st.mean(dqs), 4)
    rows.append(rec)

n = len(rows)
print(f"{A.arm}: {n} interfaces (all, not just the rescued ones); {len(skipped)} skipped")
for lab, c in skipped:
    print(f"  SKIP {lab}: only {c} poses")

print(f"\nACROSS ALL {n} INTERFACES")
print(f"{'seeds K':>8} {'poses':>7} {'E[rescued]':>11} {'rescue rate':>12} {'mean sel DockQ':>15}")
agg = {}
for K in KS:
    v = [r["p_by_k"][K] for r in rows if r["p_by_k"][K] is not None]
    m = [r["meandq_by_k"][K] for r in rows if r["meandq_by_k"][K] is not None]
    agg[K] = {"expected_rescued": round(sum(v), 2), "rescue_rate": round(sum(v) / len(v), 3),
              "mean_sel_dockq": round(st.mean(m), 4), "n_interfaces": len(v)}
    print(f"{K:>8} {K * 5:>7} {sum(v):>11.2f} {sum(v)/len(v):>12.3f} {st.mean(m):>15.3f}")

resc = [r for r in rows if r["rescued_at_50"]]
print(f"\nfor contrast, the {len(resc)} interfaces rescued at 50 seeds")
print(f"{'seeds K':>8} {'E[rescued]':>11} {'mean sel DockQ':>15}")
for K in KS:
    v = [r["p_by_k"][K] for r in resc if r["p_by_k"][K] is not None]
    m = [r["meandq_by_k"][K] for r in resc if r["meandq_by_k"][K] is not None]
    print(f"{K:>8} {sum(v):>11.2f} {st.mean(m):>15.3f}")

json.dump({"arm": A.arm, "trials": A.trials, "n_interfaces": n, "KS": KS,
           "aggregate_all": agg, "rows": rows, "skipped": skipped},
          open(f"{EXP}/artifacts/{A.arm}_seed_budget_all25.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/{A.arm}_seed_budget_all25.json")
