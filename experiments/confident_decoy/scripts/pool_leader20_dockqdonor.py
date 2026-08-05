#!/usr/bin/env python
"""Pool leader20's poses WITH dockqdonor's, select by ipTM. Does a strong condition rescue the pooling?

leader20 alone rescues 3/25: its templates generate good poses (24/25 interfaces) but ipTM prefers a
decoy by a median 0.213. dockqdonor alone rescues 9/25 -- but its donor is chosen using the native, so
it is an ORACLE. This asks whether adding that one strong condition to the pool fixes selection, or
whether ipTM simply keeps picking the same confident decoys.

NOTE: the pooled result is NOT deployable -- dockqdonor's donor requires the answer. This is a
diagnostic about ipTM's discrimination, not a candidate method.

Usage: python scripts/pool_leader20_dockqdonor.py
Writes: artifacts/pooled_leader20_dockqdonor.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

L_OUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"; L_CACHE = f"{EXP}/artifacts/leader20_ost"
D_OUT = f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"; D_CACHE = f"{EXP}/artifacts/dockqdonor_ost"
ljobs = json.load(open(f"{EXP}/artifacts/leader20_jobs.json"))
djobs = {j["interface"]: j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
lead = json.load(open(f"{EXP}/artifacts/leader20_scores.json"))
dqs = json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
by = {}
for j in ljobs: by.setdefault(j["interface"], []).append(j)


def pull(cache, out, name, tag, lab, seeds):
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    res = []
    for s in seeds:
        for k in SAMPLES:
            o = f"{cache}/{tag}_{s}_{k}.json"
            cf = f"{out}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)): continue
            try:
                d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
            except Exception: continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M): res.append((M[x][y], qs[i]))
                    break
    return res


rows = []
for lab in sorted(set(lead) & set(dqs)):
    lp = []
    for j in by[lab]:
        lp += pull(L_CACHE, L_OUT, j["name"], f"{lab}_L{j['leader_rank']:02d}", lab, j["seeds"])
    dj = djobs[lab]
    dp = pull(D_CACHE, D_OUT, dj["name"], lab, lab, SEEDS)
    if not lp or not dp: continue
    pool = lp + dp
    pick = max(pool, key=lambda x: x[0])
    rows.append({"interface": lab, "n_leader20": len(lp), "n_dockqdonor": len(dp),
                 "leader20_alone": lead[lab]["method_rescued"],
                 "dockqdonor_alone": dqs[lab]["rescued"],
                 "pooled_dockq": round(pick[1], 3), "pooled_iptm": round(pick[0], 3),
                 "pooled_rescued": pick[1] >= THRESH,
                 "from_dockqdonor": pick in dp})

n = len(rows)
la = sum(r["leader20_alone"] for r in rows); da = sum(r["dockqdonor_alone"] for r in rows)
po = sum(r["pooled_rescued"] for r in rows)
print(f"{'interface':11} {'leader20':>9} {'dockqdonor':>11} {'POOLED':>8} {'picked DockQ':>12} {'source':>12}")
for r in rows:
    print(f"{r['interface']:11} {'R' if r['leader20_alone'] else '.':>9} "
          f"{'R' if r['dockqdonor_alone'] else '.':>11} {'R' if r['pooled_rescued'] else '.':>8} "
          f"{r['pooled_dockq']:12.3f} {'dockqdonor' if r['from_dockqdonor'] else 'leader20':>12}")
print(f"\n  leader20 alone   {la}/{n}")
print(f"  dockqdonor alone {da}/{n}   (ORACLE donor)")
print(f"  POOLED           {po}/{n}")
print(f"  pooling vs the better single arm: {po-max(la,da):+d}")
lost = [r['interface'] for r in rows if r['dockqdonor_alone'] and not r['pooled_rescued']]
gain = [r['interface'] for r in rows if r['pooled_rescued'] and not r['dockqdonor_alone']]
print(f"  lost by pooling : {lost}")
print(f"  gained          : {gain}")
json.dump(rows, open(f"{EXP}/artifacts/pooled_leader20_dockqdonor.json","w"), indent=1)
