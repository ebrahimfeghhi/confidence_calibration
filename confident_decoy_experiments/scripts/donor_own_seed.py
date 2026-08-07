#!/usr/bin/env python
"""In the dockqdonor arm, is a donor's OWN seed a good seed to run it under?

Each dockqdonor job templates on one donor pose (produced by seed S, sample K) and then runs 50 seeds
101-150, so every interface contains a natural experiment: one of those 50 run-seeds IS the donor's
originating seed S. If templating worked specially under the donor's own seed, S would rank near the
top of the 50; if the donor effect is seed-independent, S should sit at a random rank.

Per interface: the max DockQ of each run-seed's 5 samples, then where the donor's own seed falls in
that ranking (1 = best of 50), and whether it cleared 0.23 at all.

Expected rank under the null (own seed is nothing special) is 25.5; a mean rank near that with a
uniform spread is the signature of no own-seed effect.

Interface matched BY NAME in every OST json (never dockq[0]).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/donor_own_seed.py
Writes: artifacts/donor_own_seed.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

donors = json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
CACHE = f"{EXP}/artifacts/dockqdonor_ost"


def dq(lab, s, k):
    f = f"{CACHE}/{lab}_{s}_{k}.json"
    if not (os.path.exists(f) and os.path.getsize(f)):
        return None
    try:
        d = json.load(open(f))
    except Exception:
        return None
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            return qs[i]
    return None


rows = []
for lab, D in sorted(donors.items()):
    own = D["seed"]
    mx = {}
    for s in SEEDS:
        v = [x for x in (dq(lab, s, k) for k in SAMPLES) if x is not None]
        if v:
            mx[s] = max(v)
    if own not in mx or len(mx) < 40:
        print(f"{lab:11} only {len(mx)}/50 seeds scored (own seed present: {own in mx}) -- SKIPPED")
        continue
    order = sorted(mx, key=lambda s: -mx[s])
    rank = order.index(own) + 1
    vals = np.array(list(mx.values()))
    rows.append({"interface": lab, "donor_seed": own, "donor_dockq": D["donor_dockq"],
                 "n_seeds": len(mx), "own_seed_max": round(mx[own], 3),
                 "own_seed_rank": rank, "own_seed_pct": round(100 * (1 - (rank - 1) / len(mx))),
                 "own_seed_acceptable": mx[own] >= THRESH,
                 "median_seed_max": round(float(np.median(vals)), 3),
                 "best_seed_max": round(float(vals.max()), 3),
                 "n_seeds_acceptable": int((vals >= THRESH).sum())})

print(f"{'interface':11} {'donor seed':>10} {'own-seed max':>12} {'rank/50':>8} {'median seed':>11} "
      f"{'best seed':>10} {'seeds ok':>9}")
for r in rows:
    print(f"{r['interface']:11} {r['donor_seed']:10d} {r['own_seed_max']:12.3f} "
          f"{r['own_seed_rank']:5d}/{r['n_seeds']:<3} {r['median_seed_max']:11.3f} "
          f"{r['best_seed_max']:10.3f} {r['n_seeds_acceptable']:6d}/{r['n_seeds']}")

R = [r["own_seed_rank"] for r in rows]
n = len(rows)
print(f"\n  n = {n} interfaces")
print(f"  own-seed rank: mean {np.mean(R):.1f}, median {np.median(R):.1f}  "
      f"(expected 25.5 if the own seed is nothing special)")
print(f"  own seed in the TOP 10 of 50: {sum(1 for x in R if x <= 10)}/{n}   "
      f"(expected {0.2*n:.1f} by chance)")
print(f"  own seed in the BOTTOM 10 of 50: {sum(1 for x in R if x > 40)}/{n}")
print(f"  own seed produced an acceptable pose: {sum(1 for r in rows if r['own_seed_acceptable'])}/{n}")
print(f"  ANY seed produced an acceptable pose: {sum(1 for r in rows if r['n_seeds_acceptable'] > 0)}/{n}")
json.dump({"n": n, "mean_own_rank": float(np.mean(R)), "rows": rows},
          open(f"{EXP}/artifacts/donor_own_seed.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/donor_own_seed.json")
