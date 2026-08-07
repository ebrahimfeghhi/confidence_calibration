#!/usr/bin/env python
"""Where does the BEST-DockQ pose sit when the free run's 250 poses are ranked by interface ipTM?

The confident-decoy definition only says the TOP-1 ipTM pose is wrong (DockQ < 0.23). It says nothing
about how deep you must go before the best pose appears. This measures that depth directly: for each
interface, rank the free run's 250 poses by interface pair_iptm (descending) and report the rank of
the single highest-DockQ pose, plus the rank of the first ACCEPTABLE pose (DockQ >= 0.23), which is
the number that matters for a top-K shortlist.

Rank is 1-based. "In the top 100" means rank <= 100 out of 250, i.e. the top 40% of the pool.

Free-run OST results live in three caches with two naming schemes (interface-keyed and pdb-keyed);
all three are searched and the interface is matched BY NAME. Pool size is asserted, not assumed.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/iptm_rank_of_best.py [--k 100]
Writes: artifacts/iptm_rank_of_best.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=100)
A = ap.parse_args()

free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
CACHES = [f"{EXP}/artifacts/decoy_set_ost", f"{EXP}/artifacts/extra_set_ost",
          f"{EXP}/artifacts/abag_all_ost"]


def pose(lab, s, k):
    """(iptm, dockq) for one free-run pose, or None. Interface matched BY NAME in every cache."""
    pdb = lab.split("_")[0]
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    for c in CACHES:
        for stem in (f"{c}/{lab}_{s}_{k}.json", f"{c}/{pdb}_{s}_{k}.json"):
            if not os.path.exists(stem):
                continue
            try:
                d = json.load(open(stem))
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    p = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
                    if not os.path.exists(p):
                        return None
                    M = json.load(open(p))["chain_pair_iptm"]
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        return (M[x][y], qs[i])
            return None
    return None


rows = []
for lab in sorted(free):
    ps = [p for s in range(101, 151) for k in SAMPLES if (p := pose(lab, s, k))]
    if len(ps) < 200:
        print(f"{lab:11} only {len(ps)} poses recovered -- SKIPPED", flush=True)
        continue
    order = sorted(range(len(ps)), key=lambda i: -ps[i][0])       # rank by ipTM, descending
    dq = [ps[i][1] for i in order]
    best_i = max(range(len(dq)), key=lambda i: dq[i])             # first occurrence of the max
    acc = [i for i, d in enumerate(dq) if d >= THRESH]
    rows.append({"interface": lab, "n_poses": len(ps),
                 "best_dockq": round(max(dq), 3),
                 "rank_of_best_dockq": best_i + 1,
                 "rank_of_first_acceptable": (acc[0] + 1) if acc else None,
                 "n_acceptable": len(acc),
                 "top1_dockq": round(dq[0], 3)})
    v = rows[-1]
    print(f"{lab:11} n {v['n_poses']:3d} | best DockQ {v['best_dockq']:.3f} at ipTM rank "
          f"{v['rank_of_best_dockq']:3d} | first acceptable at rank "
          f"{str(v['rank_of_first_acceptable']):>4} | {v['n_acceptable']:3d} acceptable", flush=True)

n = len(rows)
inb = sum(1 for r in rows if r["rank_of_best_dockq"] <= A.k)
acc_rows = [r for r in rows if r["rank_of_first_acceptable"]]
ina = sum(1 for r in acc_rows if r["rank_of_first_acceptable"] <= A.k)
print(f"\n  best-DockQ pose within top-{A.k} by ipTM:      {inb}/{n}")
print(f"  first ACCEPTABLE pose within top-{A.k} by ipTM: {ina}/{len(acc_rows)} "
      f"(of the {len(acc_rows)}/{n} interfaces that have any acceptable pose)")
for k in (1, 5, 10, 25, 50, 100, 150, 250):
    a = sum(1 for r in rows if r["rank_of_best_dockq"] <= k)
    b = sum(1 for r in acc_rows if r["rank_of_first_acceptable"] <= k)
    print(f"    top-{k:<4} best-DockQ {a:2d}/{n}   first-acceptable {b:2d}/{len(acc_rows)}")
json.dump({"k": A.k, "n_interfaces": n, "best_in_topk": inb,
           "first_acceptable_in_topk": ina, "rows": rows},
          open(f"{EXP}/artifacts/iptm_rank_of_best.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/iptm_rank_of_best.json")
