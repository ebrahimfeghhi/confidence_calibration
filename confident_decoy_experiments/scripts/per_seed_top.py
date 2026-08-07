#!/usr/bin/env python
"""Take the highest-ipTM pose from EACH seed (50 poses of 250). Is the best-DockQ pose in that set?

A different shortlist shape from top-K-of-a-global-ranking: instead of ranking all 250 poses together,
keep one representative per seed -- the sample that seed was most confident in. 50 poses, 20% of the
pool, no cross-seed comparison, no redundancy filter, no native.

  hit            the single global best-DockQ pose is one of the 50
  hit_tol        some pose in the 50 is within --tol DockQ of the global best
  shortlist_best best DockQ among the 50 (oracle over the shortlist)
  loss           global best - shortlist_best

Read alongside the top-100-of-0.8-filtered shortlist (scripts/shortlist_cost.py): that one keeps 100
poses and loses a median 0.003; this keeps 50 and is the cheaper comparison point.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/per_seed_top.py [--tol 0.05]
Writes: artifacts/per_seed_top.json
"""
import argparse, csv, glob, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.05)
A = ap.parse_args()

rows = []
for f in sorted(glob.glob(f"{EXP}/artifacts/pose_pose_dockq/*_index.csv")):
    lab = os.path.basename(f).replace("dockq_", "").replace("_seeds101-150_index.csv", "")
    rec = list(csv.DictReader(open(f)))
    ip = np.array([float(r["iptm"]) for r in rec])
    gt = np.array([float(r["dockq_gt"]) for r in rec])
    byseed = collections.defaultdict(list)
    for i, r in enumerate(rec):
        byseed[int(r["seed"])].append(i)
    sel = [max(idx, key=lambda i: ip[i]) for _, idx in sorted(byseed.items())]   # top ipTM per seed
    fb = float(gt.max()); best_i = int(np.argmax(gt))
    sb = float(gt[sel].max())
    rows.append({"interface": lab, "n_seeds": len(byseed), "n_selected": len(sel),
                 "full_best": round(fb, 3), "shortlist_best": round(sb, 3),
                 "loss": round(fb - sb, 3),
                 "hit_exact": best_i in sel, "hit_within_tol": sb >= fb - A.tol,
                 "full_acceptable": fb >= THRESH, "shortlist_acceptable": sb >= THRESH})

rows.sort(key=lambda r: -r["loss"])
n = len(rows)
print(f"{'interface':11} {'full-pool best':>14} {'best of the 50':>14} {'loss':>7}   note")
for r in rows:
    note = ("CROSSES 0.23 -- interface lost" if r["full_acceptable"] and not r["shortlist_acceptable"]
            else "exact best pose included" if r["hit_exact"] else "")
    print(f"{r['interface']:11} {r['full_best']:14.3f} {r['shortlist_best']:14.3f} {r['loss']:7.3f}   {note}")
L = [r["loss"] for r in rows]
print(f"\n  one pose per seed = {rows[0]['n_selected']} of 250 (20% of the pool)")
print(f"  contains the EXACT best-DockQ pose:      {sum(1 for r in rows if r['hit_exact'])}/{n}")
print(f"  contains a pose within {A.tol} of it:      {sum(1 for r in rows if r['hit_within_tol'])}/{n}")
print(f"  loss: median {np.median(L):.3f}  mean {np.mean(L):.3f}  max {max(L):.3f}")
print(f"  keeps an acceptable (>=0.23) pose: "
      f"{sum(1 for r in rows if r['shortlist_acceptable'])}/{sum(1 for r in rows if r['full_acceptable'])}")
json.dump({"tol": A.tol, "n": n, "rows": rows}, open(f"{EXP}/artifacts/per_seed_top.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/per_seed_top.json")
