#!/usr/bin/env python
"""DockQ cost of restricting to the top-K of the 0.8-redundancy-filtered ipTM ranking.

Ceiling arithmetic, not a selector: how much DockQ is given up by only ever LOOKING at the top K of
the filtered list, versus having the whole 250-pose pool available. Both sides are oracle-best, so
this isolates the cost of the shortlist itself and says nothing about whether ipTM could find the
pose inside it.

  full_best      best DockQ over all 250 free-run poses
  shortlist_best best DockQ among the first K survivors of greedy NMS (drop a pose whose pose-pose
                 DockQ to an already-kept higher-ipTM pose is >= 0.8)
  loss           full_best - shortlist_best  (>= 0 by construction)

Also reports whether the shortlist still clears the CAPRI-acceptable line (0.23) where the full pool
did, since a loss that crosses that line costs an interface and a loss that does not is cosmetic.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/shortlist_cost.py [--k 100] [--t 0.8]
Writes: artifacts/shortlist_cost.json
"""
import argparse, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--t", type=float, default=0.8)
A = ap.parse_args()

rows = []
for f in sorted(glob.glob(f"{EXP}/artifacts/pose_pose_dockq/*_index.csv")):
    lab = os.path.basename(f).replace("dockq_", "").replace("_seeds101-150_index.csv", "")
    M = np.load(f.replace("_index.csv", ".npy"))
    rec = list(csv.DictReader(open(f)))
    if len(rec) != M.shape[0]:
        print(f"{lab:11} index/matrix mismatch -- SKIPPED", flush=True); continue
    ip = np.array([float(r["iptm"]) for r in rec])
    gt = np.array([float(r["dockq_gt"]) for r in rec])
    kept = []
    for i in np.argsort(-ip):
        if all(M[i, j] < A.t for j in kept):
            kept.append(int(i))
    short = kept[:A.k]
    fb, sb = float(gt.max()), float(gt[short].max())
    rows.append({"interface": lab, "n_kept": len(kept), "k": len(short),
                 "full_best": round(fb, 3), "shortlist_best": round(sb, 3),
                 "loss": round(fb - sb, 3),
                 "full_acceptable": fb >= THRESH, "shortlist_acceptable": sb >= THRESH})

rows.sort(key=lambda r: -r["loss"])
print(f"{'interface':11} {'full-pool best':>14} {'top-'+str(A.k)+' best':>13} {'loss':>7}   note")
for r in rows:
    note = ""
    if r["full_acceptable"] and not r["shortlist_acceptable"]:
        note = "CROSSES 0.23 -- interface lost"
    elif r["loss"] == 0:
        note = "no loss"
    print(f"{r['interface']:11} {r['full_best']:14.3f} {r['shortlist_best']:13.3f} "
          f"{r['loss']:7.3f}   {note}")
L = [r["loss"] for r in rows]
n = len(rows)
print(f"\n  n = {n} interfaces, shortlist = top {A.k} of the {A.t}-filtered ipTM ranking")
print(f"  loss: median {np.median(L):.3f}  mean {np.mean(L):.3f}  max {max(L):.3f}")
print(f"  zero loss (best pose is in the shortlist): {sum(1 for x in L if x == 0)}/{n}")
print(f"  loss <= 0.05: {sum(1 for x in L if x <= 0.05)}/{n}   <= 0.10: {sum(1 for x in L if x <= 0.10)}/{n}")
print(f"  acceptable (>=0.23) in full pool {sum(1 for r in rows if r['full_acceptable'])}/{n}"
      f"  -> in shortlist {sum(1 for r in rows if r['shortlist_acceptable'])}/{n}")
json.dump({"k": A.k, "t": A.t, "n": n, "median_loss": float(np.median(L)),
           "mean_loss": float(np.mean(L)), "rows": rows},
          open(f"{EXP}/artifacts/shortlist_cost.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/shortlist_cost.json")
