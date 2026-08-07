#!/usr/bin/env python
"""Does redundancy-filtering the ipTM ranking pull the BEST-DockQ pose into the top K?

Unfiltered, the free run's best-DockQ pose sits at median ipTM rank ~148/250 and is in the top 100
for only 8/26 interfaces. Hypothesis: the ipTM ranking is clogged with near-duplicates of the decoy,
so deduplicating it promotes genuinely distinct poses.

FILTER (greedy NMS on ipTM). Walk poses in descending interface ipTM. Keep a pose only if its
POSE-POSE DockQ to every already-kept pose is < T. A pose within T of a higher-ipTM pose is dropped
as redundant. T=0.8/0.9 keeps only near-identical poses out; T=0.23 is the old `leaders` setting.

TARGET. The user's target is the single best-DockQ pose, or any pose within `--tol` (default 0.02)
DockQ of it. So success = "a pose with dockq_gt >= best - tol appears in the kept list at rank <= K".

THE TRAP THIS SCRIPT MUST NOT FALL INTO: filtering can DELETE the best pose (if it is redundant with
a higher-ipTM pose). Reporting only the rank of survivors would then silently turn a deletion into a
better-looking number. So every run reports `target_survived` separately from `target_rank`, and the
headline counts only interfaces where a qualifying pose actually survives.

Pose-pose DockQ matrices: artifacts/pose_pose_dockq/dockq_<iface>_seeds101-150.npy with a row index
CSV carrying seed, sample, iptm and dockq_gt -- ipTM and ground-truth DockQ are read from that CSV,
so ranking and scoring use one consistent source.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/nms_shortlist.py [--tol 0.02]
Writes: artifacts/nms_shortlist.json
"""
import argparse, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.02, help="a pose counts as the target if within this of best")
ap.add_argument("--thresholds", default="1.01,0.9,0.8,0.6,0.4,0.23")
ap.add_argument("--ks", default="10,25,50,100")
A = ap.parse_args()
THR = [float(x) for x in A.thresholds.split(",")]     # 1.01 = no filtering (nothing is >= 1.01)
KS = [int(x) for x in A.ks.split(",")]
D = f"{EXP}/artifacts/pose_pose_dockq"


def nms(M, order, t):
    """Greedy NMS: walk `order` (descending ipTM), keep a pose only if max pose-pose DockQ to the
    already-kept set is < t. Returns the kept row indices, in kept order."""
    kept = []
    for i in order:
        if all(M[i, j] < t for j in kept):
            kept.append(i)
    return kept


rows, per_thr = [], {t: {k: 0 for k in KS} for t in THR}
survive = {t: 0 for t in THR}
for f in sorted(glob.glob(f"{D}/*_index.csv")):
    lab = os.path.basename(f).replace("dockq_", "").replace("_seeds101-150_index.csv", "")
    M = np.load(f.replace("_index.csv", ".npy"))
    rec = list(csv.DictReader(open(f)))
    if len(rec) != M.shape[0]:
        print(f"{lab:11} index/matrix mismatch {len(rec)} vs {M.shape} -- SKIPPED", flush=True)
        continue
    ip = np.array([float(r["iptm"]) for r in rec])
    gt = np.array([float(r["dockq_gt"]) for r in rec])
    best = gt.max()
    tgt = set(np.where(gt >= best - A.tol)[0].tolist())    # best pose, or within tol of it
    order = list(np.argsort(-ip))
    row = {"interface": lab, "n_poses": len(rec), "best_dockq": round(float(best), 3),
           "n_target_poses": len(tgt), "by_threshold": {}}
    for t in THR:
        kept = nms(M, order, t)
        hit = [r for r, i in enumerate(kept, 1) if i in tgt]
        surv = bool(hit)
        rank = hit[0] if hit else None
        row["by_threshold"][f"{t}"] = {"n_kept": len(kept), "target_survived": surv,
                                       "target_rank": rank}
        if surv:
            survive[t] += 1
            for k in KS:
                if rank <= k:
                    per_thr[t][k] += 1
    rows.append(row)

n = len(rows)
print(f"{n} interfaces | target = best-DockQ pose or within {A.tol} of it\n")
print(f"{'pose-pose DockQ filter':24} {'median kept':>11} {'target survives':>15} | "
      + " ".join(f"in top-{k:<4}" for k in KS))
for t in THR:
    mk = int(np.median([r["by_threshold"][f"{t}"]["n_kept"] for r in rows]))
    lbl = "none (raw ipTM rank)" if t > 1 else f"drop if >= {t} to a higher-ipTM pose"
    print(f"{lbl:24} {mk:11d} {survive[t]:12d}/{n} | "
          + " ".join(f"{per_thr[t][k]:6d}/{n} " for k in KS))
print(f"\n  'target survives' = a qualifying pose was NOT deleted by the filter. "
      f"top-K counts are out of all {n}, so deletions count as failures.")
json.dump({"tol": A.tol, "n_interfaces": n, "thresholds": THR, "ks": KS,
           "survive": {str(k): v for k, v in survive.items()},
           "in_topk": {str(t): per_thr[t] for t in THR}, "rows": rows},
          open(f"{EXP}/artifacts/nms_shortlist.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/nms_shortlist.json")
