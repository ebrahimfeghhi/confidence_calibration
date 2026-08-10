#!/usr/bin/env python
"""At what leaders setting does the best pose in the free pool actually get selected as a template?

Motivation: on 8che_BE the pose that works as a donor is seed 137 sample 3 (DockQ 0.529, the best in
the 250-pose free pool). The deployed leader20 arm did NOT include it -- its 20 donors span DockQ
0.087-0.436 -- so a leaders-based iterative loop starts without the one template known to reach 0.957.
This sweeps the two knobs (same-basin threshold, number of leaders kept) and reports where that pose
enters, and what the best donor available at each setting is.

LEADERS = greedy non-maximum suppression down the ipTM ranking: take the highest-ipTM pose, drop
every pose within `thresh` pose-pose DockQ of it, take the next survivor, repeat. thresh=0.23 is the
project standard (same CAPRI-acceptable basin). thresh > 1 disables suppression, i.e. plain top-N by
ipTM.

Inputs are the cached pose-pose matrix, which is model-vs-model DockQ, and the index CSV carrying
each pose's ipTM and its DockQ against the native. No labels enter the leaders computation --
dockq_gt is only used to REPORT what was reached.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/leaders_reach.py [--iface 8che_BE]
Writes: artifacts/leaders_reach_<iface>.json
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="8che_BE")
ap.add_argument("--matdir", default="/home/ebrahim/tts_abag/results/dockq_protenix")
A = ap.parse_args()

M = np.load(f"{A.matdir}/dockq_{A.iface}.npy")
rows = list(csv.DictReader(open(f"{A.matdir}/dockq_{A.iface}_index.csv")))
assert M.shape[0] == M.shape[1] == len(rows), f"matrix {M.shape} vs index {len(rows)}"
iptm = np.array([float(r["iptm"]) for r in rows])
gt = np.array([float(r["dockq_gt"]) for r in rows])
tag = [f"{r['seed']}_{r['sample']}" for r in rows]

best_i = int(np.argmax(gt))
print(f"{A.iface}: {len(rows)} free poses")
print(f"  best pose in pool: {tag[best_i]} DockQ {gt[best_i]:.3f} ipTM {iptm[best_i]:.3f} "
      f"(ipTM rank {int(np.sum(iptm > iptm[best_i])) + 1}/{len(rows)})\n")


def leaders(thresh):
    """Greedy NMS down the ipTM ranking. Returns pose indices in leader order."""
    order = np.argsort(-iptm)
    kept = []
    for i in order:
        if all(M[i, j] < thresh for j in kept):
            kept.append(int(i))
    return kept


out = {"iface": A.iface, "n_poses": len(rows), "best_tag": tag[best_i],
       "best_dockq": float(gt[best_i]), "best_iptm_rank": int(np.sum(iptm > iptm[best_i])) + 1,
       "settings": []}

print(f"{'thresh':>7} {'#leaders':>9} {'rank of best pose':>18} "
      + " ".join(f"{'best@' + str(n):>10}" for n in (10, 20, 30, 50)))
for th in (0.23, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01):
    L = leaders(th)
    rank = L.index(best_i) + 1 if best_i in L else None
    bests = {n: (float(max(gt[L[:n]])) if L else None) for n in (10, 20, 30, 50)}
    out["settings"].append({"thresh": th, "n_leaders": len(L),
                            "rank_of_best_pose": rank,
                            "best_dockq_at_N": {str(k): v for k, v in bests.items()}})
    r = f"{rank}" if rank else "NOT A LEADER"
    print(f"{th:>7.2f} {len(L):>9} {r:>18} "
          + " ".join(f"{bests[n]:>10.3f}" for n in (10, 20, 30, 50)))

print("\nInterpretation aid: 'rank of best pose' is where the 0.529 donor sits in leader order; "
      "'best@N' is the highest true DockQ among the first N leaders -- the best template that "
      "setting would hand you.")
json.dump(out, open(f"{EXP}/artifacts/leaders_reach_{A.iface}.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/leaders_reach_{A.iface}.json")
