#!/usr/bin/env python
"""Can a redundancy filter shrink the 250-pose free pool while RELIABLY keeping the best pose?

The template sweep costs 250 donors x 10 seeds. If near-duplicate poses can be dropped without
losing the pose that actually works as a donor, the sweep shrinks proportionally. The design
constraint the user set: get under ~150 survivors while keeping the best-DockQ pose, on ALL 25-26
interfaces -- not tuned on one.

FILTER = greedy suppression down the pair_iptm ranking: keep the highest-ipTM pose, drop every pose
whose pose-pose DockQ to a kept pose is >= thresh, continue. A HIGH thresh (0.8-0.9) removes only
near-identical poses; a LOW thresh (0.23) collapses whole basins and is far more aggressive. This is
the same "leaders" procedure, swept over the aggressiveness knob.

Two retention criteria are reported, because they answer different questions:
  strict   the single best-DockQ pose in the pool survives
  tol      SOME survivor is within `--tol` DockQ of the pool best (a near-equal donor is fine)

Pose-pose DockQ is the cached exact matrix (dockq_torch, DockQ 2.1.3), never ligand-RMSD.
DockQ-vs-native is used only to SCORE retention -- the filter itself sees only ipTM and pose-pose
geometry, so it is deployable without labels.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/redundancy_retention.py [--tol 0.05]
Writes: artifacts/redundancy_retention.json
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--matdir", default="/home/ebrahim/tts_abag/results/dockq_protenix")
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--target", type=int, default=150, help="survivor count the user wants to get under")
A = ap.parse_args()

THRESHOLDS = [0.23, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
ifaces = sorted(json.load(open(f"{EXP}/artifacts/leader20_index.json")))


def load(iface):
    M = np.load(f"{A.matdir}/dockq_{iface}.npy")
    rows = list(csv.DictReader(open(f"{A.matdir}/dockq_{iface}_index.csv")))
    assert M.shape[0] == M.shape[1] == len(rows), f"{iface}: matrix {M.shape} vs index {len(rows)}"
    return (M, np.array([float(r["iptm"]) for r in rows]),
            np.array([float(r["dockq_gt"]) for r in rows]))


def survivors(M, iptm, thresh):
    order = np.argsort(-iptm)
    kept = []
    for i in order:
        if all(M[i, j] < thresh for j in kept):
            kept.append(int(i))
    return kept


rows, per_iface = [], {}
for iface in ifaces:
    M, iptm, gt = load(iface)
    best = float(gt.max())
    rec = {"iface": iface, "n_poses": int(len(gt)), "pool_best": round(best, 3), "by_thresh": {}}
    for th in THRESHOLDS:
        S = survivors(M, iptm, th)
        sb = float(gt[S].max())
        rec["by_thresh"][str(th)] = {
            "n": len(S), "best_kept": round(sb, 3),
            "strict": bool(abs(sb - best) < 1e-9),
            "within_tol": bool(best - sb <= A.tol)}
    per_iface[iface] = rec
    rows.append(rec)

print(f"{len(rows)} interfaces, pose-pose filter down the ipTM ranking\n")
print(f"{'thresh':>7} {'median #kept':>13} {'max #kept':>10} "
      f"{'strict retention':>17} {'within tol ' + str(A.tol):>16} {'#iface <= ' + str(A.target):>14}")
agg = {}
for th in THRESHOLDS:
    k = [r["by_thresh"][str(th)]["n"] for r in rows]
    st_ = sum(r["by_thresh"][str(th)]["strict"] for r in rows)
    tol = sum(r["by_thresh"][str(th)]["within_tol"] for r in rows)
    und = sum(1 for x in k if x <= A.target)
    agg[str(th)] = {"median_kept": int(np.median(k)), "max_kept": int(max(k)),
                    "strict": st_, "within_tol": tol, "n_under_target": und, "n": len(rows)}
    print(f"{th:>7.2f} {int(np.median(k)):>13} {max(k):>10} "
          f"{str(st_) + '/' + str(len(rows)):>17} {str(tol) + '/' + str(len(rows)):>16} "
          f"{str(und) + '/' + str(len(rows)):>14}")

print(f"\nInterfaces where the STRICT best pose is lost, per threshold:")
for th in THRESHOLDS:
    lost = [r["iface"] for r in rows if not r["by_thresh"][str(th)]["strict"]]
    if lost:
        print(f"  {th:.2f}: {len(lost)} lost -> {', '.join(lost[:8])}{' ...' if len(lost) > 8 else ''}")
    else:
        print(f"  {th:.2f}: none")

json.dump({"tol": A.tol, "target": A.target, "thresholds": THRESHOLDS,
           "aggregate": agg, "per_iface": per_iface},
          open(f"{EXP}/artifacts/redundancy_retention.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/redundancy_retention.json")
