#!/usr/bin/env python
"""How often is an ACCEPTABLE pose in the top-K after greedy structural NMS ("leaders")?

MOTIVATION (user, 2026-08-04): using every free-run pose as its own template is impossible -- 250
templates x 50 seeds per interface. A template scheme needs a SMALL candidate set, say 5 or 10 poses.
So: does a diversity-aware shortlist of the free run contain a correctly-docked pose?

Naive top-K by ipTM is the wrong shortlist because the top poses are usually the same basin repeated --
that is exactly what a confident decoy is. LEADERS applies greedy non-maximum suppression:

    sort poses by interface pair_iptm, descending
    take the highest as leader 1
    suppress every pose in the SAME BASIN as it  (pose-pose DockQ >= 0.23)
    the next unsuppressed pose is leader 2, and so on

so each leader is a structurally distinct binding mode. Same-basin is ALWAYS pose-pose DockQ, never
ligand-RMSD (project convention).

Reports, per interface and in aggregate: is any of the top-K leaders acceptable (DockQ vs native
>= 0.23), versus the naive top-K by ipTM.

DATA: our own matrices, artifacts/pose_pose_dockq/dockq_<interface>_seeds101-150.{npy,index.csv},
built by scripts/pose_pose_dockq.py from OUR predictions and OUR OST scoring.

DO NOT point this at tts_abag/results/dockq_protenix. That directory is HALF-MIGRATED: 11 of our 26
interfaces are on seeds 101-150 there and 15 are on seeds 0-49, with nothing in the filename to say
which, and the two ranges disagree about the answer (8vui_BC: 0 acceptable poses vs 6). Our filenames
carry the seed range for exactly this reason.

The seed-range and acceptable-count checks below are kept as a belt-and-braces guard even though our
own builder already enforces them: mixing seed ranges is how 8ux6 once entered and left the decoy set.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/leaders_topk.py [--ks 1,3,5,10,20]
Writes: artifacts/leaders_topk.json
"""
import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoy_set import EXP, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="1,3,5,10,20")
ap.add_argument("--matrices", default=f"{EXP}/artifacts/pose_pose_dockq")
A = ap.parse_args()
KS = [int(x) for x in A.ks.split(",")]

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
EXCLUDE = {"7z2m_AJ", "8udz_AC"}
DEC = sorted(k for k, v in sc.items()
             if v["free_status"] == "confident_decoy" and k not in EXCLUDE)


def leaders(M, iptm, thresh=THRESH):
    """Greedy NMS on ipTM; a pose is suppressed by a leader if pose-pose DockQ >= thresh."""
    order = np.argsort(-iptm)
    alive = np.ones(len(iptm), bool)
    out = []
    for i in order:
        if not alive[i]:
            continue
        out.append(int(i))
        alive &= ~(M[i] >= thresh)      # suppress its whole basin
        alive[i] = False
    return out


rows, excluded = [], []
for k in DEC:
    npy = f"{A.matrices}/dockq_{k}_seeds101-150.npy"
    idx = f"{A.matrices}/dockq_{k}_seeds101-150_index.csv"
    if not (os.path.exists(npy) and os.path.exists(idx)):
        excluded.append(f"{k}: no matrix"); continue
    r = list(csv.DictReader(open(idx)))
    seeds = [int(x["seed"]) for x in r]
    dq = np.array([float(x["dockq_gt"]) for x in r])
    iptm = np.array([float(x["iptm"]) for x in r])
    n_ok = int((dq >= THRESH).sum())
    if not (min(seeds) >= 101 and max(seeds) <= 150):
        excluded.append(f"{k}: matrix is seeds {min(seeds)}-{max(seeds)}, not 101-150"); continue
    if n_ok != sc[k]["free_n_acceptable"]:
        excluded.append(f"{k}: acceptable {n_ok} != our OST {sc[k]['free_n_acceptable']}"); continue

    M = np.load(npy)
    L = leaders(M, iptm)
    naive = list(np.argsort(-iptm))
    row = {"interface": k, "n_poses": len(r), "n_acceptable": n_ok, "n_leaders": len(L)}
    for K in KS:
        row[f"leaders_top{K}"] = bool((dq[L[:K]] >= THRESH).any())
        row[f"naive_top{K}"] = bool((dq[naive[:K]] >= THRESH).any())
    # rank of the first acceptable pose in each ordering
    row["leader_rank_first_ok"] = next((i + 1 for i, p in enumerate(L) if dq[p] >= THRESH), None)
    row["naive_rank_first_ok"] = next((i + 1 for i, p in enumerate(naive) if dq[p] >= THRESH), None)
    rows.append(row)

out = {"threshold": THRESH, "ks": KS, "n_interfaces": len(rows), "rows": rows, "excluded": excluded}
json.dump(out, open(f"{EXP}/artifacts/leaders_topk.json", "w"), indent=1)

print(f"{'interface':11} {'n_ok':>5} {'#lead':>6} " + " ".join(f"{'L'+str(K):>5}" for K in KS)
      + "   " + " ".join(f"{'N'+str(K):>5}" for K in KS) + f"  {'1st ok (L/N)':>14}")
for r in rows:
    print(f"{r['interface']:11} {r['n_acceptable']:5d} {r['n_leaders']:6d} "
          + " ".join(f"{'Y' if r[f'leaders_top{K}'] else '.':>5}" for K in KS) + "   "
          + " ".join(f"{'Y' if r[f'naive_top{K}'] else '.':>5}" for K in KS)
          + f"  {str(r['leader_rank_first_ok']):>6}/{str(r['naive_rank_first_ok']):<7}")
print(f"\n{'K':>4} {'leaders':>9} {'naive':>9}")
for K in KS:
    l = sum(1 for r in rows if r[f"leaders_top{K}"]); n = sum(1 for r in rows if r[f"naive_top{K}"])
    print(f"{K:4d} {l:6d}/{len(rows)} {n:6d}/{len(rows)}")
if excluded:
    print(f"\nEXCLUDED {len(excluded)} interfaces (matrix unusable):")
    for e in excluded:
        print("  " + e)
print(f"\nWROTE artifacts/leaders_topk.json")
