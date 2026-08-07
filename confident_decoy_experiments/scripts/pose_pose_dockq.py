#!/usr/bin/env python
"""Build 250x250 pose-pose DockQ matrices for OUR Protenix free runs at seeds 101-150.

WHY NOT REUSE tts_abag/results/dockq_protenix: that directory is HALF-MIGRATED. 11 of our 26 decoy
interfaces are on seeds 101-150 there; the other 15 are still on seeds 0-49, and nothing in the
filename says which. The two disagree about the answer -- 8vui_BC has 0 acceptable poses at seeds 0-49
and 6 at 101-150; 8ux6_DF is a confident decoy at 0-49 and not at 101-150. Reusing them silently mixes
seed ranges.

WHY NOT REUSE steer/experiments/dockq_torch.py's `build`: its `pose_list()` reads an ESMFold2-fast
scoring CSV and paths under ~/esm/foldbench_predictions/. It enumerates ESM poses, not Protenix ones.
Only its `dockq_matrix()` kernel is predictor-agnostic, so that is what this reuses -- the exact,
vectorised, GPU DockQ (reproduces DockQ 2.1.3, not an approximation).

Poses and per-pose chain mapping come from OUR cached OST runs (artifacts/abag_all_ost), the same
source every other number in this project uses. The per-pose model chain pair matters: on multi-copy
assemblies OST maps the native onto whichever model copy fits best, and that mapping CHANGES PER POSE.
Using the native letters directly would score the wrong pair of molecules.

OUTPUT NAMES CARRY THE SEED RANGE -- `dockq_<interface>_seeds101-150.npy` -- so a matrix can never
again be mistaken for one built on a different range.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/pose_pose_dockq.py [--interfaces a,b] [--device cuda:0]
Writes: artifacts/pose_pose_dockq/dockq_<interface>_seeds101-150.{npy,index.csv}
"""
import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import numpy as np
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--interfaces", default=None, help="comma-separated; default = all 26 decoys")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--outdir", default=f"{EXP}/artifacts/pose_pose_dockq")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)

from dockq_torch import dockq_matrix                     # the predictor-agnostic GPU kernel

CACHE = f"{EXP}/artifacts/abag_all_ost"
sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
EXCLUDE = {"7z2m_AJ", "8udz_AC"}
want = ([x.strip() for x in A.interfaces.split(",")] if A.interfaces else
        sorted(k for k, v in sc.items()
               if v["free_status"] == "confident_decoy" and k not in EXCLUDE))


def poses_for(lab):
    """[(cif, model_c1, model_c2, iptm, dockq_gt, seed, sample)] at seeds 101-150, from our OST cache."""
    v = sc[lab]; pdb = v["pdb"]
    parts = v["chains"].split("-")
    c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])
    out = []
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{CACHE}/{pdb}_{s}_{k}.json"
            cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
            conf = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cif)):
                continue
            try:
                d = json.load(open(o)); M = json.load(open(conf))["chain_pair_iptm"]
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):     # BY NAME
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        out.append((cif, t[2], t[3], M[x][y], qs[i], s, k))   # per-pose model chains
                    break
    return out


for lab in want:
    npy = f"{A.outdir}/dockq_{lab}_seeds101-150.npy"
    idx = f"{A.outdir}/dockq_{lab}_seeds101-150_index.csv"
    if os.path.exists(npy) and os.path.exists(idx):
        print(f"[{lab}] exists, skipping", flush=True); continue
    rows = poses_for(lab)
    if len(rows) < 200:
        print(f"[{lab}] only {len(rows)} usable poses -- SKIP", flush=True); continue

    n_ok = sum(1 for r in rows if r[4] >= THRESH)
    expect = sc[lab]["free_n_acceptable"]
    if n_ok != expect:
        print(f"[{lab}] *** ABORT: {n_ok} acceptable poses but native_arm_scores says {expect} "
              f"-- pose list does not match our scoring", flush=True)
        continue

    print(f"[{lab}] {len(rows)} poses, {n_ok} acceptable, seeds "
          f"{min(r[5] for r in rows)}-{max(r[5] for r in rows)} -> computing "
          f"{len(rows)**2:,} pairs", flush=True)
    M = dockq_matrix([(r[0], r[1], r[2]) for r in rows], device=A.device, verbose=False)
    np.save(npy, M.astype(np.float32))
    with open(idx, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "seed", "sample", "model_chain1", "model_chain2", "iptm", "dockq_gt"])
        for i, (_, m1, m2, ip, gt, s, k) in enumerate(rows):
            w.writerow([i, s, k, m1, m2, f"{ip:.4f}", f"{gt:.4f}"])
    print(f"[{lab}] DONE -> {npy}", flush=True)

print("ALL DONE")
