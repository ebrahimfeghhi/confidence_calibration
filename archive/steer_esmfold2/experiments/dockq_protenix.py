#!/usr/bin/env python3
"""dockq_protenix.py — build pose-vs-pose DockQ matrices for PROTENIX pools.

WHY. critical_band.py showed ESMFold2's confidence discriminates in bulk (Cliff d +0.404) but
INVERTS inside its own top decile (-0.011), while Protenix's survives there (+0.477). The
question this enables: WHAT IS IN each model's top decile?

The standing hypothesis from this project is that ESM's confidence head measures
SELF-CONSISTENCY, not correctness -- and a rigid, wrong-register dock is maximally
self-consistent. If so, ESM's most-confident poses should all collapse onto ONE tight structural
basin (the confident decoy), whereas Protenix's -- whose head is trained to predict the actual
ERROR of its own rollout samples -- should be structurally diverse at the top.

That is a statement about the STRUCTURAL DIVERSITY of the top decile, and it needs pose-vs-pose
DockQ for Protenix. We already have it for ESM (results/dockq_full/, 47 interfaces).

TWO-CODE-PATH RULE. This script does NOT reimplement DockQ. It imports dockq_torch's certified
dockq_matrix()/load_poses() and merely supplies a Protenix pose list. dockq_torch.py remains the
only file in the repo containing the DockQ formula; OST remains ground truth.

CHAIN MAPPING. Same trap as ESM: chain1/chain2 in the scoring CSV are NATIVE letters, and
`chain_mapping` (ref -> model) translates PER POSE. Using the native letters on a model file
scores the wrong pair of molecules. This is the bug that corrupted 17/48 interfaces once.

Protenix pools are 50 seeds x 5 samples = 250 poses, so each matrix is 250x250 -- trivial next
to ESM's 5125x5125.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dockq_torch import dockq_matrix          # the ONLY DockQ implementation in this repo

REPO = Path(__file__).resolve().parents[2]
PRO_CSV = REPO / "results/csv_files/dockq_ost_protenix_full.csv"
PRED = Path.home() / "Protenix/foldbench_abag/predictions_template"
OUT = REPO / "results/dockq_protenix"


def pose_list(cx, c1, c2):
    """Every diffusion sample of every seed for one Protenix interface, each with ITS OWN model
    chain pair (chain_mapping can differ per pose on multi-copy assemblies).

    Returns [(path, model_c1, model_c2, iptm, dockq_gt, seed, sample), ...], ipTM descending.
    """
    rows = []
    for r in csv.DictReader(open(PRO_CSV)):
        if (r["assembly_id"][:4], r["chain1"], r["chain2"]) != (cx, c1, c2):
            continue
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            cm = json.loads(r["chain_mapping"])
            m1, m2 = cm[c1], cm[c2]                       # ref -> model, PER POSE
            seed, k = r["seed"], int(r["sample"])
            ip, gt = float(r["pair_iptm"]), float(r["dockq"])
        except Exception:
            continue                                      # unusable mapping -> unusable pose
        p = PRED / cx / f"seed_{seed}" / "predictions" / f"{cx}_sample_{k}.cif"
        if p.exists():
            rows.append((p, m1, m2, ip, gt, seed, k))
    rows.sort(key=lambda r: -r[3])
    return rows


def interfaces():
    seen = []
    for r in csv.DictReader(open(PRO_CSV)):
        key = (r["assembly_id"][:4], r["chain1"], r["chain2"])
        if key not in seen:
            seen.append(key)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--only", nargs="*", help="restrict to these tags, e.g. 7zoz_AB")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    todo = interfaces()
    if a.only:
        todo = [k for k in todo if f"{k[0]}_{k[1]}{k[2]}" in a.only]

    t0 = time.time()
    done = skipped = 0
    for cx, c1, c2 in todo:
        tag = f"{cx}_{c1}{c2}"
        npy = OUT / f"dockq_{tag}.npy"
        if npy.exists():
            skipped += 1
            continue
        rows = pose_list(cx, c1, c2)
        if len(rows) < 20:
            print(f"  {tag}: only {len(rows)} poses -- skipping", flush=True)
            continue
        items = [(r[0], r[1], r[2]) for r in rows]
        try:
            D = dockq_matrix(items, device=a.device, verbose=False)
        except ValueError as e:
            # See dockq_esm_matched.py: dockq_torch asserts a consistent atom count per chain
            # across poses. Some multi-copy assemblies map the same chain letter onto
            # differently-sized copies. Skip loudly rather than score the wrong molecules.
            print(f"  {tag}: SKIPPED — {e}", flush=True)
            continue
        D = D.cpu().numpy() if torch.is_tensor(D) else np.asarray(D)
        np.save(npy, D.astype(np.float32))
        with open(OUT / f"dockq_{tag}_index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "seed", "sample", "iptm", "dockq_gt"])
            for i, r in enumerate(rows):
                w.writerow([i, r[5], r[6], f"{r[3]:.4f}", f"{r[4]:.4f}"])
        done += 1
        print(f"  {tag}: {len(rows)} poses -> {D.shape}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\nbuilt {done}, skipped {skipped} (already present), in {time.time()-t0:.0f}s")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
