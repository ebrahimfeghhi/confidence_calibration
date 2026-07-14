#!/usr/bin/env python3
"""dockq_esm_matched.py — depth-matched (250-pose) ESMFold2 pose-vs-pose DockQ matrices.

WHY NOT REUSE results/dockq_full/?  Because those 47 matrices are exactly the confident-decoy
TRAP interfaces -- the set where ESMFold2 fails BY CONSTRUCTION. Measuring "what is in ESM's top
decile" on a set selected for ESM failing would be rigged. And they are 5125-pose matrices, while
Protenix has 250, so top-decile comparisons would not be like-for-like either.

So: rebuild ESM matrices on the SAME interfaces Protenix has, subsampled to the SAME depth
(50 seeds x 5 samples = 250 poses), with the SAME rng seed as every other depth-matched analysis
tonight (20260713), so the pools are literally the same poses.

TWO-CODE-PATH RULE. Imports dockq_torch's certified dockq_matrix(); adds no DockQ formula.
CHAIN MAPPING. chain1/chain2 are NATIVE letters; `chain_mapping` (ref -> model) translates PER
POSE. Using native letters on a model file scores the wrong molecules (the 17/48 bug).
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dockq_torch import dockq_matrix

REPO = Path(__file__).resolve().parents[2]
ESM_CSV = REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = REPO / "results/csv_files/dockq_ost_protenix_full.csv"
PRED = Path.home() / "esm/foldbench_predictions/esmfold2fast_tts"
OUT = REPO / "results/dockq_esm250"
NSEED = 50


def rows_by_iface():
    by = {}
    for r in csv.DictReader(open(ESM_CSV)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        key = (r["assembly_id"][:4], r["chain1"], r["chain2"])
        by.setdefault(key, []).append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260713)          # SAME seed as every other matched analysis

    esm = rows_by_iface()
    pro = {(r["assembly_id"][:4], r["chain1"], r["chain2"])
           for r in csv.DictReader(open(PRO_CSV))}
    todo = sorted(set(esm) & pro)
    if a.only:
        todo = [k for k in todo if f"{k[0]}_{k[1]}{k[2]}" in a.only]
    print(f"{len(todo)} interfaces present in BOTH models\n", flush=True)

    t0 = time.time(); done = skipped = 0
    for cx, c1, c2 in todo:
        tag = f"{cx}_{c1}{c2}"
        npy = OUT / f"dockq_{tag}.npy"
        if npy.exists():
            skipped += 1
            continue
        rs = esm[(cx, c1, c2)]
        seeds = sorted({r["seed"] for r in rs})
        keep = set(rng.choice(seeds, min(NSEED, len(seeds)), replace=False))
        rows = []
        for r in rs:
            if r["seed"] not in keep:
                continue
            try:
                cm = json.loads(r["chain_mapping"])
                m1, m2 = cm[c1], cm[c2]                       # ref -> model, PER POSE
                s, k = r["seed"], int(r["sample"])
                ip, gt = float(r["pair_iptm"]), float(r["dockq"])
            except Exception:
                continue
            sub = "baseline_seed" if s == "baseline_seed" else f"seed_{s}"
            p = PRED / cx / sub / "predictions" / f"{cx}_sample_{k}.cif"
            if p.exists():
                rows.append((p, m1, m2, ip, gt, s, k))
        if len(rows) < 20:
            print(f"  {tag}: only {len(rows)} poses -- skip", flush=True)
            continue
        rows.sort(key=lambda r: -r[3])
        try:
            D = dockq_matrix([(r[0], r[1], r[2]) for r in rows], device=a.device, verbose=False)
        except ValueError as e:
            # dockq_torch asserts every pose has the same atom count per chain. On some multi-copy
            # assemblies the chain_mapping points at differently-sized copies across poses, so the
            # pool is not a consistent set of molecules. That is a real data problem, not a bug --
            # skip the interface rather than score the wrong thing (this guard is what caught the
            # 17/48 chain-mapping corruption). Log it so the skip is never silent.
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
        if done % 10 == 0:
            print(f"  [{done}] {tag}: {len(rows)} poses  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nbuilt {done}, skipped {skipped}, in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
