#!/usr/bin/env python
"""Cluster 9x1h's 250 sweep poses by pairwise DockQ, and ask whether the good poses form a cluster.

WHY POSE-POSE DockQ AND NOT EPITOPE. On the 5 samples of seed 300014, epitope Jaccard FAILED to
separate good from bad: the 0.683 pose shared Jaccard 0.50 with a 0.031 failure -- as much as it
shared with the other good pose. Those failures sit on roughly the right antigen patch in the wrong
register, which contact identity cannot see. Pairwise DockQ did separate them: good-good 0.368,
good-bad 0.024, bad-bad 0.038. So orientation is the discriminator, and this scales that to n=250.

THE POINT. If the good poses form a recoverable cluster, cluster-then-rerank is a label-free selector
for this target -- which matters because ipTM is not (the confident decoy scores 0.700 against a good
seed's 0.120). Prior work found this worked on 8ts0 (23% of seeds, 97% purity) but not 8vzo.

PER-POSE CHAIN MAPPING IS MANDATORY. 9x1h is a 2-copy assembly and OST/PXMeter maps the native onto
whichever model copy fits best, POSE BY POSE. Using fixed chain letters would score the wrong pair of
molecules on roughly half the poses. The mapping comes from each pose's own PXMeter json
(ref_to_model_chain_mapping), the same source every other number in this project uses.

The kernel is steer/experiments/dockq_torch.py: exact DockQ 2.1.3, vectorised over all pairs, not an
approximation. Entry [i,j] = DockQ(reference=pose i, model=pose j); it is not symmetric, so the
clustering symmetrises.

Usage: cluster_9x1h_posepose.py [--device cuda:0] [--k 4]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
ACC = 0.23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=4, help="clusters to cut the tree into")
    ap.add_argument("--out", default=f"{EXP}/artifacts/9x1h_posepose.npz")
    A = ap.parse_args()

    # ---- collect poses + their own chain mapping + their true DockQ
    truth = {}
    for r in __import__("csv").DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] == "9x1h" and r["dockq"] and "".join(sorted((r["chain1"], r["chain2"]))) == "CD":
            truth[(int(r["seed"]), int(r["sample"]))] = float(r["dockq"])

    items, keys, native = [], [], []
    for f in sorted(glob.glob(f"{P}/pxm_json/9x1h_*.json")):
        m = re.search(r"9x1h_(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        seed, smp = int(m.group(1)), int(m.group(2))
        try:
            J = json.load(open(f))
        except Exception:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        if "C" not in r2m or "D" not in r2m:
            continue
        cif = f"{P}/out/9x1h/9x1h/seed_{seed}/predictions/9x1h_sample_{smp}.cif"
        if not os.path.exists(cif) or (seed, smp) not in truth:
            continue
        items.append((cif, r2m["C"], r2m["D"]))
        keys.append((seed, smp))
        native.append(truth[(seed, smp)])
    native = np.array(native)
    print(f"poses: {len(items)}   acceptable vs native (>={ACC}): {(native>=ACC).sum()}")
    if len(items) < 10:
        sys.exit("too few poses")

    from dockq_torch import dockq_matrix
    M = dockq_matrix(items, device=A.device, verbose=True)
    M = np.asarray(M.cpu() if hasattr(M, "cpu") else M, dtype=np.float64)
    S = (M + M.T) / 2.0
    np.savez_compressed(A.out, M=M, native=native,
                        keys=np.array(keys), items=np.array([i[0] for i in items]))
    print(f"wrote {A.out}")

    # ---- hierarchical clustering on 1 - DockQ
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Dm = 1.0 - S
    np.fill_diagonal(Dm, 0.0)
    Dm[Dm < 0] = 0.0
    Z = linkage(squareform(Dm, checks=False), method="average")
    lab = fcluster(Z, A.k, criterion="maxclust")

    good = native >= ACC
    print(f"\n{A.k} clusters (average linkage on 1 - pairwise DockQ):")
    print(f"{'cluster':>8} {'size':>5} {'n_good':>7} {'purity':>7} {'mean native DockQ':>18}")
    for c in sorted(set(lab)):
        m = lab == c
        print(f"{c:>8} {m.sum():>5} {int(good[m].sum()):>7} "
              f"{good[m].mean():>7.2f} {native[m].mean():>18.3f}")
    # how well does the biggest good-enriched cluster recover the good poses?
    best = max(set(lab), key=lambda c: good[lab == c].sum())
    m = lab == best
    tp = int(good[m].sum())
    print(f"\nbest good-enriched cluster: id {best}, size {int(m.sum())}")
    print(f"   recall  {tp}/{int(good.sum())} = {tp/max(good.sum(),1):.2f}"
          f"   purity {good[m].mean():.2f}")
    print(f"   -> picking this cluster's centroid pose would give DockQ "
          f"{native[m][np.argmax(S[m][:, m].mean(1))]:.3f}")


if __name__ == "__main__":
    main()
