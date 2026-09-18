#!/usr/bin/env python
"""LABEL-FREE within-seed pose diversity, and whether ranking seeds by it finds the acceptable seeds.

THE IDEA. Ranking seeds by DockQ-to-native variance finds an acceptable seed in the top 3 on 100% of
confident-decoy interfaces -- but DockQ-to-native IS the label, so that number is unusable and its very
perfection is the tell. This computes the same notion of "the five poses disagree" WITHOUT the native:
DockQ of each pose against each OTHER pose of the same seed. Nothing here touches the reference
structure, so the ranking is computable at inference time.

  diversity(seed) = 1 - mean(off-diagonal pose-pose DockQ)   in [0, 1], higher = poses disagree more

The native DockQ is still read, but ONLY to label a seed acceptable/not when scoring the ranking. That
is evaluation, not selection -- the ranking never sees it.

CHAIN LETTERS. The kernel needs, PER POSE, the MODEL chain pair for this interface. PXMeter keys its
interface on REFERENCE chains and gives ref_to_model_chain_mapping separately, and on multi-copy
assemblies that mapping changes from pose to pose -- so it is read per pose. Using the reference letters
on a model file scores a different pair of molecules, which is the bug that once put 45% of this
project's interface measurements on the wrong chains.

Usage: pose_diversity.py [--device cuda:0] [--limit N] [--out <csv>]
"""
import argparse
import collections
import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
import numpy as np
from dockq_torch import dockq_matrix

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
AP = 0.23


def model_chains(pdb, seed, smp, c1, c2):
    """Per-pose ref->model chain pair, or None if this pose has no usable mapping."""
    f = f"{P}/pxm_json/{pdb}_{seed}_{smp}.json"
    try:
        m = json.load(open(f)).get("ref_to_model_chain_mapping") or {}
        return (m[c1], m[c2])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="first N interfaces, for a smoke test")
    # --shard I/N splits the interface list across processes so two GPUs can share the work. The
    # ranking table is NOT printed for a shard: it must be computed once over the merged rows, or each
    # shard would report a table over its own subset and the two would look like disagreeing results.
    ap.add_argument("--shard", default=None, help="I/N -- take every Nth interface starting at I")
    ap.add_argument("--out", default=f"{EXP}/artifacts/pose_diversity_cd.csv")
    # decoy      = pooled-ipTM top pose is below 0.23 while an acceptable pose exists (confident decoys)
    # acceptable = pooled-ipTM top pose IS acceptable -- the interfaces where selection already works,
    #              i.e. the contrast group for asking whether confident seeds are unusually consistent
    #              everywhere or only where confidence fails
    ap.add_argument("--set", dest="iset", choices=("decoy", "acceptable", "all"), default="decoy")
    A = ap.parse_args()

    # --- the confident-decoy set, defined exactly as elsewhere: pooled-ipTM top pose is a decoy while
    #     an acceptable pose exists somewhere in the pool.
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        try:
            pool[(r["pdb"], r["chain1"], r["chain2"])][r["seed"]].append(
                (int(r["sample"]), float(r["dockq"]), float(r["pair_iptm"])))
        except (TypeError, ValueError):
            continue
    cd = []
    for key, seeds in pool.items():
        allp = [p for v in seeds.values() for p in v]
        if len(allp) < 25:
            continue
        sel_ok = max(allp, key=lambda x: x[2])[1] >= AP    # does pooled ipTM pick an acceptable pose?
        has_good = max(allp, key=lambda x: x[1])[1] >= AP
        if A.iset == "decoy" and (sel_ok or not has_good):
            continue
        if A.iset == "acceptable" and not sel_ok:
            continue
        cd.append(key)
    cd.sort()
    if A.limit:
        cd = cd[:A.limit]
    sharded = False
    if A.shard:
        i, n = (int(x) for x in A.shard.split("/"))
        cd = [c for j, c in enumerate(cd) if j % n == i]
        sharded = True
    print(f"  {len(cd)} confident-decoy interfaces", flush=True)

    rows = []
    for n, (pdb, c1, c2) in enumerate(cd, 1):
        iface = f"{pdb}_{c1}{c2}"
        for seed, v in sorted(pool[(pdb, c1, c2)].items()):
            items, gts = [], []
            for smp, gt, ip in sorted(v):
                mc = model_chains(pdb, seed, smp, c1, c2)
                cif = f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{smp}.cif"
                if mc is None or not os.path.exists(cif):
                    continue
                items.append((cif, mc[0], mc[1])); gts.append(gt)
            if len(items) < 3:          # need at least 3 poses for a meaningful spread
                continue
            try:
                M = dockq_matrix(items, device=A.device, verbose=False)
            except Exception as e:
                print(f"    {iface} seed {seed}: kernel failed -- {type(e).__name__}: {e}", flush=True)
                continue
            M = np.asarray(M, dtype=float)
            off = M[~np.eye(len(items), dtype=bool)]
            rows.append(dict(iface=iface, seed=seed, n_poses=len(items),
                             diversity=1.0 - float(np.mean(off)),
                             pp_min=float(np.min(off)),
                             gt_max=max(gts), gt_var=st.pvariance(gts) if len(gts) > 1 else 0.0,
                             acceptable=int(max(gts) >= AP)))
        print(f"  [{n}/{len(cd)}] {iface}: {sum(1 for r in rows if r['iface']==iface)} seeds", flush=True)

    if not rows:
        raise SystemExit("no seed-rows computed -- check CUDA_VISIBLE_DEVICES (an empty value hides "
                         "every GPU from torch) and that pxm_json mappings exist")
    with open(A.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {A.out} ({len(rows)} seed-rows)")

    if sharded:
        print("  shard complete -- run report_diversity on the MERGED csvs for the ranking table")
        return

    # --- does the LABEL-FREE ranking find an acceptable seed in the top k?
    by = collections.defaultdict(list)
    for r in rows:
        by[r["iface"]].append(r)
    KS = [3, 5, 10, 20]
    for name, keyf in [("pose-pose diversity (LABEL-FREE)", lambda r: -r["diversity"]),
                       ("DockQ-to-native variance (ORACLE)", lambda r: -r["gt_var"])]:
        hits = {k: 0 for k in KS}; rand = {k: 0.0 for k in KS}; n_if = 0
        for iface, rs in by.items():
            acc = [r for r in rs if r["acceptable"]]
            if not acc or len(rs) < 20:
                continue
            n_if += 1
            order = sorted(rs, key=keyf)
            N, Aa = len(rs), len(acc)
            for k in KS:
                hits[k] += any(r["acceptable"] for r in order[:k])
                p = 1.0
                for i in range(k):
                    p *= max(0, N - Aa - i) / max(1, N - i)
                rand[k] += 1 - p
        print(f"\n  {name}   n={n_if} interfaces")
        for k in KS:
            print(f"    top {k:>2}: {hits[k]/n_if*100:>5.0f}%   random {rand[k]/n_if*100:>5.0f}%   "
                  f"lift {hits[k]/n_if/(rand[k]/n_if):.2f}x")


if __name__ == "__main__":
    main()
