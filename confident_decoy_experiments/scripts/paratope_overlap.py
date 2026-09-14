#!/usr/bin/env python
"""Per-pose paratope OVERLAP with the ground truth: how much of the true ANTIBODY patch did we use?

WHY THIS IS NOT fnat. fnat is the fraction of native CONTACT PAIRS (antibody residue i, antigen
residue j) recovered, so it scores epitope choice and docking register TOGETHER. A pose that lands on
exactly the right antigen patch but rotated within it keeps a high epitope overlap while its fnat
collapses, because almost no individual pair survives. Separating those two failure modes is the whole
question: does seed scaling FIND the right surface, or REFINE the orientation on a surface it already
had? This is the antibody-side mirror of scripts/paratope_overlap.py: together they separate
"docked the wrong antigen surface" from "used the wrong antibody face", which fnat blends into one
number. A pose can hit the true epitope with the framework instead of the CDRs -- right surface, wrong
face -- and only the paratope side sees that.

TWO MEASURES, same numerator |pred & GT|, different denominator:
  recall  = |pred & GT| / |GT|          -- of the true paratope, how much did we use? Blind to
                                           over-prediction (a pose smearing over the whole antibody
                                           scores 1.0).
  jaccard = |pred & GT| / |pred | GT|   -- adds a penalty for touching surface that is not native,
                                           but is therefore also sensitive to interface SIZE.
Recall is the cleaner answer to "did it use the right face"; jaccard is kept because a large
recall-vs-jaccard gap identifies exactly the poses that are smearing rather than picking.

DEFINITIONS, matching everything else in this project:
  - paratope of a pose = ANTIBODY residues with a heavy atom within 5.0 A of the antigen chain, the
    standard CAPRI interface definition with the roles swapped.
  - native paratope = the same computation on the ground truth. Fixed per interface.
  - model residue numbers are mapped into GT numbering by sequence alignment of the ANTIBODY chain
    before intersecting; the two numbering schemes are not interchangeable.

CHAINS COME FROM scripts/iface_chains.py, PER POSE -- read its docstring before changing anything here.
FoldBench's antibody_chain/antigen_chain labels are REFERENCE chain names, not model ones; a first
version of this script treated them as model chains and so measured 72 of 161 interfaces on the wrong
chains. Both sides now come from the same `dockq_interfaces` row, which is the correspondence DockQ was
actually computed with, re-resolved for every pose (multi-copy assemblies pick a different copy per
pose -- 8gy5 alternates between model D|F and C|E across samples of the same seed).

Usage:
  /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/paratope_overlap.py \
      --complexes 8qf4,8c5h [--jobs 24] [--out artifacts/paratope_overlap.csv]
Outputs: pdb,ab_chain,ag_chain,seed,sample,n_native,n_pred,n_inter,recall,jaccard
"""
import argparse
import collections
import csv
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import (EXP, GT_DIR, IFACE_CSV, _ost, align_map, chain_residues,  # noqa: E402
                         poses_free)
from epitope_clusters import contact_residues  # noqa: E402
from paratope_rmsd import native_paratope  # noqa: E402
from iface_chains import first_pose_with, resolve  # noqa: E402


def score_pose(arg):
    pdb, ab, ag, cif, gt_path, seed, smp, native = arg
    try:
        _, io, _ = _ost()
        # BOTH sides of the correspondence from OST, for THIS pose. `ab`/`ag` are REFERENCE chain
        # names (FoldBench's convention); the model chains are whatever OST matched them to here.
        ch = resolve(pdb, seed, smp, ab, ag)
        m_ab, m_ag, ab_gt = ch["mdl_ab"], ch["mdl_ag"], ch["ref_ab"]
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(gt_path, fault_tolerant=True)
        mres = chain_residues(mdl, m_ab)
        rres = chain_residues(ref, ab_gt)
        if not mres or not rres:
            return ["__ERR__", f"{pdb}_{ab}{ag}", f"empty chain (model {len(mres)}, GT {len(rres)})"]
        amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
        # model residue number -> GT residue number, so the two sets live in one numbering
        m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
        # contact_residues returns residues of its SECOND chain arg -> pass the antibody second
        pred_model = contact_residues(mdl, m_ag, m_ab)
        pred = {m2g[r] for r in pred_model if r in m2g}
        inter = pred & native
        union = pred | native
        recall = len(inter) / len(native)
        jaccard = len(inter) / len(union) if union else 0.0
        return [pdb, ab, ag, seed, smp, len(native), len(pred), len(inter),
                round(recall, 4), round(jaccard, 4)]
    except Exception as e:
        return ["__ERR__", f"{pdb}_{ab}{ag}", f"{type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--out", default=f"{EXP}/artifacts/paratope_overlap.csv")
    A = ap.parse_args()

    ifm = collections.defaultdict(list)
    for r in csv.DictReader(open(IFACE_CSV)):
        ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))

    work = []
    _, io, _ = _ost()
    for pdb in A.complexes.split(","):
        gt = f"{GT_DIR}/{pdb}-assembly1.cif"
        ps = poses_free(pdb)
        if A.max_poses:
            ps = ps[:A.max_poses]
        if not ps or not os.path.exists(gt):
            continue
        ref = io.LoadMMCIF(gt, fault_tolerant=True)
        for ab, ag in ifm.get(pdb, []):
            # GT chain names come from OST too -- they carry a "-2" suffix on duplicated chains, which
            # the FoldBench labels do not, so they are not interchangeable with the labels either.
            try:
                ch0, _, _ = first_pose_with(pdb, ab, ag)
            except KeyError as e:
                print(f"{pdb} {ab}|{ag}: {e} -- skip", flush=True)
                continue
            ab_gt, ag_gt = ch0["ref_ab"], ch0["ref_ag"]
            native = native_paratope(ref, ab_gt, ag_gt)
            if len(native) < 4:
                print(f"{pdb} {ab}|{ag}: native paratope only {len(native)} residues "
                      f"(GT {ab_gt},{ag_gt}) -- skip", flush=True)
                continue
            print(f"{pdb} {ab}|{ag}: native paratope {len(native)} residues (GT {ab_gt},{ag_gt}), "
                  f"{len(ps)} poses", flush=True)
            work += [(pdb, ab, ag, p[3], gt, p[1], p[2], native) for p in ps]

    print(f"\nscoring {len(work)} pose-interfaces, {A.jobs} jobs", flush=True)
    n = 0
    bad = collections.Counter()
    HDR = ["pdb", "ab_chain", "ag_chain", "seed", "sample", "n_native", "n_pred", "n_inter",
           "recall", "jaccard"]
    # MERGE, do not truncate: this script is routinely re-run on a subset of complexes.
    prev = {}
    if os.path.exists(A.out):
        for r in csv.DictReader(open(A.out)):
            prev[(r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"])] = \
                [r[h] for h in HDR]
    with Pool(A.jobs) as pool:
        for r in pool.imap_unordered(score_pose, work, chunksize=4):
            if r and r[0] == "__ERR__":
                bad[f"{r[1]} :: {r[2][:90]}"] += 1
                continue
            if r:
                prev[(r[0], r[1], r[2], str(r[3]), str(r[4]))] = r
                n += 1
                if n % 5000 == 0:
                    print(f"  {n}", flush=True)
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(prev[k] for k in sorted(prev))
    if bad:
        print("\nEXCLUDED:", flush=True)
        for k, v in bad.most_common():
            print(f"  {k}: {v}", flush=True)
    print(f"\nscored {n} rows, wrote {len(prev)} total -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
