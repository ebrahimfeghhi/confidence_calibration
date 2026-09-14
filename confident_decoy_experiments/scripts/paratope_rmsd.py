#!/usr/bin/env python
"""PARATOPE RMSD (local and global) -- the antibody-side counterpart of epitope RMSD.

WHY: seed scaling improves the winning pose almost entirely through fnat (r=+0.935, r^2=0.87), i.e.
through contacting more of the right epitope. Before concluding that is a SEARCH effect, rule out the
alternative that the antigen's epitope is simply FOLDED better in the good seeds -- the antigen-side
version of the CDR H3 hypothesis, which we already rejected on the antibody side.

DEFINITIONS, mirroring cdr_h3_rmsd.py so the two sides are directly comparable:
  - paratope = ANTIBODY-chain residues with a heavy atom within 5.0 A of the antigen IN THE GROUND
    TRUTH. Better than CDR H3 RMSD for this purpose: it is defined by actual contacts rather than a
    numbering scheme, it covers every CDR that participates (H3 is only 20-50% of the contacts), and
    it is per-INTERFACE -- so a light-chain interface row is scored on the light chain's own paratope
    instead of being paired with a heavy-chain H3 measurement. A fixed residue set per interface, defined without reference to any prediction -- if it
    were defined from each pose's own contacts, a pose that docked elsewhere would be scored on a
    different patch and the numbers would not be comparable across seeds.
  - LOCAL rmsd  = superpose the epitope Calpha onto their GT counterparts -> patch CONFORMATION only.
  - GLOBAL rmsd = superpose the whole antigen chain, then measure the epitope where it lands ->
    conformation plus placement within the antigen.
  - Calpha only, same as the H3 measurement.

CHAIN MAPPING COMES FROM OST, per pose (`chain_mapping` in artifacts/abag_all_ost/*.json, which is
REFERENCE -> MODEL and must be inverted). Never hand-rolled: in a multi-copy assembly OST re-picks
which copy to align against for each pose, and using a different correspondence than DockQ used would
decouple this measurement from the DockQ it is compared against.

Usage:
  /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/epitope_rmsd.py \
      --complexes 8qf4,8c5h [--jobs 24] [--out artifacts/epitope_rmsd.csv]
Outputs: pdb,ab_chain,ag_chain,seed,sample,n_epitope,epi_local,epi_global
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import (EXP, GT_DIR, IFACE_CSV, _ost, align_map, chain_residues,  # noqa: E402
                         poses_free, superpose_rmsd)
from iface_chains import first_pose_with, resolve  # noqa: E402

CUTOFF = 5.0


def native_paratope(ref, ab_gt, ag_gt, cutoff=CUTOFF):
    """GT ANTIBODY residue numbers within `cutoff` of the GT antigen chain (heavy atoms)."""
    sel_ag = ref.Select(f"chain='{ag_gt}' and ele!=H")   # partner we measure distance TO
    sel_ab = ref.Select(f"chain='{ab_gt}' and ele!=H")   # chain we score
    if not sel_ab.GetAtomCount() or not sel_ag.GetAtomCount():
        return set()
    partner = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in sel_ag.atoms])
    out = set()
    for r in sel_ab.residues:
        pos = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in r.atoms])
        if pos.size and np.min(np.linalg.norm(pos[:, None, :] - partner[None, :, :], axis=-1)) <= cutoff:
            out.add(r.GetNumber().GetNum())
    return out


def score_pose(arg):
    pdb, ab, ag, cif, gt_path, seed, smp, epi_gt_nums = arg
    try:
        _, io, _ = _ost()
        # `ab`/`ag` are REFERENCE chain labels; take BOTH sides from OST's row for THIS pose.
        ch = resolve(pdb, seed, smp, ab, ag)
        m_ab, ab_gt = ch["mdl_ab"], ch["ref_ab"]
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(gt_path, fault_tolerant=True)
        mres = chain_residues(mdl, m_ab)
        rres = chain_residues(ref, ab_gt)
        if not mres or not rres:
            return ["__ERR__", f"{pdb}_{ab}{ag}", f"empty chain (model {len(mres)}, GT {len(rres)})"]
        mseq = "".join(x[1] for x in mres)
        rseq = "".join(x[1] for x in rres)
        amap = align_map(mseq, rseq)               # model index -> GT index
        gt_pos = {rres[j][0]: j for j in range(len(rres))}
        pairs = []
        for i, j in amap.items():
            if rres[j][0] in epi_gt_nums and mres[i][2] is not None and rres[j][2] is not None:
                if mres[i][1] != rres[j][1]:
                    return ["__BAD__", f"{pdb} {ag}: residue mismatch at GT {rres[j][0]}"]
                pairs.append((i, j))
        if len(pairs) < 4:
            return ["__ERR__", f"{pdb}_{ab}{ag}",
                    f"only {len(pairs)} epitope residues mapped into the model"]
        em = np.array([mres[i][2] for i, _ in pairs])
        er = np.array([rres[j][2] for _, j in pairs])
        # whole antibody chain, for the global superposition
        allp = [(i, j) for i, j in amap.items()
                if mres[i][2] is not None and rres[j][2] is not None]
        if len(allp) < 20:
            return ["__ERR__", f"{pdb}_{ab}{ag}", f"antibody chain only {len(allp)} mapped residues"]
        am = np.array([mres[i][2] for i, _ in allp])
        ar = np.array([rres[j][2] for _, j in allp])
        local = superpose_rmsd(em, er)             # epitope onto itself
        glob = superpose_rmsd(am, ar, em, er)      # antigen chain, measure the epitope
        return [pdb, ab, ag, seed, smp, len(pairs), round(local, 3), round(glob, 3)]
    except Exception as e:
        return ["__ERR__", f"{pdb}_{ab}{ag}", f"{type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--out", default=f"{EXP}/artifacts/paratope_rmsd.csv")
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
            try:
                ch0, _, _ = first_pose_with(pdb, ab, ag)
            except KeyError as e:
                print(f"{pdb} {ab}|{ag}: {e} -- skip", flush=True)
                continue
            ab_gt, ag_gt = ch0["ref_ab"], ch0["ref_ag"]
            epi = native_paratope(ref, ab_gt, ag_gt)
            if len(epi) < 4:
                print(f"{pdb} {ab}|{ag}: native paratope only {len(epi)} residues "
                      f"(GT {ab_gt},{ag_gt}) -- skip", flush=True)
                continue
            print(f"{pdb} {ab}|{ag}: native paratope {len(epi)} residues, {len(ps)} poses", flush=True)
            work += [(pdb, ab, ag, p[3], gt, p[1], p[2], epi) for p in ps]

    print(f"\nscoring {len(work)} pose-interfaces, {A.jobs} jobs", flush=True)
    n = 0
    bad = collections.Counter()
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "ab_chain", "ag_chain", "seed", "sample", "n_paratope",
                    "para_local", "para_global"])
        with Pool(A.jobs) as pool:
            for r in pool.imap_unordered(score_pose, work, chunksize=4):
                if r and r[0] == "__BAD__":
                    bad[r[1].split(":")[0]] += 1
                    continue
                if r and r[0] == "__ERR__":
                    bad[f"{r[1]} :: {r[2][:90]}"] += 1
                    continue
                if r:
                    w.writerow(r)
                    n += 1
                    if n % 2000 == 0:
                        fh.flush()
                        print(f"  {n}", flush=True)
    if bad:
        print("\nEXCLUDED (model<->GT residue mismatch):", flush=True)
        for k, v in bad.most_common():
            print(f"  {k}: {v}", flush=True)
    print(f"\nwrote {n} rows -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
