#!/usr/bin/env python
"""Is the epitope search DISCRETE or CONTINUOUS? Look at the raw pairwise distances, not a partition.

For ONE interface: compute each pose's epitope (antigen residues within 5 A of the antibody, in GT
numbering), then the 250x250 Jaccard DISTANCE between every pair of poses. Reorder rows/columns by
average linkage purely so similar poses sit next to each other, and plot the matrix.

  crisp blocks with a sharp edge -> a few discrete modes
  a smooth gradient, no boundary -> a continuum

WHY NOT JUST CLUSTER. An earlier version of this question ran single-linkage at Jaccard>=0.5 and
reported the number of clusters. Single linkage merges two groups whenever ANY pair across them is
close, so a chain of gradually drifting intermediates collapses into one cluster -- a true continuum
and a single tight mode both come out as "1 cluster". That analysis could not distinguish the two
hypotheses and was retracted. Here the linkage only ORDERS the display; the evidence is the distances.

Also emits an MDS embedding coloured by DockQ (are the modes separated, and is one of them correct?)
and a silhouette-vs-k curve (a peak supports discreteness, a flat/declining curve supports a continuum).

Usage: .../foldbench/bin/python scripts/epitope_distmat.py --interface 8aon_CA
"""
import argparse
import collections
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import GT_DIR, _ost, align_map, chain_residues, poses_free  # noqa: E402
from epitope_clusters import contact_residues  # noqa: E402
from epitope_rmsd import native_epitope  # noqa: E402
from iface_chains import first_pose_with, resolve  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def pose_epitope(arg):
    """Predicted epitope of ONE pose, mapped into GT residue numbering."""
    pdb, ab, ag, seed, smp, cif, gt = arg
    try:
        _, io, _ = _ost()
        ch = resolve(pdb, seed, smp, ab, ag)
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(gt, fault_tolerant=True)
        mres = chain_residues(mdl, ch["mdl_ag"])
        rres = chain_residues(ref, ch["ref_ag"])
        amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
        m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
        pred = {m2g[r] for r in contact_residues(mdl, ch["mdl_ab"], ch["mdl_ag"]) if r in m2g}
        return (seed, smp, sorted(pred), ch["dockq"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface")
    ap.add_argument("--all", action="store_true",
                    help="every interface in artifacts/abag_selection_protocol.csv")
    ap.add_argument("--jobs", type=int, default=24)
    A = ap.parse_args()
    if A.all:
        import csv as _csv
        todo = [r["interface"] for r in
                _csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv"))]
        for i, iface in enumerate(todo, 1):
            out = f"{EXP}/artifacts/distmat/epitope_distmat_{iface}.npz"
            if os.path.exists(out):
                continue
            try:
                run_one(iface, A.jobs, out)
                print(f"[{i}/{len(todo)}] {iface}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {iface} FAILED: {type(e).__name__}: {e}", flush=True)
        return
    run_one(A.interface, A.jobs,
            f"{EXP}/artifacts/distmat/epitope_distmat_{A.interface}.npz")


def run_one(interface, jobs, out_npz):
    A = argparse.Namespace(interface=interface, jobs=jobs)
    pdb, ch2 = A.interface.rsplit("_", 1)
    ab, ag = ch2[0], ch2[1]
    gt = f"{GT_DIR}/{pdb}-assembly1.cif"

    work = [(pdb, ab, ag, str(s), str(k), cif, gt) for _, s, k, cif in poses_free(pdb)
            if 101 <= s <= 150]
    with Pool(A.jobs) as p:
        res = [r for r in p.map(pose_epitope, work, chunksize=4) if r]
    sets = [set(r[2]) for r in res]
    dq = np.array([r[3] for r in res])
    print(f"{A.interface}: {len(res)} poses, DockQ {dq.min():.3f}-{dq.max():.3f}", flush=True)

    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            D[i, j] = D[j, i] = 1.0 - (len(a & b) / u if u else 1.0)

    _, io, _ = _ost()
    c0, _, _ = first_pose_with(pdb, ab, ag)
    native = native_epitope(io.LoadMMCIF(gt, fault_tolerant=True), c0["ref_ab"], c0["ref_ag"])
    jac_native = np.array([len(s & native) / len(s | native) if (s | native) else 0 for s in sets])

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, D=D, dockq=dq, jac_native=jac_native,
             sets=np.array([",".join(map(str, sorted(s))) for s in sets], dtype=object))


if __name__ == "__main__":
    main()
