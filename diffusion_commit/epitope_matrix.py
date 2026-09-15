#!/usr/bin/env python
"""250x250 pose-pose EPITOPE distance matrix per confident-decoy interface. Label-free.

  epitope of a pose   antigen residues with a heavy atom within 5.0 A of any antibody heavy atom
                      (contact_residues reused verbatim from scripts/epitope_clusters.py -- same CAPRI
                       definition, same cutoff, same OST selection, so results stay comparable)
  distance            1 - Jaccard of the two poses' epitope residue sets.  0 = identical surface,
                      1 = no shared contact residue at all.

Nothing here touches the native structure: the matrix is computable at inference. DockQ and epitope
recall are saved alongside ONLY so a plot can be annotated after the fact.

CHAIN MAPPING is read per pose from that pose's own PXMeter json (`ref_to_model_chain_mapping`).
Model chain names are not reference chain names, and on multi-copy assemblies the mapping changes from
pose to pose -- using the reference letters directly would score a different pair of molecules.

Writes artifacts/epitope_matrices/<iface>.npz  with: D (250x250), dockq, recall, iptm, seed, sample.

Usage: epitope_matrix.py [--jobs 24] [--interfaces a,b]
"""
import argparse
import collections
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
CUTOFF = 5.0
AP = 0.23


def _ost():
    import ost
    from ost import io, mol
    return ost, io, mol


def contact_residues(ent, ab_chain, ag_chain, cutoff=CUTOFF):
    """Antigen residue numbers within `cutoff` of the antibody chain (heavy atoms, no hydrogens)."""
    _ost()
    sel_ab = ent.Select(f"chain='{ab_chain}' and ele!=H")
    sel_ag = ent.Select(f"chain='{ag_chain}' and ele!=H")
    if not sel_ab.GetAtomCount() or not sel_ag.GetAtomCount():
        return set()
    ab = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in sel_ab.atoms])
    out = set()
    for r in sel_ag.residues:
        pos = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in r.atoms])
        if pos.size and np.min(np.linalg.norm(pos[:, None, :] - ab[None, :, :], axis=-1)) <= cutoff:
            out.add(r.GetNumber().GetNum())
    return out


def pose_epitope(arg):
    pdb, seed, smp, c_ab, c_ag = arg
    cif = f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{smp}.cif"
    if not os.path.exists(cif):
        return (seed, smp), None
    try:
        _, io, _ = _ost()
        return (seed, smp), contact_residues(io.LoadMMCIF(cif), c_ab, c_ag)
    except Exception:
        return (seed, smp), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--interfaces", default=None)
    ap.add_argument("--outdir", default=f"{EXP}/artifacts/epitope_matrices")
    A = ap.parse_args()
    os.makedirs(A.outdir, exist_ok=True)

    rec, role = {}, {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        fs = frozenset((r["ab_chain"], r["ag_chain"]))
        rec[(r["pdb"], fs, r["seed"], r["sample"])] = float(r["recall"])
        role[(r["pdb"], fs)] = (r["ab_chain"], r["ag_chain"])

    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        try:
            pool[(r["pdb"], r["chain1"], r["chain2"])].append(
                (r["seed"], r["sample"], float(r["dockq"]), float(r["pair_iptm"])))
        except (TypeError, ValueError):
            continue

    want = set(A.interfaces.split(",")) if A.interfaces else None
    todo = []
    for key, v in pool.items():
        if len(v) < 25:
            continue
        if max(v, key=lambda x: x[3])[2] >= AP:      # ipTM already picks an acceptable pose
            continue
        if max(v, key=lambda x: x[2])[2] < AP:       # no acceptable pose exists
            continue
        iface = f"{key[0]}_{key[1]}{key[2]}"
        if want and iface not in want:
            continue
        todo.append((iface, key, v))
    print(f"  {len(todo)} confident-decoy interfaces", flush=True)

    for n, (iface, key, v) in enumerate(todo, 1):
        out = f"{A.outdir}/{iface}.npz"
        if os.path.exists(out):
            print(f"  [{n}/{len(todo)}] {iface}: cached", flush=True); continue
        fs = frozenset((key[1], key[2]))
        if (key[0], fs) not in role:
            print(f"  [{n}/{len(todo)}] {iface}: no chain roles, skipped", flush=True); continue
        ab, ag = role[(key[0], fs)]
        v = sorted(v, key=lambda x: (x[0], int(x[1])))
        work = []
        for seed, smp, _, _ in v:
            try:
                m = json.load(open(f"{P}/pxm_json/{key[0]}_{seed}_{smp}.json"))["ref_to_model_chain_mapping"]
                work.append((key[0], seed, smp, m[ab], m[ag]))
            except Exception:
                work.append((key[0], seed, smp, None, None))
        with Pool(A.jobs) as pl:
            got = dict(pl.map(pose_epitope, [w for w in work if w[3]], chunksize=8))
        sets = [got.get((s, k)) for s, k, _, _ in v]
        ok = [i for i, x in enumerate(sets) if x]
        N = len(v)
        D = np.full((N, N), np.nan)
        for i in ok:
            for j in ok:
                a, b = sets[i], sets[j]
                D[i, j] = 0.0 if (not a and not b) else (1.0 - (len(a & b) / len(a | b) if (a or b) else 1.0))
        np.savez_compressed(out, D=D,
                            dockq=np.array([x[2] for x in v]),
                            iptm=np.array([x[3] for x in v]),
                            recall=np.array([rec.get((key[0], fs, s, k), np.nan) for s, k, _, _ in v]),
                            seed=np.array([x[0] for x in v]), sample=np.array([x[1] for x in v]),
                            n_resolved=len(ok))
        print(f"  [{n}/{len(todo)}] {iface}: {len(ok)}/{N} poses resolved -> {out}", flush=True)


if __name__ == "__main__":
    main()
