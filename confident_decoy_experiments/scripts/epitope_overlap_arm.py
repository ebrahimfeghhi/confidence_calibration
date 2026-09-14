#!/usr/bin/env python
"""Per-pose epitope Jaccard for a TEMPLATE ARM, using the arm's own OST cache for chain mapping.

WHY A THIRD COPY. scripts/epitope_overlap.py reads chain correspondence from
artifacts/abag_all_ost (the free pool) and epitope_overlap_pxm.py reads it from PXMeter's pose jsons.
An arm built by scripts/build_multitmpl.py writes neither -- scripts/score_arm.py leaves its own cache
at artifacts/<arm>_ost/<pdb>_<chains>_<seed>_<sample>.json. Same measurement, third mapping source.

DEFINITIONS, identical to both siblings so the numbers are comparable:
  epitope of a pose = antigen residues with a heavy atom within 5.0 A of the antibody chain
  native epitope    = the same computation on the ground-truth assembly, fixed per interface
  model residue numbers are aligned into GT numbering by sequence alignment of the antigen chain
  jaccard = |pred & GT| / |pred | GT|,  recall = |pred & GT| / |GT|

CHAIN MAPPING IS PER POSE, taken from the `dockq_interfaces` row whose REFERENCE pair matches the
interface being scored. Rows are [ref1, ref2, mdl1, mdl2]. This is the same correspondence DockQ was
computed with, which is the point -- deriving it any other way is how 72 of 161 FoldBench interfaces
once got measured on the wrong chains.

ROLES come from FoldBench's interface_antibody_antigen.csv, whose last two columns are the antibody
and antigen REFERENCE chain names (not model names).

Usage:
  .../foldbench/bin/python scripts/epitope_overlap_arm.py --arm mt8vyl --pdb 8vyl \
      --interfaces 8vyl_FA,8vyl_FC [--jobs 16]
Outputs artifacts/<arm>_epitope.csv: pdb,ab_chain,ag_chain,seed,sample,n_native,n_pred,n_inter,
                                     recall,jaccard
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import _ost, align_map, chain_residues  # noqa: E402
from epitope_clusters import contact_residues  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
TGT = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
COLS = ["pdb", "ab_chain", "ag_chain", "seed", "sample",
        "n_native", "n_pred", "n_inter", "recall", "jaccard"]


def roles_for(pdb):
    """{frozenset(ref pair): (antibody_ref, antigen_ref)} from FoldBench's labels."""
    out = {}
    for r in csv.DictReader(open(TGT)):
        if r["pdb_id"].split("-")[0] != pdb:
            continue
        ab, ag = r["antibody_chain"], r["antigen_chain"]
        out[frozenset((ab, ag))] = (ab, ag)
    return out


def score_pose(a):
    pdb, ab, ag, seed, smp, ostf, cif, native = a
    try:
        _, io, _ = _ost()
        d = json.load(open(ostf))
        rows = d.get("dockq_interfaces") or []
        m_ab = m_ag = None
        for t in rows:
            if len(t) >= 4 and {t[0], t[1]} == {ab, ag}:
                # rows are [ref1, ref2, mdl1, mdl2]; keep the antibody/antigen roles straight
                m_ab, m_ag = (t[2], t[3]) if t[0] == ab else (t[3], t[2])
                break
        if m_ab is None:
            return ["__ERR__", f"{pdb}_{ab}{ag}", f"seed {seed} sample {smp}: pair absent from "
                    f"dockq_interfaces {[(t[0], t[1]) for t in rows]}"]

        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(f"{GT}/{pdb}-assembly1.cif", fault_tolerant=True)
        mres, rres = chain_residues(mdl, m_ag), chain_residues(ref, ag)
        if not mres or not rres:
            return ["__ERR__", f"{pdb}_{ab}{ag}",
                    f"empty chain (model {len(mres)}, GT {len(rres)})"]
        amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
        m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
        pred = {m2g[x] for x in contact_residues(mdl, m_ab, m_ag) if x in m2g}
        inter, union = pred & native, pred | native
        return [pdb, ab, ag, seed, smp, len(native), len(pred), len(inter),
                round(len(inter) / len(native), 4),
                round(len(inter) / len(union), 4) if union else 0.0]
    except Exception as e:
        return ["__ERR__", f"{pdb}_{ab}{ag}", f"seed {seed} sample {smp}: {type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--interfaces", required=True, help="comma list, e.g. 8vyl_FA,8vyl_FC")
    ap.add_argument("--jobs", type=int, default=16)
    # The job directory name is <pdb>_<arm> for arms built by build_multitmpl.py, but a stock
    # FoldBench json keeps its own name (just the pdb), so the control arm needs this override.
    ap.add_argument("--name", default=None, help="output job dir name (default <pdb>_<arm>)")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or f"{EXP}/artifacts/{A.arm}_epitope.csv"

    roles = roles_for(A.pdb)
    _, io, _ = _ost()
    ref = io.LoadMMCIF(f"{GT}/{A.pdb}-assembly1.cif", fault_tolerant=True)

    jobs = []
    for lab in A.interfaces.split(","):
        chains = lab.split("_", 1)[1]
        c1, c2 = chains[0], chains[1:]
        key = frozenset((c1, c2))
        if key not in roles:
            sys.exit(f"{lab}: pair {c1}|{c2} not in FoldBench labels for {A.pdb} "
                     f"(have {[tuple(k) for k in roles]})")
        ab, ag = roles[key]
        native = contact_residues(ref, ab, ag)
        if not native:
            sys.exit(f"{lab}: native epitope EMPTY -- jaccard would be 0/0")
        print(f"{lab}: antibody {ab}, antigen {ag}, native epitope {len(native)} residues",
              file=sys.stderr)
        for f in sorted(glob.glob(f"{EXP}/artifacts/{A.arm}_ost/{lab}_*.json")):
            m = re.search(rf"{lab}_(\d+)_(\d+)\.json$", f)
            if not m:
                continue
            seed, smp = int(m.group(1)), int(m.group(2))
            # the arm's poses live under one job directory, named <pdb>_<arm>
            name = A.name or f"{A.pdb}_{A.arm}"
            cif = (f"{EXP}/artifacts/sweep_out/{A.arm}/{name}/seed_{seed}/predictions/"
                   f"{name}_sample_{smp}.cif")
            if os.path.exists(cif):
                jobs.append((A.pdb, ab, ag, seed, smp, f, cif, native))

    print(f"{len(jobs)} poses on {A.jobs} workers", file=sys.stderr)
    errs, n = collections.Counter(), 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        with Pool(A.jobs) as pool:
            for row in pool.imap_unordered(score_pose, jobs, chunksize=4):
                if row[0] == "__ERR__":
                    errs[row[1]] += 1
                    if sum(errs.values()) <= 5:
                        print(f"  ERR {row[1]}: {row[2]}", file=sys.stderr)
                    continue
                w.writerow(row)
                n += 1
    print(f"wrote {n} poses to {out}")
    if errs:
        print(f"{sum(errs.values())} errors: {dict(errs)}")


if __name__ == "__main__":
    main()
