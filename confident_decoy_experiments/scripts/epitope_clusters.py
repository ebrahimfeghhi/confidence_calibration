#!/usr/bin/env python
"""How many DISTINCT antigen surfaces does a complex dock to across its seeds?

WHY: seed scaling clearly works by re-rolling the epitope choice (gain is an inverted U in epitope
findability, peaking at +0.46 where the epitope is findable-but-rare). The open question is the SHAPE
of that search -- does the model sample a handful of discrete candidate surfaces, or scatter broadly?
An earlier attempt binned LRMSD into 2A bins and counted modes; that cannot distinguish one diffuse
placement from several distinct ones, so it settled nothing. This measures contact sets directly.

METHOD (all literature-standard quantities):
  - epitope of a pose = antigen residues with any heavy atom within 5.0 A of any antibody heavy atom
    (the standard CAPRI/interface definition, same cutoff used elsewhere in this project).
  - similarity between two poses = JACCARD index of their epitope residue sets.
  - cluster poses by single-linkage at Jaccard >= 0.5, i.e. two poses are the same "surface" if they
    share at least half their contact residues. Single linkage so a continuously-drifting placement
    stays ONE cluster -- that is the conservative choice against finding spurious discreteness.
  - a cluster is counted if it holds >=5% of the poses, so one-off outliers do not inflate the count.
  - native epitope taken from the GROUND TRUTH the same way; a cluster is "correct" if its consensus
    set has Jaccard >= 0.5 with the native epitope.

CHAIN MATCHING COMES FROM OST -- do not hand-roll it. Every artifacts/abag_all_ost/<pdb>_<seed>_<k>.json
carries `chain_mapping` (model chain -> reference chain), which is the correspondence DockQ was actually
computed with, plus `nnat` (native contact count per interface). Using anything else silently decouples
this analysis from the DockQ it is compared against. A previous sequence-identity matcher here rejected
36 of 161 interfaces as "chain match too weak (1.00/0.21)" / "native epitope too small" while OST had
already mapped them identity-to-identity with nnat = [41, 21, 108].

Usage:
  .../foldbench/bin/python scripts/epitope_clusters.py --complexes 8qf4,8c5h [--jobs 24]
Outputs artifacts/epitope_clusters.csv (per interface) and artifacts/epitope_pose_sets.json.gz
"""
import argparse
import collections
import csv
import gzip
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iface_chains import first_pose_with, resolve  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
GT_DIR = "/home/ebrahim/FoldBench/examples/ground_truths"
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
FREE_PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
CUTOFF = 5.0


def _ost():
    import ost
    from ost import io, mol
    return ost, io, mol


def contact_residues(ent, ab_chain, ag_chain, cutoff=CUTOFF):
    """Antigen residue numbers within `cutoff` of the antibody chain (heavy atoms, no hydrogens)."""
    _, _, mol = _ost()
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


def seq_of(ent, ch):
    _T = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
    c = ent.FindChain(ch)
    if not c.IsValid():
        return ""
    return "".join(_T.get(r.GetName().upper(), "") for r in c.residues)


def ost_chain_map(pdb):
    """model chain -> reference chain, from OST's own DockQ output.

    NOTE THE DIRECTION. OST's `chain_mapping` is REFERENCE -> MODEL, which is the opposite of what
    callers here want, and reading it the wrong way round manufactured 16 phantom "disagreements"
    with the sequence-identity matcher. Confirmed against `dockq_interfaces`, whose entries are
    [ref1, ref2, mdl1, mdl2]: for 8gy5, chain_mapping {'A':'B',...} and interface ['A','B','B','D']
    together mean reference A <-> model B. So invert it.
    """
    import glob as _g
    for f in sorted(_g.glob(f"{EXP}/artifacts/abag_all_ost/{pdb}_*.json"))[:5]:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cm = d.get("chain_mapping")
        if cm:
            return {v: k for k, v in cm.items()}
    return {}


def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pose_epitope(arg):
    cif, pdb, seed, smp, ab, ag = arg
    try:
        _, io, _ = _ost()
        ch = resolve(pdb, seed, smp, ab, ag)   # REFERENCE labels in, THIS pose's model chains out
        e = io.LoadMMCIF(cif, fault_tolerant=True)
        return sorted(contact_residues(e, ch["mdl_ab"], ch["mdl_ag"]))
    except Exception:
        return None


def cluster(sets, thr=0.5, min_frac=0.05):
    """Single-linkage clustering on Jaccard >= thr. Conservative: a drifting placement stays ONE
    cluster, so this under-counts rather than over-counts discreteness."""
    n = len(sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if not sets[i]:
            continue
        for j in range(i + 1, n):
            if sets[j] and jac(sets[i], sets[j]) >= thr:
                union(i, j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    big = [g for g in groups.values() if len(g) / n >= min_frac]
    return groups, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--max-poses", type=int, default=0)
    A = ap.parse_args()

    ifm = collections.defaultdict(list)
    for r in csv.DictReader(open(IFACE_CSV)):
        ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))

    out_rows, pose_sets = [], {}
    for pdb in A.complexes.split(","):
        gt = f"{GT_DIR}/{pdb}-assembly1.cif"
        root = f"{FREE_PRED}/{pdb}"
        if not (os.path.exists(gt) and os.path.isdir(root)):
            continue
        poses = []
        for sd in sorted(os.listdir(root)):
            m = re.match(r"seed_(\d+)$", sd)
            if not m or not (101 <= int(m.group(1)) <= 150):
                continue
            pd = f"{root}/{sd}/predictions"
            if os.path.isdir(pd):
                for f in sorted(os.listdir(pd)):
                    if f.endswith(".cif") and "_sample_" in f:
                        poses.append((int(m.group(1)),
                                      int(f.rsplit("_sample_", 1)[1].split(".")[0]), f"{pd}/{f}"))
        if A.max_poses:
            poses = poses[:A.max_poses]
        if len(poses) < 50:
            continue
        _, io, _ = _ost()
        ref = io.LoadMMCIF(gt, fault_tolerant=True)
        mdl0 = io.LoadMMCIF(poses[0][2], fault_tolerant=True)

        for ab, ag in ifm.get(pdb, []):
            # OST's own mapping, read from any cached json for this complex (see module docstring)
            def match(mc):
                # ALIGN, do not compare position-by-position: the ground truth usually starts a few
                # residues in (disordered termini), so a positional comparison of two identical
                # chains scores ~0.05 and every chain looks unmatched.
                from Bio import Align
                al = Align.PairwiseAligner(mode="global", open_gap_score=-10,
                                           extend_gap_score=-0.5, match_score=2, mismatch_score=-1)
                ms = seq_of(mdl0, mc)
                best, bid = None, -1.0
                for c in ref.chains:
                    rs = seq_of(ref, c.GetName())
                    if len(rs) < 20 or not ms:
                        continue
                    aln = al.align(ms, rs)[0]
                    same = tot = 0
                    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
                        for k in range(a1 - a0):
                            tot += 1
                            same += (ms[a0 + k] == rs[b0 + k])
                    idn = same / max(1, min(len(ms), len(rs)))
                    if idn > bid:
                        best, bid = c.GetName(), idn
                return best, bid
            # CHAINS FROM OST, BOTH SIDES (scripts/iface_chains.py). `ab`/`ag` are REFERENCE
            # labels, not model chains -- treating them as model chains measured 72 of 161
            # interfaces on the wrong chains. Model chains are resolved PER POSE below, because a
            # multi-copy assembly matches a different copy from one pose to the next.
            try:
                ch0, _, _ = first_pose_with(pdb, ab, ag)
            except KeyError as e:
                print(f"{pdb} {ab}|{ag}: {e} -- skip", flush=True)
                continue
            gab, gag = ch0["ref_ab"], ch0["ref_ag"]
            native = contact_residues(ref, gab, gag)
            if len(native) < 3:
                # The clustering itself needs NO ground truth, so a GT problem must not discard it.
                # Keep the surface counts and mark the correctness columns unavailable.
                print(f"{pdb} {ab}|{ag}: GT epitope unavailable (model {ab},{ag} -> GT {gab},{gag}) "
                      f"-- clustering kept, correctness columns blank", flush=True)
                native = None
            with Pool(A.jobs) as pool:
                res = pool.map(pose_epitope, [(p[2], pdb, p[0], p[1], ab, ag) for p in poses],
                               chunksize=4)
            sets = [set(r) if r else set() for r in res]
            groups, big = cluster(sets)
            empty = sum(1 for s in sets if not s)
            sizes = sorted((len(g) for g in big), reverse=True)
            # sizes can be EMPTY: if every pose lands on a slightly different patch, no cluster
            # reaches the 5% floor. That is itself a finding (a diffuse search), not an error.
            lf = sizes[0] / len(sets) if sizes else 0.0
            # is any big cluster the CORRECT surface?
            corr = 0 if native is not None else ""
            for g in (big if native is not None else []):
                cons = collections.Counter()
                for i in g:
                    cons.update(sets[i])
                keep = {r for r, c in cons.items() if c >= 0.5 * len(g)}
                if jac(keep, native) >= 0.5:
                    corr += 1
            key = f"{pdb}_{ab}{ag}"
            frac_find = (sum(1 for s in sets if jac(s, native) >= 0.5) / len(sets)
                         if native is not None else "")
            out_rows.append([key, len(sets), empty, len(big), len(groups), corr,
                             round(lf, 3),
                             round(frac_find, 3) if native is not None else "",
                             len(native) if native is not None else ""])
            pose_sets[key] = [sorted(s) for s in sets]
            print(f"  {key}: {len(big)} big clusters (of {len(groups)} total), "
                  f"largest {lf:.0%}, correct-surface clusters {corr}, "
                  f"no-contact poses {empty}", flush=True)

    # MERGE, do not truncate: this script is often re-run on a SUBSET of complexes, and opening the
    # csv in "w" mode once wiped 134 previously-computed interfaces down to the 2 just recomputed.
    HDR = ["interface", "n_poses", "n_no_contact", "n_big_clusters", "n_all_clusters",
           "n_correct_clusters", "largest_frac", "frac_find_native", "native_size"]
    out_csv = f"{EXP}/artifacts/epitope_clusters.csv"
    prev = {}
    if os.path.exists(out_csv):
        for r in csv.DictReader(open(out_csv)):
            prev[r["interface"]] = [r[h] for h in HDR]
    for row in out_rows:
        prev[row[0]] = row
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(prev[k] for k in sorted(prev))
    with gzip.open(f"{EXP}/artifacts/epitope_pose_sets.json.gz", "wt") as fh:
        json.dump(pose_sets, fh)
    print(f"\nwrote {len(out_rows)} interfaces -> artifacts/epitope_clusters.csv")


if __name__ == "__main__":
    main()
