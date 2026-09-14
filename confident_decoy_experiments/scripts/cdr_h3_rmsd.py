#!/usr/bin/env python
"""CDR H3 loop RMSD (local and global), following Hitawala & Gray, mAbs 2025.

THEIR METHOD (PMC12360200), which this mirrors:
  - CDR H3 defined by CHOTHIA numbering (they renumbered with the AbNum webserver; we use ANARCI,
    the standard offline equivalent -- boundaries can differ by a residue, so absolute values are
    comparable in spirit, not to the decimal).
  - C-alpha atoms only.
  - LOCAL RMSD: extract the H3 loop as a sub-pose and superpose it ON ITSELF -> measures the loop's
    internal conformation, independent of where the loop sits.
  - GLOBAL RMSD: superpose on the heavy chain, then measure H3 where it lands -> also captures the
    loop's placement/orientation relative to the framework.
  A model can get the loop's shape right but hang it off the framework at the wrong angle; that shows
  up as low local + high global, which is why both are reported.

WHAT WE ARE TESTING: their finding that P(H3 RMSD <= 1A | DockQ >= 0.8) = 86% -- i.e. good docks
almost always have a good H3. We test it on OUR model (Protenix), which is the question that matters
for us: if our confident decoys have a CORRECT H3 and merely dock it to the wrong antigen surface,
then H3 RMSD is useless as a selection signal here.

SUPERPOSITION uses OST's SuperposeSVD rather than a hand-rolled Kabsch -- hand-rolled structural math
has already cost this project a frame-transpose bug.

CHAIN MAPPING COMES FROM OST. Every artifacts/abag_all_ost/<pdb>_<seed>_<k>.json carries
`chain_mapping` (model chain -> reference chain), which is the correspondence DockQ was computed with.
A hand-rolled sequence-identity matcher disagreed with it on 16 of 89 complexes -- all multi-copy
assemblies where it picked a different (identical) copy. The RMSD is probably unchanged by that, but
measuring H3 against a different copy than DockQ used would silently decouple the two numbers we
correlate. So: OST first, sequence identity only as a fallback when no json exists.

RESIDUE MAPPING is the real risk, not the RMSD. The model is built from the full input sequence while
the ground truth usually has gaps, so model and GT residue numbering do NOT necessarily agree. We
align the two sequences explicitly and assert residue-name identity at every mapped position, failing
loudly rather than silently scoring mismatched residues.

Usage:
  .../foldbench/bin/python scripts/cdr_h3_rmsd.py --complexes 8qf4,8c5h [--arm free] [--jobs 8]
Outputs artifacts/cdr_h3_rmsd.csv : pdb,chain,seed,sample,h3_len,h3_local,h3_global
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
GT_DIR = "/home/ebrahim/FoldBench/examples/ground_truths"
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
ANARCI_PY = "/home/ebrahim/miniconda3/envs/anarci/bin/python"
ANARCI_BIN = "/home/ebrahim/miniconda3/envs/anarci/bin/ANARCI"
FREE_PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"

# Chothia CDR H3. AbNum/Chothia H3 runs H95-H102 inclusive; the flanking Cys(H92)/Trp(H103) are the
# conserved anchors and are NOT part of the loop.
H3_LO, H3_HI = 95, 102

_THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}


def _ost():
    import ost
    from ost import io, mol
    return ost, io, mol


def chain_residues(ent, chain_name):
    """[(resnum, one_letter, CA numpy xyz or None)] for a chain, in order."""
    _, _, mol = _ost()
    ch = ent.FindChain(chain_name)
    if not ch.IsValid():
        raise KeyError(f"chain {chain_name!r} not found; have "
                       f"{[c.GetName() for c in ent.chains]}")
    out = []
    for r in ch.residues:
        one = _THREE2ONE.get(r.GetName().upper())
        if one is None:
            continue
        ca = r.FindAtom("CA")
        pos = np.array([ca.GetPos()[0], ca.GetPos()[1], ca.GetPos()[2]]) if ca.IsValid() else None
        out.append((r.GetNumber().GetNum(), one, pos))
    return out


def anarci_h3(seq):
    """Return (chain_type, [0-based indices into seq that are Chothia H3]).

    ANARCI's python API is broken in that env under py3.14 (_domains_are_same compares None), so the
    CLI is invoked with the env's own interpreter.
    """
    with tempfile.TemporaryDirectory() as td:
        fa, out = f"{td}/q.fa", f"{td}/o.txt"
        with open(fa, "w") as fh:
            fh.write(f">q\n{seq}\n")
        subprocess.run([ANARCI_PY, ANARCI_BIN, "-i", fa, "-s", "chothia", "-o", out],
                       capture_output=True, timeout=300,
                       env=dict(os.environ, PATH=f"{os.path.dirname(ANARCI_PY)}:{os.environ['PATH']}"))
        if not os.path.exists(out):
            return None, []
        ctype, idxs, seqstart = None, [], 0
        for line in open(out):
            if line.startswith("#|") and "|" in line[2:] and not line.startswith("#|species"):
                f = [x.strip() for x in line.strip().strip("#|").split("|")]
                # columns: species|chain_type|e-value|score|seqstart_index|seqend_index
                # seqstart is field 4; field 5 is seqEND (reading it put H3 past the sequence end)
                if len(f) >= 6 and f[1] in ("H", "K", "L"):
                    ctype = f[1]
                    seqstart = int(f[4])
            elif re.match(r"^[HKL]\s+\d+", line):
                p = line.split()
                num, aa = int(p[1]), p[-1]
                if aa == "-":
                    continue
                # ANARCI emits numbered residues in sequence order starting at seqstart_index
                idxs.append((num, aa))
        if ctype != "H":
            return ctype, []
        # walk the numbered residues in order, tracking position in the original sequence
        pos = seqstart
        h3 = []
        for num, aa in idxs:
            if H3_LO <= num <= H3_HI:
                h3.append(pos)
            pos += 1
        return ctype, h3


def ost_pose_map(pdb, seed, smp):
    """model chain -> reference chain for ONE pose, from that pose's OST json.

    Per POSE, not per complex: in a multi-copy assembly OST re-picks which copy to align against for
    every pose (8gy5 pose 0 maps A->B, pose 1 maps A->A). Caching one mapping per complex would score
    some poses against the wrong copy -- and against a different copy than DockQ used.
    """
    f = f"{EXP}/artifacts/abag_all_ost/{pdb}_{seed}_{smp}.json"
    if not os.path.exists(f):
        return {}
    with open(f) as fh:
        cm = json.load(fh).get("chain_mapping")
    return {v: k for k, v in cm.items()} if cm else {}


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


def align_map(seq_a, seq_b):
    """Map index-in-a -> index-in-b via global alignment. Identical sequences map 1:1."""
    if seq_a == seq_b:
        return {i: i for i in range(len(seq_a))}
    from Bio import Align
    al = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5,
                               match_score=2, mismatch_score=-1)
    aln = al.align(seq_a, seq_b)[0]
    m = {}
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(a1 - a0):
            m[a0 + k] = b0 + k
    return m


def _rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def superpose_rmsd(mob, ref, mob_eval=None, ref_eval=None):
    """Kabsch via OST SuperposeSVD on (mob->ref); RMSD reported over *_eval (default the same set)."""
    import ost
    from ost import geom
    m = geom.Vec3List([geom.Vec3(*x) for x in mob])
    r = geom.Vec3List([geom.Vec3(*x) for x in ref])
    res = ost.mol.alg.SuperposeSVD(m, r)
    T = np.array(res.transformation.data).reshape(4, 4)
    if mob_eval is None:
        mob_eval, ref_eval = mob, ref
    P = np.asarray(mob_eval)
    P = (T[:3, :3] @ P.T).T + T[:3, 3]
    return _rmsd(P, np.asarray(ref_eval))


def score_pose(args):
    # h3_idx is computed ONCE per complex in main() and passed in: the heavy-chain sequence is
    # identical across a complex's poses, so calling ANARCI per pose would spawn ~12k subprocesses.
    pdb, hchain, cif, gt_path, seed, smp, h3_idx, gt_chain = args
    try:
        _, io, _ = _ost()
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(gt_path, fault_tolerant=True)
        mres = chain_residues(mdl, hchain)
        # The GT chain is resolved by SEQUENCE IDENTITY in main(), not by name: model and ground
        # truth do not use the same chain labels. Model chain E of 8v52 is GT chain D (identity 1.00)
        # while GT chain E is a different molecule entirely (0.32) -- assuming the name matched put
        # an Ala opposite a His and tripped the residue assertion.
        # OST's mapping for THIS pose wins; gt_chain is the fallback for poses with no json
        _pm = ost_pose_map(pdb, seed, smp)
        _gc = _pm.get(hchain, gt_chain)
        try:
            rres = chain_residues(ref, _gc)
        except KeyError:
            rres = chain_residues(ref, gt_chain)
        mseq = "".join(x[1] for x in mres)
        rseq = "".join(x[1] for x in rres)

        if not h3_idx or max(h3_idx) >= len(mres):
            return None
        amap = align_map(mseq, rseq)

        pairs = [(i, amap[i]) for i in h3_idx
                 if i in amap and mres[i][2] is not None and rres[amap[i]][2] is not None]
        if len(pairs) < 4:
            return None
        for i, j in pairs:                      # fail loudly on a bad mapping
            if mres[i][1] != rres[j][1]:
                raise AssertionError(
                    f"{pdb} {hchain}: residue mismatch at model {mres[i][0]}{mres[i][1]} vs "
                    f"GT {rres[j][0]}{rres[j][1]} -- alignment is wrong, not scoring this")
        h3_m = np.array([mres[i][2] for i, _ in pairs])
        h3_r = np.array([rres[j][2] for _, j in pairs])

        # framework = every mapped heavy-chain residue that is NOT in H3
        h3_set = {i for i, _ in pairs}
        fw = [(i, amap[i]) for i in range(len(mres))
              if i not in h3_set and i in amap
              and mres[i][2] is not None and rres[amap[i]][2] is not None]
        if len(fw) < 20:
            return None
        fw_m = np.array([mres[i][2] for i, _ in fw])
        fw_r = np.array([rres[j][2] for _, j in fw])

        local = superpose_rmsd(h3_m, h3_r)                       # loop onto itself
        glob = superpose_rmsd(fw_m, fw_r, h3_m, h3_r)            # framework, measure H3
        return [pdb, hchain, seed, smp, len(pairs), round(local, 3), round(glob, 3)]
    except AssertionError as e:
        # Surface it, but do NOT kill the run: one complex with a bad model<->GT alignment should be
        # excluded and reported, not cost the other 48 their results (8v52 did exactly that once).
        return ["__BAD__", str(e)]
    except Exception:
        return None


def heavy_chain_of(pdb, cif):
    """Which labelled antibody chain is the HEAVY one (ANARCI chain_type == H)."""
    cands = set()
    with open(IFACE_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["pdb_id"].split("-")[0] == pdb:
                cands.add(row["antibody_chain"])
    if not cands:
        raise KeyError(f"no antibody chain for {pdb} in {IFACE_CSV}")
    _, io, _ = _ost()
    ent = io.LoadMMCIF(cif, fault_tolerant=True)
    allc = [c.GetName() for c in ent.chains]
    # labelled antibody chains first, then every other chain: for ~9 complexes the chain named in
    # interface_antibody_antigen.csv is the LIGHT chain, so restricting to it finds no H at all.
    for c in sorted(cands) + [c for c in allc if c not in cands]:
        try:
            seq = "".join(x[1] for x in chain_residues(ent, c))
        except KeyError:
            continue
        if len(seq) < 90:
            continue
        if anarci_h3(seq)[0] == "H":
            return c
    return None


def poses_free(pdb):
    out = []
    root = f"{FREE_PRED}/{pdb}"
    if not os.path.isdir(root):
        return out
    for sd in sorted(os.listdir(root)):
        m = re.match(r"seed_(\d+)$", sd)
        if not m:
            continue
        pd = f"{root}/{sd}/predictions"
        if not os.path.isdir(pd):
            continue
        for f in sorted(os.listdir(pd)):
            if f.endswith(".cif") and "_sample_" in f:
                out.append((pdb, int(m.group(1)),
                            int(f.rsplit("_sample_", 1)[1].split(".")[0]), f"{pd}/{f}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", required=True, help="comma-separated pdb ids")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-poses", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=f"{EXP}/artifacts/cdr_h3_rmsd.csv")
    A = ap.parse_args()

    work, meta = [], {}
    for pdb in A.complexes.split(","):
        ps = poses_free(pdb)
        if A.max_poses:
            ps = ps[:A.max_poses]
        if not ps:
            print(f"{pdb}: no poses", flush=True)
            continue
        gt = f"{GT_DIR}/{pdb}-assembly1.cif"
        if not os.path.exists(gt):
            print(f"{pdb}: no ground truth", flush=True)
            continue
        h = heavy_chain_of(pdb, ps[0][3])
        if h is None:
            print(f"{pdb}: no heavy chain identified by ANARCI -- skipping", flush=True)
            continue
        _, io_, _ = _ost()
        mseq0 = "".join(x[1] for x in chain_residues(io_.LoadMMCIF(ps[0][3], fault_tolerant=True), h))
        _ct, h3i = anarci_h3(mseq0)          # once per complex, not once per pose
        if _ct != "H" or not h3i:
            print(f"{pdb}: ANARCI found no H3 -- skipping", flush=True)
            continue
        meta[pdb] = h
        print(f"{pdb}: heavy chain {h}, H3 len {len(h3i)}, {len(ps)} poses", flush=True)
        # GT chain from OST (fallback: sequence identity) -- see module docstring
        refent = io_.LoadMMCIF(gt, fault_tolerant=True)
        _om = ost_chain_map(pdb)
        if h in _om:
            best, bid, _src = _om[h], 1.0, "OST"
            try:
                _rs = "".join(x[1] for x in chain_residues(refent, best))
                _mp = align_map(mseq0, _rs)
                bid = sum(1 for i, j in _mp.items() if mseq0[i] == _rs[j]) / max(1, len(_mp))
            except Exception:
                bid = 0.0
            if bid >= 0.90:
                print(f"{pdb}: model {h} <-> GT {best} (OST mapping, identity {bid:.2f})", flush=True)
                work += [(pdb, h, cif, gt, seed, smp, h3i, best) for pdb, seed, smp, cif in ps]
                meta[pdb] = h
                continue
            print(f"{pdb}: OST maps {h}->{best} but identity {bid:.2f} -- falling back", flush=True)
        best, bid = None, -1.0
        for c in refent.chains:
            try:
                rs = "".join(x[1] for x in chain_residues(refent, c.GetName()))
            except Exception:
                continue
            if len(rs) < 50:
                continue
            mp = align_map(mseq0, rs)
            idn = sum(1 for i, j in mp.items() if mseq0[i] == rs[j]) / max(1, len(mp))
            if idn > bid:
                best, bid = c.GetName(), idn
        if best is None or bid < 0.90:
            print(f"{pdb}: no GT chain matches model {h} (best {best} id={bid:.2f}) -- skipping", flush=True)
            continue
        print(f"{pdb}: model {h} <-> GT {best} (identity {bid:.2f})", flush=True)
        work += [(pdb, h, cif, gt, seed, smp, h3i, best) for pdb, seed, smp, cif in ps]

    print(f"\nscoring {len(work)} poses, {A.jobs} jobs", flush=True)
    n = 0
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain", "seed", "sample", "h3_len", "h3_local", "h3_global"])
        with Pool(A.jobs) as pool:
            bad = {}
            for r in pool.imap_unordered(score_pose, work, chunksize=4):
                if r and r[0] == "__BAD__":
                    k = r[1].split(":")[0]
                    bad[k] = bad.get(k, 0) + 1
                    continue
                if r:
                    w.writerow(r)
                    n += 1
                    if n % 200 == 0:
                        fh.flush()
                        print(f"  {n}", flush=True)
    if bad:
        print("\nEXCLUDED (model<->GT alignment mismatch, NOT scored):", flush=True)
        for k, v in sorted(bad.items()):
            print(f"  {k}: {v} poses", flush=True)
    print(f"wrote {n} rows -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
