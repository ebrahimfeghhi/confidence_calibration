#!/usr/bin/env python
"""Paratope / epitope CA-RMSD with per-chain superposition that EXCLUDES the scored region.

Measures intra-chain fold quality decoupled from docking -- i.e. the thing a template still transmits
when multichain_mask is on. Recipe (as specified):

  1. Residue sets are defined ONCE on the NATIVE complex and held fixed for every pose.
     paratope = antibody residues with any heavy atom within 5 A of the antigen (native)
     epitope  = antigen  residues with any heavy atom within 5 A of the antibody (native)
     Re-deriving contacts from each prediction's own geometry would mean a badly-docked pose simply
     has no "epitope" by its own contacts, which biases the comparison.

  2. Superpose PER CHAIN, on everything EXCEPT the region being scored, then measure RMSD over the
     region under that alignment -- the same "don't align on what you're scoring" logic as CDR-H3.
     paratope: superpose antibody chain on non-paratope CAs, score paratope CAs
     epitope:  superpose antigen  chain on non-epitope  CAs, score epitope  CAs
     Antibody-to-antibody and antigen-to-antigen INDEPENDENTLY -- never a whole-complex superposition,
     which would fold the relative docking pose back into the number.

  3. CA atoms only.

CAVEAT (expected, not a bug): these subsets are often 10-25 residues, so RMSD is noisy and one outlier
residue moves it a lot. Per-region lDDT is the planned superposition-free cross-check.

Residue correspondence is by label_seq_id, and only residues present in BOTH native and model are used
(natives have unresolved gaps). Chain roles come from FoldBench's interface_antibody_antigen.csv, and
the MODEL's chain letters come from the per-pose OST mapping -- they differ from the native's.

Usage: .../python scripts/region_rmsd.py --iface 9jbq_AC --poses <cif> [<cif> ...]
"""
import argparse, collections, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
import numpy as np
from decoy_set import EXP, GT

AB_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--iface", required=True, help="e.g. 9jbq_AC")
ap.add_argument("--poses", nargs="+", required=True, help="model cif files")
ap.add_argument("--cutoff", type=float, default=5.0)
ap.add_argument("--selftest", action="store_true", help="validate Kabsch on a randomly rotated native")
A = ap.parse_args()

PDB = A.iface.split("_")[0]


def parse(path):
    """label_asym_id -> {label_seq_id: {atom_name: xyz}}, polymer ATOM records only."""
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z",
             "group_PDB")]
    D = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ch, seq, at, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        D[ch][int(seq)][at.strip('"')] = np.array([float(x), float(y), float(z)])
    return D


def seqs(path):
    """label_asym_id -> {label_seq_id: residue name}, for sequence-based chain matching."""
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "label_comp_id", "group_PDB")]
    S = collections.defaultdict(dict)
    for ch, seq, comp, g in zip(*cols):
        if g == "ATOM":
            S[ch][int(seq)] = comp
    return S


def match_chain(nat_seq, mdl_seqs, exclude=()):
    """Model chain whose RESIDUE NAMES best match the native chain's.

    Matching on residue-number overlap alone is wrong: 9jbq's 213-residue light chain contains every
    residue number the 134-residue antigen has, so it ties and can win, and the epitope then gets
    scored against the wrong molecule (produced a 15.5 A intra-chain RMSD on a chain whose real
    whole-chain RMSD is 0.72 A).
    """
    best, bid = -1.0, None
    for c, s in mdl_seqs.items():
        if c in exclude:
            continue
        sh = [r for r in nat_seq if r in s]
        if not sh:
            continue
        frac = sum(nat_seq[r] == s[r] for r in sh) / len(nat_seq)
        if frac > best:
            best, bid = frac, c
    return bid, best


def kabsch(P, Q):
    """Rotation+translation that best fits P onto Q (both [N,3]). Returns (R, t)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def heavy(res):  # all heavy atoms of a residue (gemmi already excludes H in these files)
    return np.array([v for k, v in res.items()])


# ---- chain roles -------------------------------------------------------------------------------
ab_ch = ag_ch = None
for r in csv.DictReader(open(AB_CSV)):
    if r["pdb_id"] == f"{PDB}-assembly1" and {r["interface_chain_id_1"], r["interface_chain_id_2"]} == \
            set(A.iface.split("_")[1][0]) | set(A.iface.split("_")[1][1]):
        ab_ch, ag_ch = r["antibody_chain"], r["antigen_chain"]
        break
if ab_ch is None:
    sys.exit(f"{A.iface}: no antibody/antigen roles in {AB_CSV}")

# ---- native, and the FIXED region definitions ---------------------------------------------------
NAT = parse(f"{GT}/{PDB}-assembly1.cif")
for c in (ab_ch, ag_ch):
    if c not in NAT:
        sys.exit(f"native has no chain {c} (has {sorted(NAT)})")
para, epi = set(), set()
for ri, ra in NAT[ab_ch].items():
    for rj, rb in NAT[ag_ch].items():
        if np.min(np.linalg.norm(heavy(ra)[:, None, :] - heavy(rb)[None, :, :], axis=-1)) < A.cutoff:
            para.add(ri); epi.add(rj)
print(f"{A.iface}: antibody chain {ab_ch}, antigen chain {ag_ch} (native numbering)")
print(f"  paratope {len(para)} residues, epitope {len(epi)} residues  (native, {A.cutoff} A heavy-atom)")


def region_rmsd(nat_ch, mdl_ch, region):
    """Superpose model onto native using CAs OUTSIDE `region`, then RMSD over CAs INSIDE it."""
    shared = [r for r in nat_ch if r in mdl_ch and "CA" in nat_ch[r] and "CA" in mdl_ch[r]]
    fit = [r for r in shared if r not in region]
    sco = [r for r in shared if r in region]
    if len(fit) < 3 or not sco:
        return None, len(fit), len(sco)
    P = np.array([mdl_ch[r]["CA"] for r in fit]); Q = np.array([nat_ch[r]["CA"] for r in fit])
    R, t = kabsch(P, Q)
    Pm = np.array([mdl_ch[r]["CA"] for r in sco]) @ R.T + t
    Qm = np.array([nat_ch[r]["CA"] for r in sco])
    return float(np.sqrt(((Pm - Qm) ** 2).sum(1).mean())), len(fit), len(sco)


if A.selftest:
    rng = np.random.default_rng(0)
    Rr, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Rr) < 0: Rr[:, 0] *= -1
    rot = {c: {r: {a: Rr @ v + np.array([11., -4., 7.]) for a, v in res.items()}
               for r, res in NAT[c].items()} for c in (ab_ch, ag_ch)}
    v, nf, ns = region_rmsd(NAT[ab_ch], rot[ab_ch], para)
    print(f"  SELFTEST rotated native -> paratope RMSD {v:.6f} A (must be ~0; fit {nf}, scored {ns})")

def whole_rmsd(nat_ch, mdl_ch):
    """Global check: superpose on ALL shared CAs and score ALL of them (spec item 1)."""
    sh = [r for r in nat_ch if r in mdl_ch and "CA" in nat_ch[r] and "CA" in mdl_ch[r]]
    if len(sh) < 3:
        return None
    P = np.array([mdl_ch[r]["CA"] for r in sh]); Q = np.array([nat_ch[r]["CA"] for r in sh])
    R, t = kabsch(P, Q)
    return float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))


print(f"\n{'pose':<30}{'whole_ab':>10}{'whole_ag':>10}{'paratope':>10}{'epitope':>9}{'n_para':>8}{'n_epi':>7}")
NSEQ = seqs(f"{GT}/{PDB}-assembly1.cif")
for cif in A.poses:
    M, MS = parse(cif), seqs(cif)
    cab, fab = match_chain(NSEQ[ab_ch], MS)
    cag, fag = match_chain(NSEQ[ag_ch], MS, exclude=(cab,))
    if min(fab, fag) < 0.9:
        print(f"*** {os.path.basename(cif)}: weak chain match (ab {cab} {fab:.2f}, ag {cag} {fag:.2f}) -- skipped")
        continue
    p, npf, nps = region_rmsd(NAT[ab_ch], M[cab], para)
    e, nef, nes = region_rmsd(NAT[ag_ch], M[cag], epi)
    wa, wg = whole_rmsd(NAT[ab_ch], M[cab]), whole_rmsd(NAT[ag_ch], M[cag])
    f = lambda v: f"{v:.3f}" if v is not None else "  -"
    tag = os.path.basename(os.path.dirname(os.path.dirname(cif))) + "/" + os.path.basename(cif)[:-4]
    print(f"{tag[-29:]:<30}{f(wa):>10}{f(wg):>10}{f(p):>10}{f(e):>9}{nps:>8}{nes:>7}")
