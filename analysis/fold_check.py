"""Are the antibody and antigen monomers folded CORRECTLY (vs ground truth)?

Distinct from orientation: superpose each predicted chain ALONE onto the GT chain
and report RMSD. Residues are matched GT<->prediction by the dominant integer
numbering offset on matching residue types (handles construct renumbering / partial
GT). Reported separately for good vs bad seeds, so we can see whether bad seeds
misfold or only mis-orient.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_7txw_poses import kabsch  # noqa: E402
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, TARGETS_CSV, CSV_DIR, GROUND_TRUTHS  # noqa: E402

DOCKQ = CSV_DIR / "dockq_ost_esmfold2fast_tts_all_samples.csv"
RUN = "esmfold2fast_tts"
GOOD_T, BAD_T = 0.5, 0.23
INTERFACES = [
    ("7txw", "A", "C"), ("8vui", "B", "C"), ("7yxu", "A", "C"),
    ("8tmz", "B", "C"), ("9au1", "C", "E"), ("8ido", "A", "D"), ("8fdd", "B", "C"),
]
T3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
      "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
      "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
      "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U"}


def parse_seq_ca(path):
    """{chain: {resid: (aa, xyz)}} for CA atoms (label chain, auth_seq_id)."""
    cols, idx, inl, out = {}, 0, False, {}
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = idx; idx += 1; inl = True; continue
        if inl and s.startswith(("ATOM", "HETATM")):
            p = s.split()
            if p[cols["label_atom_id"]] != "CA":
                continue
            try:
                rid = int(p[cols["auth_seq_id"]])
            except ValueError:
                continue
            ch = p[cols["label_asym_id"]]
            aa = T3.get(p[cols["label_comp_id"]], "X")
            xyz = np.array([float(p[cols["Cartn_x"]]), float(p[cols["Cartn_y"]]),
                            float(p[cols["Cartn_z"]])])
            out.setdefault(ch, {})[rid] = (aa, xyz)
        elif inl and s.startswith("#"):
            inl = False
    return out


def offset(gt_ch, pr_ch):
    """Dominant integer d with prediction_resid = gt_resid + d (by matching AA)."""
    c = Counter()
    for gr, (ga, _) in gt_ch.items():
        for pr, (pa, _) in pr_ch.items():
            if ga == pa:
                c[pr - gr] += 1
    return c.most_common(1)[0][0] if c else 0


def fold_rmsd(gt_ch, pr_ch, d):
    keys = [r for r in gt_ch if (r + d) in pr_ch and gt_ch[r][0] == pr_ch[r + d][0]]
    if len(keys) < 3:
        return float("nan"), 0
    P = np.array([pr_ch[r + d][1] for r in keys])
    Q = np.array([gt_ch[r][1] for r in keys])
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean())), len(keys)


def best_dockq(pdb, c1, c2):
    out = {}
    with open(DOCKQ) as f:
        for r in csv.DictReader(f):
            if r["assembly_id"].split("-")[0] != pdb or {r["chain1"], r["chain2"]} != {c1, c2}:
                continue
            try:
                dq = float(r["dockq"])
            except ValueError:
                continue
            out[r["seed"]] = max(out.get(r["seed"], 0.0), dq)
    return out


def targets_map():
    out = {}
    with open(TARGETS_CSV) as f:
        for r in csv.DictReader(f):
            out[(r["pdb_id"].split("-")[0],
                 frozenset({r["interface_chain_id_1"], r["interface_chain_id_2"]}))] = \
                (r["antibody_chain"], r["antigen_chain"])
    return out


def main():
    tgt = targets_map()
    print(f"monomer fold RMSD vs GROUND TRUTH (Angstrom), median over seeds\n")
    print(f"{'interface':14s} {'antibody (good/bad)':>22s} {'antigen (good/bad)':>22s}  notes")
    print("-" * 78)
    for pdb, c1, c2 in INTERFACES:
        ab_chain, ag_chain = tgt[(pdb, frozenset({c1, c2}))]
        ab_iface = c1 if c2 == ag_chain else c2
        gt = parse_seq_ca(GROUND_TRUTHS / f"{pdb}-assembly1.cif")
        dq = best_dockq(pdb, c1, c2)
        good = [int(s) for s, v in dq.items() if s != "baseline_seed" and v >= GOOD_T]
        bad = [int(s) for s, v in dq.items() if s != "baseline_seed" and v < BAD_T]
        pred_dir = ESM_PREDICTIONS / RUN / pdb
        if ab_iface not in gt or ag_chain not in gt:
            print(f"{pdb} {c1}|{c2}: GT missing chain"); continue
        # offset from one prediction
        any_pred = next(pred_dir.glob("seed_*/predictions/*_sample_0.cif"))
        pr0 = parse_seq_ca(any_pred)
        d_ab = offset(gt[ab_iface], pr0[ab_iface])
        d_ag = offset(gt[ag_chain], pr0[ag_chain])

        ab_g, ab_b, ag_g, ag_b, nmatch_ag = [], [], [], [], 0
        for s in good + bad:
            f = pred_dir / f"seed_{s}" / "predictions" / f"{pdb}_sample_0.cif"
            if not f.exists():
                continue
            pr = parse_seq_ca(f)
            r_ab, _ = fold_rmsd(gt[ab_iface], pr[ab_iface], d_ab)
            r_ag, n_ag = fold_rmsd(gt[ag_chain], pr[ag_chain], d_ag)
            nmatch_ag = max(nmatch_ag, n_ag)
            (ab_g if s in good else ab_b).append(r_ab)
            (ag_g if s in good else ag_b).append(r_ag)
        m = lambda x: np.nanmedian(x) if x else float("nan")
        ag_n = len(gt[ag_chain])
        note = "PEPTIDE" if ag_n < 30 else ("multidomain?" if m(ag_g) > 2.5 else "")
        print(f"{pdb} {c1}|{c2:9s} {m(ab_g):8.2f} / {m(ab_b):<11.2f} "
              f"{m(ag_g):8.2f} / {m(ag_b):<11.2f}  ag={ag_n}res match={nmatch_ag} {note}")


if __name__ == "__main__":
    main()
