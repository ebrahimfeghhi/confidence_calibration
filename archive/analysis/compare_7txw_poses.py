"""Compare 3D predictions for 7txw A|C across "good" vs "bad" seeds.

Question: do the high-DockQ seeds predict a fundamentally different complex
(different antigen pose / epitope) or just a small tweak of the same pose?

numpy-only. Parses mmCIF atom_site loops directly, superposes on the antibody
(chain A) CA atoms, then measures where the antigen (chain C) lands relative to
ground truth and relative to other predictions, plus epitope overlap.
"""
import sys
from pathlib import Path
import numpy as np

PRED = Path("/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts/7txw")
GT = Path("/home/ebrahim/FoldBench/examples/ground_truths/7txw-assembly1.cif")

GOOD = ["seed_10", "seed_14", "seed_19", "seed_33"]
BAD = ["seed_0", "seed_1", "seed_5", "seed_20", "seed_40", "baseline_seed"]


def parse_cif_ca(path):
    """Return {chain: {resid(int): xyz}} for CA atoms, parsing column order from header."""
    cols = {}
    in_loop = False
    idx = 0
    out = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols[s.split(".", 1)[1]] = idx
                idx += 1
                in_loop = True
                continue
            if in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
                p = s.split()
                if p[cols["label_atom_id"]] != "CA":
                    continue
                # label_asym_id is consistent between predictions and GT (A,B,C);
                # auth_seq_id gives construct numbering shared across both.
                chain = p[cols["label_asym_id"]]
                try:
                    resid = int(p[cols["auth_seq_id"]])
                except ValueError:
                    continue
                xyz = np.array([float(p[cols["Cartn_x"]]),
                                float(p[cols["Cartn_y"]]),
                                float(p[cols["Cartn_z"]])])
                out.setdefault(chain, {})[resid] = xyz
            elif in_loop and s.startswith("#"):
                in_loop = False
    return out


def common(a, b, chain):
    keys = sorted(set(a.get(chain, {})) & set(b.get(chain, {})))
    A = np.array([a[chain][k] for k in keys])
    B = np.array([b[chain][k] for k in keys])
    return A, B, keys


def kabsch(P, Q):
    """Rotation+translation aligning P onto Q. Returns (R, t) so that P@R.T+t ~ Q."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def apply(coords_dict, R, t):
    return {c: {k: R @ v + t for k, v in res.items()} for c, res in coords_dict.items()}


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(1).mean()))


def superpose_on_chainA(mob, ref):
    """Superpose mob onto ref using chain A CAs; return transformed mob + chainA rmsd."""
    P, Q, _ = common(mob, ref, "A")
    R, t = kabsch(P, Q)
    return apply(mob, R, t), rmsd(P @ R.T + t, Q)


def epitope(struct, ab="A", ag="C", cutoff=8.0):
    """Antigen residues with any CA within cutoff of an antibody CA."""
    abc = np.array(list(struct[ab].values()))
    ep = set()
    for k, v in struct[ag].items():
        if np.sqrt(((abc - v) ** 2).sum(1)).min() < cutoff:
            ep.add(k)
    return ep


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    gt = parse_cif_ca(GT)
    # Antibody (A,B) numbering matches the predictions exactly, but the GT antigen
    # (C) is numbered 23..108 while predictions number the same residues 1..86.
    # GT seq is an exact substring of the predicted antigen at offset 22, so shift.
    gt["C"] = {k - 22: v for k, v in gt["C"].items()}
    print("GT chains (C remapped to prediction numbering):",
          {c: len(v) for c, v in gt.items()})
    gt_ep = epitope(gt)
    print(f"GT epitope (antigen res within 8A of antibody A): {sorted(gt_ep)}\n")

    # pick representative sample 0 for each seed
    reps = {}
    for seed in GOOD + BAD:
        f = PRED / seed / "predictions" / "7txw_sample_0.cif"
        if f.exists():
            reps[seed] = parse_cif_ca(f)

    print(f"{'seed':14s} {'A-fold':>7s} {'C-fold':>7s} {'C@A vs GT':>10s} {'epi-Jac':>8s} {'epi∩GT':>7s}")
    print("-" * 60)
    for seed, st in reps.items():
        # monomer fold consistency vs GT (superpose each chain alone)
        PA, QA, _ = common(st, gt, "A")
        Ra, ta = kabsch(PA, QA)
        a_fold = rmsd(PA @ Ra.T + ta, QA)
        PC, QC, _ = common(st, gt, "C")
        Rc, tc = kabsch(PC, QC)
        c_fold = rmsd(PC @ Rc.T + tc, QC)
        # docking: superpose on antibody A, measure antigen C displacement vs GT
        st_al, _ = superpose_on_chainA(st, gt)
        CA, CB, _ = common(st_al, gt, "C")
        c_on_a = rmsd(CA, CB)
        ep = epitope(st)
        print(f"{seed:14s} {a_fold:7.2f} {c_fold:7.2f} {c_on_a:10.2f} "
              f"{jaccard(ep, gt_ep):8.2f} {len(ep & gt_ep):3d}/{len(ep):d}")

    # pairwise antigen pose distance (superpose on A, RMSD of C) within/between groups
    print("\nPairwise antigen-C RMSD after antibody(A) superposition:")
    print("(small => same docking pose; large => different complex)\n")
    names = list(reps)
    ref = reps[names[0]]
    print(f"{'':14s}" + "".join(f"{n.replace('seed_','s'):>8s}" for n in names))
    for n1 in names:
        row = f"{n1:14s}"
        for n2 in names:
            st_al, _ = superpose_on_chainA(reps[n1], reps[n2])
            CA, CB, _ = common(st_al, reps[n2], "C")
            row += f"{rmsd(CA, CB):8.1f}"
        print(row)


if __name__ == "__main__":
    main()
