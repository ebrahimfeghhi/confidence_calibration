#!/usr/bin/env python
"""Epitope/paratope of every diffusion sample of ONE seed, against each other and against GT.

QUESTION (8che_BE seed 137): the good pose (sample 3, GT DockQ 0.529) has pose-pose DockQ of only
0.03-0.04 against its four siblings. Is it on a genuinely DIFFERENT antigen surface, or the same
surface with a shifted register?

METHOD. 5 A heavy-atom contacts across the interface, giving two residue sets per pose:
    epitope  = contacted residues on the ANTIGEN chain
    paratope = contacted residues on the ANTIBODY chain
Compared by Jaccard overlap, pose-vs-pose and pose-vs-GT. Jaccard 0 = disjoint surfaces; ~1 = same
residues. A shifted register on the same surface shows up as MODERATE overlap; a different surface
shows up as ~0.

NUMBERING. Chains are keyed by label_asym_id and residues by label_seq_id, mirroring
decoy_set.py:native_chains -- gemmi's reader names chains by auth_asym_id and the two DIFFER for 8che
(also 8bf4/8fbw/8slb/8the), so a hand-rolled gemmi lookup silently compares the wrong chains.

Usage: .../python scripts/epitope_paratope_seed.py --label 8che_BE --seed 137
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
import numpy as np
from decoy_set import EXP, GT, PRED

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="8che_BE")
ap.add_argument("--seed", type=int, default=137)
ap.add_argument("--cutoff", type=float, default=5.0)
A = ap.parse_args()

nat = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
PDB = nat[A.label]["pdb"]
C_AB, C_AG = nat[A.label]["chains"].split("-")   # antibody chain, antigen chain


def chains_of(path):
    """label_asym_id -> (coords Nx3, label_seq_id per atom). Same convention as native_chains."""
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB")]
    D = collections.defaultdict(lambda: ([], []))
    for a, s, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        D[a][0].append([float(x), float(y), float(z)])
        D[a][1].append(s)
    if not D:
        raise ValueError(f"no polymer atoms parsed from {path}")
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in D.items()}


def interface(path, c_ab, c_ag):
    """(paratope, epitope) as sets of label_seq_id. RAISES if a chain is absent."""
    D = chains_of(path)
    for c in (c_ab, c_ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, IA = D[c_ab]
    XB, IB = D[c_ag]
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < A.cutoff
    return set(IA[m.any(1)]), set(IB[m.any(0)])


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


gt_para, gt_epi = interface(f"{GT}/{PDB}-assembly1.cif", C_AB, C_AG)
print(f"{A.label} seed {A.seed}   antibody chain {C_AB}, antigen chain {C_AG}, cutoff {A.cutoff} A")
print(f"GT: paratope {len(gt_para)} res, epitope {len(gt_epi)} res")

# GT DockQ per sample, from the pose-pose index (same source as every other number here)
import csv
gtq, mdl_chains = {}, {}
idx = f"{EXP}/artifacts/pose_pose_dockq/dockq_{A.label}_seeds101-150_index.csv"
if os.path.exists(idx):
    for r in csv.DictReader(open(idx)):
        if int(r["seed"]) == A.seed:
            k = int(r["sample"])
            gtq[k] = float(r["dockq_gt"])
            # model_chain1/2 correspond to GT chains 1/2 in the same order as `chains` above
            mdl_chains[k] = (r["model_chain1"], r["model_chain2"])

poses = {}
for k in range(5):
    cif = f"{PRED}/{PDB}/seed_{A.seed}/predictions/{PDB}_sample_{k}.cif"
    if not os.path.exists(cif):
        print(f"sample {k}: MISSING {cif}")
        continue
    # The MODEL's chain letters are not the GT's. OST maps the native onto whichever model copy fits
    # best and that mapping changes per pose (8che GT B-E maps onto model C-E). Using the GT letters
    # on a prediction silently measures the wrong pair of molecules -- it returned 0 contacts for a
    # pose with GT DockQ 0.529. The per-pose mapping is recorded in the pose_pose_dockq index.
    mc = mdl_chains.get(k)
    if mc is None:
        raise KeyError(f"no model chain mapping for sample {k} in {idx}")
    poses[k] = interface(cif, mc[0], mc[1])

print(f"\n{'sample':>7}{'GT DockQ':>10}{'|para|':>8}{'|epi|':>7}{'epi vs GT':>11}{'para vs GT':>12}")
for k, (p, e) in sorted(poses.items()):
    print(f"{k:>7}{gtq.get(k, float('nan')):>10.3f}{len(p):>8}{len(e):>7}"
          f"{jac(e, gt_epi):>11.3f}{jac(p, gt_para):>12.3f}")

print(f"\npairwise epitope Jaccard (rows/cols = sample):")
ks = sorted(poses)
print("        " + "".join(f"{k:>8}" for k in ks))
for i in ks:
    print(f"{i:>7} " + "".join(f"{jac(poses[i][1], poses[j][1]):>8.3f}" for j in ks))
print(f"\npairwise paratope Jaccard:")
print("        " + "".join(f"{k:>8}" for k in ks))
for i in ks:
    print(f"{i:>7} " + "".join(f"{jac(poses[i][0], poses[j][0]):>8.3f}" for j in ks))

json.dump({"label": A.label, "seed": A.seed,
           "gt": {"paratope": sorted(gt_para), "epitope": sorted(gt_epi)},
           "poses": {str(k): {"paratope": sorted(v[0]), "epitope": sorted(v[1]),
                              "dockq_gt": gtq.get(k)} for k, v in poses.items()}},
          open(f"{EXP}/artifacts/epitope_{A.label}_seed{A.seed}.json", "w"), indent=1)
print(f"\nWROTE artifacts/epitope_{A.label}_seed{A.seed}.json")
