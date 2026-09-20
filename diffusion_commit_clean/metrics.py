#!/usr/bin/env python
"""Three ways to measure how similar two poses of the same complex are.

WHY THREE. We want to cluster 250 poses into basins and self-template each basin, so the clustering
metric decides what a "basin" is. Three candidates, and the right one is an empirical question:

    epitope   which ANTIGEN RESIDUES the antibody touches            (a patch)
    fcc       which ANTIBODY-ANTIGEN RESIDUE PAIRS are in contact    (a patch AND a register)
    dockq     the CAPRI measure, computed pose against pose          (the field standard)

They differ in a way we have already been bitten by: two poses can touch the same antigen patch from
opposite directions. Epitope overlap calls those identical; FCC does not, because the pairs differ.
Measured earlier on 250 poses, epitope-Jaccard-to-truth in 0.4-0.6 was LESS often acceptable (9/65)
than 0.2-0.3 (15/18), and it traced to one interface whose poses shared the surface with the wrong
register -- exactly the failure epitope-only similarity cannot see.

CUTOFFS follow the conventions we would be compared against:
    epitope   4.5 A heavy atom, the SAbDab/AsEP convention
    fcc       5.0 A heavy atom residue-residue pairs, the CAPRI/HADDOCK convention

FCC IS SYMMETRISED HERE. Rodrigues et al. define it as |A and B| / |A|, normalised by the reference,
so FCC(a,b) != FCC(b,a). Clustering needs a symmetric distance, so this returns the Jaccard form
|A and B| / |A or B| over contact pairs. Noted because the asymmetric number is what the FCC papers
report, and the two are not interchangeable.

SINGLE-COPY COMPLEXES ONLY. On a multi-copy assembly the model builds two copies of the antigen and
different poses bind different copies, so "chain A-C" is not the same interface in every pose (see
dockq_torch.load_poses). Every metric here would then compare different things. Callers must restrict
to single-copy complexes, or resolve the chain pick per pose first.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit/clustering")

from epitope_leader import pose_chains  # noqa: E402  (validated label_asym_id parser)

EPITOPE_CUTOFF = 4.5
CONTACT_CUTOFF = 5.0


def epitope_set(path, ab, ag, cutoff=EPITOPE_CUTOFF):
    """Antigen residues with a heavy atom within cutoff of the antibody."""
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, _ = D[ab]
    XB, IB = D[ag]
    hit = {j for js in cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff) for j in js}
    return {IB[j] for j in hit}


def contact_pairs(path, ab, ag, cutoff=CONTACT_CUTOFF):
    """(antibody residue, antigen residue) pairs in contact -- the patch AND the register."""
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, IA = D[ab]
    XB, IB = D[ag]
    # query_ball_tree gives, for each antibody atom, the antigen atoms within cutoff; keep the
    # RESIDUE pair rather than the atom pair, which is what FCC counts
    near = cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff)
    return {(IA[i], IB[j]) for i, js in enumerate(near) for j in js}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def set_matrix(paths, ab, ag, kind):
    """Symmetric similarity matrix from a per-pose SET -- epitope residues or contact pairs.

    Sets are computed once per pose (N parses), then compared pairwise (N^2/2 cheap set ops). Doing
    it the other way round would re-parse the same file N times.
    """
    fn = epitope_set if kind == "epitope" else contact_pairs
    sets = [fn(p, ab, ag) for p in paths]
    n = len(sets)
    M = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = jaccard(sets[i], sets[j])
    return M, [len(s) for s in sets]


def dockq_pose_matrix(paths, ab, ag, device="cuda"):
    """Pose-against-pose DockQ, symmetrised.

    dockq_matrix returns [i, j] = DockQ(reference=i, model=j), which is not symmetric because Fnat
    divides by the reference's contact count. The average of the two directions is what we cluster on.
    """
    sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit/clustering")
    from dockq_torch import dockq_matrix
    items = [(p, ab, ag) for p in paths]      # single-copy: one chain pair serves every pose
    M = np.asarray(dockq_matrix(items, device=device, verbose=False), dtype=np.float32)
    return (M + M.T) / 2.0


def all_metrics(paths, ab, ag, device="cuda"):
    """{name: NxN similarity} for the three candidates, plus the per-pose set sizes."""
    out, sizes = {}, {}
    for kind in ("epitope", "fcc"):
        out[kind], sizes[kind] = set_matrix(paths, ab, ag, kind)
    out["dockq"] = dockq_pose_matrix(paths, ab, ag, device=device)
    return out, sizes
