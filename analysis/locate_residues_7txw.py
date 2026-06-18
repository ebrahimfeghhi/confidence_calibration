"""Where are the mask-enriched antigen residues C48-C50 in the native 7txw?

Reports, for the candidate set (C48,C49,C50) and the inert control set
(C65,C70,C98), the minimum heavy-atom distance from each antigen residue to the
antibody (chains A,B) in the ground-truth structure. <8A = interface contact.
Mapping model-0based-pos -> native residue uses the same sequence alignment as
analyze_masks.get_interface_ref, so numbering matches the mask analysis.
"""
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))
from paths import GROUND_TRUTHS
import analyze_masks as am

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.sequence as bseq
import biotite.sequence.align as align

COMPLEX = "7txw-assembly1"
ANTIGEN = "C"
ANTIBODY = ["A", "B"]

spi, model_seqs = am.build_spi("7txw")
cif = pdbx.CIFFile.read(GROUND_TRUTHS / f"{COMPLEX}.cif")
gt = pdbx.get_structure(cif, model=1, use_author_fields=False)
submat = align.SubstitutionMatrix.std_protein_matrix()


def chain_seq(label):
    cs = gt[gt.chain_id == label]
    rids, aas = [], []
    for res in struc.residue_iter(cs):
        aa = am.THREE_TO_ONE.get(res[0].res_name)
        if aa:
            rids.append(res[0].res_id)
            aas.append(aa)
    return rids, "".join(aas)


rids, seqstr = chain_seq(ANTIGEN)
gt_obj = bseq.ProteinSequence(seqstr)
best, bscore = None, None
for ms in set(model_seqs):
    al = align.align_optimal(gt_obj, bseq.ProteinSequence(ms), submat,
                             gap_penalty=(-10, -1), terminal_penalty=False)[0]
    if bscore is None or al.score > bscore:
        bscore, best = al.score, al.trace
pos_to_res = {mp: rids[gp] for gp, mp in best if gp >= 0 and mp >= 0}

ab = gt[np.isin(gt.chain_id, ANTIBODY) & ~struc.filter_solvent(gt)]
cC = gt[(gt.chain_id == ANTIGEN) & ~struc.filter_solvent(gt)]


def min_dist(model_pos):
    rid = pos_to_res.get(model_pos)
    if rid is None:
        return None, None
    atoms = cC[cC.res_id == rid]
    if len(atoms) == 0:
        return None, rid
    d = np.sqrt(((ab.coord[None] - atoms.coord[:, None]) ** 2).sum(-1))
    return float(d.min()), rid


# how many antigen residues are actually at the interface (<8A) for context
n_epi = sum(1 for mp in pos_to_res
            if (min_dist(mp)[0] or 1e9) < 8.0)
print(f"antigen chain C: {len(pos_to_res)} residues mapped, "
      f"{n_epi} are interface contacts (<8A to antibody)\n")

for label, specs in [("CANDIDATE", ["C48", "C49", "C50"]),
                     ("CONTROL", ["C65", "C70", "C98"])]:
    print(label)
    for s in specs:
        mp = int(s[1:]) - 1
        d, rid = min_dist(mp)
        tag = "  <-- INTERFACE (<8A)" if (d is not None and d < 8) else ""
        print(f"  {s} (model pos {mp}, native res {rid}): "
              f"min dist to antibody = {d:.1f} A{tag}" if d is not None
              else f"  {s}: unmapped")
    print()
