#!/usr/bin/env python
"""Does the pair rep carry mass on the TRUE epitope, or only on the decoy one?

The distogram head reads straight out of z: contact_probs = compute_contact_prob(distogram_head(z)),
[N_token, N_token], with NO diffusion involved. So the antibody x antigen block is the trunk's belief
about which residues touch, as a distribution rather than one realised pose.

Per antigen residue we take the MAX contact probability against any antibody residue -- an epitope
residue only needs to touch one paratope residue, so max answers "is this residue in the interface at
all", whereas a sum would reward residues merely near many antibody residues and a mean would dilute a
real contact across ~200 irrelevant partners. A single spurious pair can light one residue, so read
the profile shape, not individual values.

8che is a TWO-COPY assembly (A/B 215, C/D 219, E/F 37), so there are two antibody x antigen blocks
(C-E and D-F) in the same matrix. OST picks whichever copy fits the reference better, per pose -- for
seed 137 that was C-E for samples 0,2,3,4 and D-F for sample 1. Both blocks are reported: if they
disagree, "the trunk's epitope belief" is not a single object.

Usage: .../python scripts/trunk_epitope_profile.py
"""
import json, sys
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
d = np.load(f"{EXP}/artifacts/reps/8che_seed137.npz")
cp, aid, ri = d["contact_probs"].astype(np.float32), d["asym_id"], d["residue_index"]
ep = json.load(open(f"{EXP}/artifacts/epitope_8che_BE_seed137.json"))

# chain letter -> asym_id: chains are emitted in order A,B,C,... so A=0, B=1, ...
def asym_of(letter):
    return ord(letter) - ord("A")

PAIRS = [("C", "E"), ("D", "F")]
gt_epi = {int(x) for x in ep["gt"]["epitope"]}
s3_epi = {int(x) for x in ep["poses"]["3"]["epitope"]}
bad_epi = set()
for k in ("0", "1", "2", "4"):
    bad_epi |= {int(x) for x in ep["poses"][k]["epitope"]}

print(f"GT epitope      : {sorted(gt_epi)}")
print(f"sample 3 epitope: {sorted(s3_epi)}   (the GOOD pose, DockQ 0.529)")
print(f"bad samples     : {sorted(bad_epi)}   (union of samples 0,1,2,4)")

profiles = {}
for ab, ag in PAIRS:
    m_ab, m_ag = aid == asym_of(ab), aid == asym_of(ag)
    assert m_ab.sum() and m_ag.sum(), f"empty chain {ab}/{ag}"
    block = cp[np.ix_(m_ab, m_ag)]                 # [n_ab, n_ag]
    res = ri[m_ag]                                  # antigen residue numbering
    prof = block.max(axis=0)                        # max over antibody residues
    profiles[(ab, ag)] = (res, prof)
    # numbering sanity: epitope sets are label_seq_id; residue_index must cover them
    miss = (gt_epi | s3_epi) - set(int(x) for x in res)
    print(f"\n=== {ab} x {ag}  block {block.shape}, antigen residues {res.min()}..{res.max()} ===")
    if miss:
        print(f"  *** numbering mismatch: epitope residues absent from residue_index: {sorted(miss)}")
    for r, p in zip(res, prof):
        tag = ""
        if int(r) in gt_epi:  tag += " GT"
        if int(r) in s3_epi:  tag += " s3"
        if int(r) in bad_epi: tag += " bad"
        bar = "#" * int(round(p * 40))
        print(f"  res {int(r):>3}  {p:5.3f} {bar:<40}{tag}")
    inset = np.array([int(r) in gt_epi for r in res])
    print(f"  mean propensity  GT-epitope residues {prof[inset].mean():.3f} (n={inset.sum()})   "
          f"other residues {prof[~inset].mean():.3f} (n={(~inset).sum()})")
    top = res[np.argsort(-prof)][:10]
    print(f"  top-10 residues by propensity: {[int(x) for x in top]}")

(rA, pA), (rB, pB) = profiles[("C", "E")], profiles[("D", "F")]
if len(pA) == len(pB):
    print(f"\nC-E vs D-F profile agreement: pearson r = {np.corrcoef(pA, pB)[0,1]:.3f}, "
          f"mean |diff| = {np.abs(pA-pB).mean():.3f}")
