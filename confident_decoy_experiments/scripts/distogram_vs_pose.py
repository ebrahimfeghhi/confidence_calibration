#!/usr/bin/env python
"""Is the distogram a good readout of the epitope the model actually builds?

The steering plan pushes on the DISTOGRAM's epitope. That only makes sense if the distogram agrees
with the epitope of the structure the diffusion module then produces. This compares, per seed:

  distogram epitope : top-K antigen residues by P(d < 8 A) at the LAST recycle
  pose epitope      : antigen residues within 5 A (heavy atom) of the antibody in the decoded cif
  true epitope      : the same 5 A definition on the crystal structure

Agreement is reported as Jaccard and as recall of the pose epitope. If the distogram disagrees with
its own decoded pose, steering the distogram steers the wrong object.
"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, "scripts")
from commit_contact import load
from cdr_h3_rmsd import GT_DIR, _ost, chain_residues, align_map
from epitope_clusters import contact_residues
from epitope_rmsd import native_epitope
from iface_chains import first_pose_with

ARM = sys.argv[1] if len(sys.argv) > 1 else "nodrop2"
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 40
_, io, _ = _ost()

ref = io.LoadMMCIF(f"{GT_DIR}/8slb-assembly1.cif", fault_tolerant=True)
c0, _, _ = first_pose_with("8slb", "B", "A")
native = native_epitope(ref, c0["ref_ab"], c0["ref_ag"])          # GT numbering

D = load(ARM)                                                      # pass -> cycle -> P(contact)
rows = []
for cif in sorted(glob.glob(f"artifacts/commit/out_{ARM}/8slb/seed_*/predictions/*_sample_0.cif")):
    seed = int(re.search(r"seed_(\d+)", cif).group(1))
    mdl = io.LoadMMCIF(cif, fault_tolerant=True)
    names = [c.GetName() for c in mdl.chains]
    ag_c, ab_c = names[0], names[1]                                # asym 0 = antigen, 1 = heavy
    pose = contact_residues(mdl, ab_c, ag_c)                       # model numbering
    # model numbering -> GT numbering, via the antigen chain alignment
    mres = chain_residues(mdl, ag_c); rres = chain_residues(ref, c0["ref_ag"])
    amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
    m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
    pose_gt = {m2g[r] for r in pose if r in m2g}
    # distogram epitope: entity index -> GT number is +1 for this construct
    p = D[seed - 101][9]
    dis_gt = {int(i) + 1 for i in np.argsort(p.max(axis=1))[::-1][:TOPK]}
    j = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    rows.append((seed, len(pose_gt), j(dis_gt, pose_gt),
                 len(dis_gt & pose_gt) / max(len(pose_gt), 1),
                 j(pose_gt, native), j(dis_gt, native)))

print(f"ARM {ARM}, distogram top-{TOPK} at the final recycle vs the decoded pose\n")
print(f"{'seed':>5}{'pose epi size':>15}{'jaccard dis/pose':>18}{'recall of pose':>16}"
      f"{'jaccard pose/true':>19}{'jaccard dis/true':>18}")
for r in rows:
    print(f"{r[0]:>5}{r[1]:>15}{r[2]:>18.2f}{r[3]:>16.2f}{r[4]:>19.2f}{r[5]:>18.2f}")
a = np.array([r[2:] for r in rows])
print(f"\n{'mean':>5}{'':>15}{a[:,0].mean():>18.2f}{a[:,1].mean():>16.2f}{a[:,2].mean():>19.2f}{a[:,3].mean():>18.2f}")
print(f"\ntrue epitope has {len(native)} residues")
