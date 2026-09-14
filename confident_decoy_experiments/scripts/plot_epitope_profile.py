#!/usr/bin/env python
"""Epitope profile of two seeds at the final recycle.

For each antigen residue: P(contact) = the distogram's probability mass below 8 A, taken as the MAX
over all antibody residues -- "how likely is this residue touched by anything on the antibody".
One curve per seed, over the 288 antigen residues of 8slb.

Shaded bands mark, for reference, each seed's ACTUAL decoded epitope (antigen residues within 5 A of
the antibody in the output structure) and the crystal epitope.

CAVEAT built into the x-axis: residue index is sequence order, not 3D adjacency, so a spatially
compact patch can look scattered here.
"""
import glob, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
from commit_contact import load
from cdr_h3_rmsd import GT_DIR, _ost, chain_residues, align_map
from epitope_clusters import contact_residues
from epitope_rmsd import native_epitope
from iface_chains import first_pose_with

ARM, S1, S2 = "stock2", 101, 106
_, io, _ = _ost()
ref = io.LoadMMCIF(f"{GT_DIR}/8slb-assembly1.cif", fault_tolerant=True)
c0, _, _ = first_pose_with("8slb", "B", "A")
native = set(native_epitope(ref, c0["ref_ab"], c0["ref_ag"]))
rres = chain_residues(ref, c0["ref_ag"])

def pose_epitope(seed):
    cif = glob.glob(f"artifacts/commit/out_{ARM}/8slb/seed_{seed}/predictions/*_sample_0.cif")[0]
    mdl = io.LoadMMCIF(cif, fault_tolerant=True); nm = [c.GetName() for c in mdl.chains]
    mres = chain_residues(mdl, nm[0])
    amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
    m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
    return {m2g[r] for r in contact_residues(mdl, nm[1], nm[0]) if r in m2g}

D = load(ARM)
prof = {s: D[s - 101][9].max(axis=1) for s in (S1, S2)}            # P(contact) per antigen residue
gt = np.arange(288) + 1                                            # entity index + 1 = GT number
ep = {s: pose_epitope(s) for s in (S1, S2)}

fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True, sharey=True)
COL = {S1: "#1f6f8b", S2: "#c1553a"}
for ax, s in zip(axes, (S1, S2)):
    for r in sorted(native):
        ax.axvline(r, color="#2f6d54", alpha=.13, lw=2.2)
    for r in sorted(ep[s]):
        ax.axvline(r, color=COL[s], alpha=.16, lw=2.2)
    ax.plot(gt, prof[s], color=COL[s], lw=1.5)
    ax.set_ylabel("P(contact)", fontsize=10)
    ax.set_title(f"seed {s}   —   decoded epitope {len(ep[s])} residues, "
                 f"jaccard to crystal {len(ep[s]&native)/len(ep[s]|native):.2f}",
                 fontsize=11, loc="left", pad=8)
    ax.grid(alpha=.2)
axes[1].set_xlabel("antigen residue (ground-truth numbering)", fontsize=10, labelpad=7)
h = [plt.Line2D([], [], color="#2f6d54", lw=6, alpha=.35, label="crystal epitope"),
     plt.Line2D([], [], color=COL[S1], lw=6, alpha=.4, label=f"seed {S1} decoded epitope"),
     plt.Line2D([], [], color=COL[S2], lw=6, alpha=.4, label=f"seed {S2} decoded epitope")]
axes[0].legend(handles=h, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.28),
               ncol=3, frameon=False)
fig.suptitle("8slb, Arm B (stock) — distogram epitope profile at the final recycle\n"
             "P(contact) = distogram mass below 8 A, max over antibody residues",
             fontsize=12, y=1.10)
fig.tight_layout()
out = "artifacts/epitope_profile_101_106.png"
fig.savefig(out, dpi=155, bbox_inches="tight")
print("wrote", out)
print(f"seed {S1}: profile max {prof[S1].max():.3f}, median {np.median(prof[S1]):.4f}")
print(f"seed {S2}: profile max {prof[S2].max():.3f}, median {np.median(prof[S2]):.4f}")
print(f"decoded epitopes share {len(ep[S1]&ep[S2])} residues")
