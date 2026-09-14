#!/usr/bin/env python
"""One figure: the pose-vs-pose epitope Jaccard distance matrix, reordered by average linkage.

Blocks with a sharp edge -> discrete modes. A smooth gradient -> a continuum.
The linkage only ORDERS the display; the evidence is the distances themselves.
See scripts/epitope_distmat.py for why a cluster count is not the right output here.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"

ap = argparse.ArgumentParser()
ap.add_argument("--interface", required=True)
A = ap.parse_args()
# scripts/epitope_distmat.py writes to artifacts/distmat/; older runs left files directly in
# artifacts/. Accept both rather than silently failing on the layout.
NPZ = next(p for p in (f"{EXP}/artifacts/distmat/epitope_distmat_{A.interface}.npz",
                       f"{EXP}/artifacts/epitope_distmat_{A.interface}.npz")
           if os.path.exists(p))
z = np.load(NPZ, allow_pickle=True)
D, dq = z["D"], z["dockq"]
n = len(dq)
order = leaves_list(linkage(squareform(D, checks=False), method="average"))

fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(D[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
ax.set_xlabel("pose (reordered)", fontsize=11, labelpad=8)
ax.set_ylabel("pose (reordered)", fontsize=11, labelpad=8)
ax.set_title(f"{A.interface}  —  epitope distance between poses\n"
             f"{n} poses, DockQ {dq.min():.2f}–{dq.max():.2f}",
             fontsize=12, pad=14)
ax.set_xticks([0, 50, 100, 150, 200, 250][:6])
ax.set_yticks([0, 50, 100, 150, 200, 250][:6])
ax.tick_params(labelsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Jaccard distance   (0 = same epitope, 1 = no shared residues)",
             fontsize=10, labelpad=10)
cb.ax.tick_params(labelsize=9)
fig.tight_layout()
out = f"{EXP}/artifacts/epitope_distmat_{A.interface}.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
