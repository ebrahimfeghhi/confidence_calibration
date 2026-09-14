#!/usr/bin/env python
"""Per-seed within-trajectory loss curves: how much the epitope moves between consecutive recycles.

Reference is the SAME seed at the previous recycle, so this is "when does the model stop changing
its mind", not the repulsion objective (which compares different seeds at the same recycle).

Arm A = no MSA subsampling + no dropout.  Arm B = stock inference.
One line per seed, both losses on separate panels.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
from commit_contact import load, loss_A, loss_B

ARMS = [("Arm A  —  full MSA, no dropout", "nodrop2", "#1f6f8b"),
        ("Arm B  —  stock (MSA subsampled + dropout)", "stock2", "#c1553a")]
LOSSES = [("sorted-W1", loss_A,
           "sum over antigen residues of W1 between their\nP(contact) vectors at consecutive recycles"),
          ("profile-L1", loss_B,
           "sum over antigen residues of |change| in\nmax P(contact) at consecutive recycles")]

data = {tag: load(tag) for _, tag, _ in ARMS}
for lname, fn, subtitle in LOSSES:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, tag, col) in zip(axes, ARMS):
        M = data[tag]
        for p in sorted(M):
            y = [fn(M[p][k], M[p][k + 1]) for k in range(9)]
            ax.plot(range(9), y, color=col, alpha=0.45, lw=1.2, marker="o", ms=3)
        mean = np.mean([[fn(M[p][k], M[p][k + 1]) for k in range(9)] for p in M], axis=0)
        ax.plot(range(9), mean, color="k", lw=2.4, marker="o", ms=5, label=f"mean of {len(M)} seeds")
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("recycle transition", fontsize=10, labelpad=7)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{k}→{k+1}" for k in range(9)], fontsize=8)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel(f"{lname}  (vs the same seed's previous recycle)", fontsize=10, labelpad=8)
    fig.suptitle(f"8slb — epitope movement per recycle, {lname}\n{subtitle}",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    out = f"artifacts/commit_loss_{lname.replace('-', '')}.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    print("wrote", out)
