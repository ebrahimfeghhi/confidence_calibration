#!/usr/bin/env python
"""1-seed vs 50-seed ipTM-selected DockQ, per interface, ab-ag beside protein-protein.

Both axes are DockQ of an ipTM-SELECTED pose -- never an oracle -- so vertical distance from the
diagonal is what extra sampling actually buys a user at inference time. Points above the diagonal
improved with 50 seeds, points below got worse (ipTM selection is not monotone in pool size).
Quadrant lines at the 0.23 acceptability threshold split the interfaces into rescued / lost /
already-fine / still-broken.

Usage: .../python scripts/plot_selection_scatter.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
T = 0.23
ARMS = [("antibody-antigen", f"{EXP}/artifacts/abag_selection_protocol.csv", "#b3452c"),
        ("protein-protein", f"{EXP}/artifacts/pp_selection_protocol.csv", "#31688e")]


def main():
    import sys
    # "selected" = what ipTM actually picks; "oracle" = the best pose available, i.e. the same
    # comparison with selection removed, which separates COVERAGE gains from SELECTION gains.
    mode = sys.argv[1] if len(sys.argv) > 1 else "selected"
    c1, c50, lab, out = (("sel1_dockq", "sel50_dockq", "ipTM-selected", "selection_scatter.png")
                         if mode == "selected" else
                         ("oracle1_dockq", "oracle_dockq", "oracle (best pose)",
                          "selection_scatter_oracle.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.6), sharex=True, sharey=True)
    for ax, (name, path, col) in zip(axes, ARMS):
        r = list(csv.DictReader(open(path)))
        a = np.array([float(x[c1]) for x in r])
        b = np.array([float(x[c50]) for x in r])
        ax.plot([0, 1], [0, 1], color="0.45", lw=1, ls="--", zorder=1)
        ax.axhline(T, color="0.8", lw=0.8, zorder=0)
        ax.axvline(T, color="0.8", lw=0.8, zorder=0)
        up = b > a
        ax.scatter(a[up], b[up], s=26, c=col, alpha=0.75, lw=0, zorder=3)
        ax.scatter(a[~up], b[~up], s=26, facecolors="none", edgecolors=col, lw=0.9,
                   alpha=0.75, zorder=3)
        resc = int(((a < T) & (b >= T)).sum())
        lost = int(((a >= T) & (b < T)).sum())
        ax.set_title(f"{name}   n={len(r)} interfaces\n"
                     f"mean {a.mean():.3f} -> {b.mean():.3f}  (delta {b.mean()-a.mean():+.3f})   "
                     f"{100*up.mean():.0f}% improved\n"
                     f"crossed 0.23 upward: {resc}    fell below 0.23: {lost}", fontsize=10)
        ax.set_xlabel(f"DockQ with 1 seed ({lab})")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"DockQ with 50 seeds ({lab})")
    h = [plt.Line2D([], [], marker="o", ls="", color=ARMS[0][2], label="improved with 50 seeds"),
         plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ARMS[0][2],
                    label="same or worse with 50 seeds"),
         plt.Line2D([], [], ls="--", color="0.45", label="no change (y = x)"),
         plt.Line2D([], [], ls="-", color="0.8", label="DockQ 0.23 acceptability threshold")]
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(f"{EXP}/artifacts/{out}", dpi=150)
    print(f"wrote artifacts/{out}")


if __name__ == "__main__":
    main()
