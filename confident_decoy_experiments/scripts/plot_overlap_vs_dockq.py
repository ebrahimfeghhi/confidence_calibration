#!/usr/bin/env python
"""Epitope and paratope Jaccard against DockQ, on the pose a user would actually ship.

One point per interface. The pose is the 50-seed ipTM pick (`sel50_*` in
artifacts/abag_selection_protocol.csv) -- not an oracle, not a per-pose scatter -- so this shows how
well surface identification predicts final quality for the structure the pipeline hands you.

Both overlaps are 5 A heavy-atom contact-set Jaccard against the crystal, computed on the antigen
side (epitope) and the antibody side (paratope). Neither is a DockQ component, so plotting them
against DockQ is not circular the way fnat would be.

Usage: .../python scripts/plot_overlap_vs_dockq.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
T = 0.23


def main():
    r = list(csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")))
    d = np.array([float(v["sel50_dockq"]) for v in r])
    panels = [("epitope", np.array([float(v["sel50_epi_jac"]) for v in r]), "#b3452c"),
              ("paratope", np.array([float(v["sel50_para_jac"]) for v in r]), "#31688e")]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.1), sharey=True)
    for ax, (name, x, col) in zip(axes, panels):
        ax.axhline(T, color="0.8", lw=0.9, zorder=0)
        ax.scatter(x, d, s=28, c=col, alpha=0.72, lw=0, zorder=3)
        rr = np.corrcoef(x, d)[0, 1]
        m, b = np.polyfit(x, d, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, color="0.25", lw=1.4, zorder=4,
                label=f"slope {m:.2f},  r$^2$ = {rr**2:.3f}")
        ax.set_title(f"{name} Jaccard of the shipped pose\n"
                     f"mean {x.mean():.3f} +/- {x.std():.3f}", fontsize=11)
        ax.set_xlabel(f"{name} Jaccard (50-seed ipTM-selected pose)")
        ax.set_xlim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("DockQ (50-seed ipTM-selected pose)")
    axes[0].set_ylim(-0.03, 1.03)
    fig.suptitle(f"how well does surface overlap predict DockQ?   "
                 f"n = {len(r)} ab-ag interfaces, one point each", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{EXP}/artifacts/overlap_vs_dockq.png", dpi=150)
    for name, x, _ in panels:
        print(f"{name:<9} r2 {np.corrcoef(x, d)[0,1]**2:.3f}  slope {np.polyfit(x, d, 1)[0]:.3f}  "
              f"mean {x.mean():.3f}+/-{x.std():.3f}")
    print("wrote artifacts/overlap_vs_dockq.png")


if __name__ == "__main__":
    main()
