#!/usr/bin/env python
"""Epitope Jaccard vs DockQ, with paratope Jaccard as colour -- all three from the SAME pose.

One point per interface; the pose is the 50-seed ipTM pick (`sel50_*`), i.e. what the pipeline would
actually ship. The two-panel version of this (scripts/plot_overlap_vs_dockq.py) showed epitope
predicts DockQ far better than paratope (r2 0.81 vs 0.47) but could not show whether the epitope
plot's residual scatter is explained by the paratope. Encoding paratope as colour answers that
directly: if the low-DockQ points at high epitope Jaccard were a paratope failure, they would be
uniformly dark.

Usage: .../python scripts/plot_epi_dockq_para_color.py
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
    epi = np.array([float(v["sel50_epi_jac"]) for v in r])
    par = np.array([float(v["sel50_para_jac"]) for v in r])
    dq = np.array([float(v["sel50_dockq"]) for v in r])

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    ax.axhline(T, color="0.8", lw=0.9, zorder=0)
    m, b = np.polyfit(epi, dq, 1)
    xs = np.linspace(epi.min(), epi.max(), 50)
    ax.plot(xs, m * xs + b, color="0.3", lw=1.4, zorder=2,
            label=f"fit on epitope only: slope {m:.2f},  r$^2$ = {np.corrcoef(epi,dq)[0,1]**2:.3f}")
    s = ax.scatter(epi, dq, c=par, s=46, cmap="viridis", vmin=0, vmax=1,
                   edgecolors="0.3", lw=0.4, zorder=3)
    cb = fig.colorbar(s, ax=ax, pad=0.02)
    cb.set_label("paratope Jaccard (same pose)")

    # the diagnostic population: epitope found but DockQ still unacceptable
    bad = (epi >= 0.5) & (dq < T)
    ax.set_title("epitope Jaccard vs DockQ, coloured by paratope Jaccard\n"
                 f"n = {len(r)} ab-ag interfaces; all three metrics from the 50-seed ipTM pose\n"
                 f"epitope >= 0.5 but DockQ < 0.23: {int(bad.sum())} interfaces, "
                 f"mean paratope {par[bad].mean():.2f}" if bad.any() else "", fontsize=10)
    ax.set_xlabel("epitope Jaccard (50-seed ipTM-selected pose)")
    ax.set_ylabel("DockQ (50-seed ipTM-selected pose)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/epi_dockq_para_color.png", dpi=150)
    print(f"n={len(r)}  epitope r2 {np.corrcoef(epi,dq)[0,1]**2:.3f}")
    print(f"epitope>=0.5 & DockQ<0.23: n={int(bad.sum())}  "
          f"mean paratope {par[bad].mean():.3f} vs {par[~bad].mean():.3f} elsewhere")
    print("wrote artifacts/epi_dockq_para_color.png")


if __name__ == "__main__":
    main()
