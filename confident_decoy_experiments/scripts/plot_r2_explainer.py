#!/usr/bin/env python
"""What r^2 means, drawn on our own data.

r^2 = 1 - (leftover spread) / (total spread), where each "spread" is the sum of the SQUARED lengths
of the vertical red sticks in the corresponding panel:
  panel 1  sticks run from each point to the MEAN of y      -> total spread   (the null model)
  panel 2  sticks run from each point to the FITTED LINE    -> leftover spread
Shrinking the sticks is the whole game. Panel 3 repeats panel 2 for the paratope, where the sticks
stay long -- that is what a low r^2 looks like.

Uses the same numbers as scripts/plot_gain_epitope_vs_dockq.py: one point per ab-ag interface, both
axes ipTM-selected gains from seed scaling.

Usage: .../python scripts/plot_r2_explainer.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
DOT, STICK = "#b3452c", "#c8102e"


def main():
    r = list(csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")))
    F = lambda k: np.array([float(v[k]) for v in r])
    y = F("sel50_dockq") - F("sel1_dockq")
    epi = F("sel50_epi_jac") - F("sel1med_epi_jac")
    par = F("sel50_para_jac") - F("sel1med_para_jac")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharey=True)

    def panel(ax, x, mode, title, sub):
        if mode == "mean":
            base = np.full_like(y, y.mean())
            ax.axhline(y.mean(), color="0.25", lw=2, zorder=2)
        else:
            m, b = np.polyfit(x, y, 1)
            base = m * x + b
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, color="0.25", lw=2, zorder=2)
        for xi, yi, bi in zip(x, y, base):
            ax.plot([xi, xi], [bi, yi], color=STICK, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter(x, y, s=26, c=DOT, alpha=0.9, lw=0, zorder=3)
        ss = float(((y - base) ** 2).sum())
        ax.set_title(f"{title}\n{sub}\nsum of squared sticks = {ss:.2f}", fontsize=14)
        ax.set_xlabel("gain in Jaccard", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.18)
        return ss

    tot = panel(axes[0], epi, "mean", "1. no predictor: distance to the MEAN",
                "this is ALL the variation we want to explain")
    res_e = panel(axes[1], epi, "fit", "2. predictor = EPITOPE gain",
                  "sticks are now short")
    res_p = panel(axes[2], par, "fit", "3. predictor = PARATOPE gain",
                  "sticks stay long")
    axes[0].set_ylabel("gain in DockQ", fontsize=14)
    fig.text(0.5, 0.015,
             f"r$^2$ = 1  -  (leftover / total).      "
             f"epitope: 1 - {res_e:.2f}/{tot:.2f} = {1-res_e/tot:.2f}       "
             f"paratope: 1 - {res_p:.2f}/{tot:.2f} = {1-res_p/tot:.2f}",
             ha="center", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{EXP}/artifacts/r2_explainer.png", dpi=150)
    print(f"total {tot:.3f}  epitope residual {res_e:.3f} -> r2 {1-res_e/tot:.3f}")
    print(f"                paratope residual {res_p:.3f} -> r2 {1-res_p/tot:.3f}")
    print("wrote artifacts/r2_explainer.png")


if __name__ == "__main__":
    main()
