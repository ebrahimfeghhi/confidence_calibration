#!/usr/bin/env python
"""Does seed scaling's DockQ gain come from finding the right epitope?

x  gain in epitope Jaccard  = (ipTM pick pooled over 50 seeds) - (median epitope Jaccard among the
                              50 per-seed ipTM picks)
y  gain in DockQ            = (ipTM pick pooled over 50 seeds) - (median DockQ among the same 50
                              per-seed ipTM picks)

Each axis uses the median of the quantity it plots, so each baseline is a real pose but the two
baselines are DIFFERENT poses. That is deliberate: taking the DockQ-median pose's incidental epitope
Jaccard would make the x-axis a property of a pose chosen for another reason.

WHICH NUMBER TO QUOTE: r2 = 0.82, from this default "per-axis" mode. The alternative
(`consistent`, both axes off the single DockQ-median pose) gives 0.75, and that gap is REGRESSION
ATTENUATION, not a confound being removed: replacing a median-of-50 baseline with one pose's
incidental value adds noise of SD 0.094 to x. Attenuation predicts the r2 should fall by
var(clean x)/var(noisy x) = 0.901 and the slope by the same factor; observed 0.914 and
0.874 -> 0.793. So 0.75 is the attenuated estimate and 0.82 the less biased one.

Both arms are ipTM-selected -- no oracle -- so this is the gain a user actually receives.
Ab-ag only; protein-protein has no epitope.

Usage: .../python scripts/plot_gain_epitope_vs_dockq.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def main():
    import sys
    # per-axis   each baseline is the median of the quantity plotted -> two different poses
    # consistent both baselines are ONE pose, the DockQ-median pick, so x and y describe the same
    #            structure and no part of the correlation comes from mixing poses
    mode = sys.argv[1] if len(sys.argv) > 1 else "per-axis"
    # third arg picks which side of the interface: epitope (antigen) or paratope (antibody)
    side = sys.argv[2] if len(sys.argv) > 2 else "epi"
    # fourth arg picks the DATASET. "foldbench" is the original ab-ag set scored with OST; "pxm" is
    # PXMeter's independent PXM-2025 (2025 depositions, zero PDB overlap) scored with PXMeter's own
    # DockQ. Same estimator either way -- that is the point of routing both through this script.
    data = sys.argv[3] if len(sys.argv) > 3 else "foldbench"
    SRC = {"foldbench": f"{EXP}/artifacts/abag_selection_protocol.csv",
           "pxm": f"{EXP}/artifacts/pxm2025/pxm2025_selection_protocol.csv"}[data]
    suffix = "" if data == "foldbench" else f"_{data}"
    tag = "epitope" if side == "epi" else "paratope"
    # ARM. "selected" (default) is what a user receives: both ends ipTM-picked. "oracle" strips
    # selection out entirely -- best-available DockQ against best-available epitope Jaccard, each with
    # a median-of-per-seed-bests baseline -- so it asks whether seed scaling's COVERAGE gain is
    # epitope discovery, independent of whether confidence can find it.
    arm = os.environ.get("ARM", "selected")
    if arm == "oracle":
        x50, xmed, x1 = f"oracle_{side}_jac", f"oracle1_{side}_jac", f"oracle1_{side}_jac"
        ycol, ybase = "oracle_dockq", "oracle1_dockq"
    else:
        x50, xmed, x1 = f"sel50_{side}_jac", f"sel1med_{side}_jac", f"sel1_{side}_jac"
        ycol, ybase = "sel50_dockq", "sel1_dockq"
    xcol = xmed if mode == "per-axis" else x1
    arm_sfx = "" if arm == "selected" else f"_{arm}"
    pick = ("50-seed ipTM pick" if arm == "selected" else "best of 50 seeds")
    base = ("median of the 50 per-seed picks" if arm == "selected"
            else "median of the 50 per-seed bests")
    note = ("baselines: median of each quantity separately (two different poses)"
            if mode == "per-axis" else
            "baselines: the SAME pose (the DockQ-median per-seed pick) for both axes")
    out = (f"gain_{tag}_vs_dockq{arm_sfx}{suffix}.png" if mode == "per-axis"
           else f"gain_{tag}_vs_dockq_consistent{arm_sfx}{suffix}.png")
    r = [v for v in csv.DictReader(open(SRC)) if v.get(x50) and v.get(xcol)]
    x = np.array([float(v[x50]) - float(v[xcol]) for v in r])
    y = np.array([float(v[ycol]) - float(v[ybase]) for v in r])

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.axhline(0, color="0.75", lw=0.9, zorder=0)
    ax.axvline(0, color="0.75", lw=0.9, zorder=0)
    ax.scatter(x, y, s=30, c="#b3452c", alpha=0.72, lw=0, zorder=3)

    m, b = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color="0.25", lw=1.4, ls="-", zorder=4,
            label=f"least squares: slope {m:.2f},  r$^2$ = {r2:.3f}")

    q = [int(((x > 0) & (y > 0)).sum()), int(((x <= 0) & (y > 0)).sum()),
         int(((x <= 0) & (y <= 0)).sum()), int(((x > 0) & (y <= 0)).sum())]
    ax.set_xlabel(f"gain in {tag} Jaccard\n({pick}  -  {base})", fontsize=15)
    ax.set_ylabel(f"gain in DockQ\n({pick}  -  {base})", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/{out}", dpi=150)
    print(f"n={len(r)}  slope {m:.3f}  r2 {r2:.3f}")
    print(f"quadrants  (+,+) {q[0]}  (0/-,+) {q[1]}  (0/-,0/-) {q[2]}  (+,0/-) {q[3]}")
    print(f"wrote artifacts/{out}")


if __name__ == "__main__":
    main()
