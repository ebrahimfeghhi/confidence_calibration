"""Reproduce the centroid-distance-across-runs view (the original figure) for our
finished repel configs: W1 -> G/B1/B2 vs run, per trial, star at real-G (wg<0.08)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CONFIGS = ["nudge_fz0", "nudge_zv", "maskreps_fz0", "maskreps_zv"]

fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(15, 11), squeeze=False)
for ci, cfg in enumerate(CONFIGS):
    f = OUT / f"repel_{cfg}.csv"
    if not f.exists():
        continue
    rows = list(csv.DictReader(open(f)))
    trials = sorted({int(r["trial"]) for r in rows})
    for ti in range(3):
        ax = axes[ci][ti]
        tr = [r for r in rows if int(r["trial"]) == ti]
        runs = [int(r["run"]) for r in tr]
        wg = [float(r["wg"]) for r in tr]
        ax.plot(runs, [float(r["wb1"]) for r in tr], "-o", color="0.5", ms=3, label="->B1")
        ax.plot(runs, [float(r["wb2"]) for r in tr], "-o", color="tab:orange", ms=3, label="->B2")
        ax.plot(runs, wg, "-o", color="tab:green", ms=3, label="->G")
        ax.axhspan(0, 0.08, color="green", alpha=0.12)
        for rn, g in zip(runs, wg):
            if g < 0.08:
                ax.plot(rn, g, "*", color="darkgreen", ms=16)
        ax.set_ylim(0, 1.4)
        if ti == 0:
            ax.set_ylabel(f"{cfg}\nW1 to centroid", fontsize=9)
        if ci == 0:
            ax.set_title(f"trial {ti}")
        if ci == 0 and ti == 0:
            ax.legend(fontsize=7)
        ax.set_xlabel("run")
plt.suptitle("Repulsion: W1 distance to G/B1/B2 across runs  (* = real G, wg<0.08)", y=0.995)
plt.tight_layout()
plt.savefig(OUT / "repel_curves.png", dpi=120)
print("saved repel_curves.png")
# also print a compact text view of wg per run for the winner
print("\nnudge_fz0  wg-to-G per run (per trial):")
rows = list(csv.DictReader(open(OUT / "repel_nudge_fz0.csv")))
for ti in range(3):
    seq = [f"{float(r['wg']):.2f}{'*' if float(r['wg'])<0.08 else ''}"
           for r in rows if int(r["trial"]) == ti]
    print(f"  trial {ti}: {' '.join(seq)}")
