"""Plot individual B1-seed velocity profiles (W1 between consecutive loops) from
velocity_probe.csv, with the B1 mean and the G mean overlaid for reference.
CPU only."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(OUT / "velocity_probe.csv")))
loops = sorted({int(r["loop"]) for r in rows})


def series(seed):
    d = {int(r["loop"]): (float(r["velocity"]) if r["velocity"] != "" else np.nan)
         for r in rows if int(r["seed"]) == seed}
    return [d.get(lp, np.nan) for lp in loops]


def group_seeds(lbl):
    return sorted({int(r["seed"]) for r in rows if r["label"] == lbl})


def group_mean(lbl):
    seeds = group_seeds(lbl)
    M = np.array([series(s) for s in seeds], dtype=float)
    return np.nanmean(M, axis=0)


b1 = group_seeds("B1")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# panel 1: every B1 seed + means
for s in b1:
    ax[0].plot(loops, series(s), color="0.7", lw=0.8, alpha=0.7)
ax[0].plot(loops, group_mean("B1"), color="tab:blue", lw=2.5, label="B1 mean")
ax[0].plot(loops, group_mean("G"), color="tab:green", lw=2.5, ls="--", label="G mean (ref)")
ax[0].set_xlabel("recycling loop"); ax[0].set_ylabel("velocity (W1 loop i vs i-1)")
ax[0].set_title(f"B1 seed velocity profiles (n={len(b1)})"); ax[0].legend()

# panel 2: highlight any B1 seed with an anomalous late spike (near-escaper?)
late = {s: np.nanmax([series(s)[lp] for lp in range(5, len(loops))]) for s in b1}
top = sorted(late, key=late.get, reverse=True)[:5]
for s in b1:
    ax[1].plot(loops, series(s), color="0.85", lw=0.8)
for s in top:
    ax[1].plot(loops, series(s), lw=2, label=f"seed {s} (late max {late[s]:.2f})")
ax[1].plot(loops, group_mean("B1"), color="k", lw=1.5, ls=":", label="B1 mean")
ax[1].set_xlabel("recycling loop"); ax[1].set_ylabel("velocity")
ax[1].set_title("B1 seeds with the largest LATE (loop>=5) velocity"); ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "b1_velocity_profiles.png", dpi=130)
print("saved", OUT / "b1_velocity_profiles.png")
print("\nB1 late-velocity (loop>=5) ranking (top 8 — possible near-escapers):")
for s in sorted(late, key=late.get, reverse=True)[:8]:
    print(f"  seed {s:2d}: late-max velocity {late[s]:.3f}")
print(f"\nB1 mean velocity by loop: " +
      " ".join(f"{v:.2f}" for v in group_mean('B1')))
