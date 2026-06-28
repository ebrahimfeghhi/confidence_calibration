"""For the restless B1 seeds (high late velocity), is the velocity spike an
excursion-and-return (d_dense rises then falls) or in-place oscillation (d_dense
stays flat)? Plot velocity and distance-to-dense together. CPU only."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(OUT / "velocity_probe.csv")))
loops = sorted({int(r["loop"]) for r in rows})


def ser(seed, key):
    d = {int(r["loop"]): (float(r[key]) if r[key] != "" else np.nan)
         for r in rows if int(r["seed"]) == seed}
    return [d.get(lp, np.nan) for lp in loops]


B1 = sorted({int(r["seed"]) for r in rows if r["label"] == "B1"})
late = {s: np.nanmax([ser(s, "velocity")[lp] for lp in range(5, len(loops))]) for s in B1}
restless = sorted(late, key=late.get, reverse=True)[:4]
quiet = sorted(late, key=late.get)[:1]            # a calm B1 seed for contrast
GOOD = [10, 14]                                    # G refs

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for s in restless:
    ax[0].plot(loops, ser(s, "velocity"), lw=1.8, label=f"B1 s{s}")
    ax[1].plot(loops, ser(s, "d_dense"), lw=1.8, label=f"B1 s{s}")
for s in quiet:
    ax[0].plot(loops, ser(s, "velocity"), "k:", lw=1.5, label=f"B1 s{s} (quiet)")
    ax[1].plot(loops, ser(s, "d_dense"), "k:", lw=1.5, label=f"B1 s{s} (quiet)")
for s in GOOD:
    ax[0].plot(loops, ser(s, "velocity"), "--", color="0.5", lw=1.3, label=f"G s{s}")
    ax[1].plot(loops, ser(s, "d_dense"), "--", color="0.5", lw=1.3, label=f"G s{s}")

ax[0].set_title("velocity (W1 loop i vs i-1)"); ax[0].set_xlabel("loop"); ax[0].legend(fontsize=8)
ax[1].set_title("distance to dense (B1) centroid"); ax[1].set_xlabel("loop"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "b1_restless_vs_dense.png", dpi=130)
print("saved", OUT / "b1_restless_vs_dense.png")

print("\nrestless B1 seeds — does d_dense excurse (rise then fall) or stay flat?")
for s in restless:
    dd = ser(s, "d_dense")
    print(f"  seed {s:2d}: d_dense  L0={dd[0]:.2f}  max={np.nanmax(dd):.2f}@L{int(np.nanargmax(dd))}  "
          f"L16={dd[-1]:.2f}   vel late-max {late[s]:.2f}")
print("  (excursion-return: max in the middle, L16 back near L0; in-place: d_dense ~flat)")
