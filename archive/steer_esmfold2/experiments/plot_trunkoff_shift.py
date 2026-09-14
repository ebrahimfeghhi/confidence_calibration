"""plot_trunkoff_shift.py — per Type-2 interface, show how ipTM shifts trunk-ON -> trunk-OFF for
GOOD docks (DockQ>0.4) vs CONFIDENT DECOYS (top-20 trunk-on decoys, DockQ<0.1). Slope chart: median line
+ IQR band for each group. Shows whether trunk-off lifts the good docks toward/past the confident decoys.
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = defaultdict(list)
for g in [0, 1, 2, 3]:
    try:
        for r in csv.DictReader(open(f"results/csv_files/trunkoff_type2_s{g}.csv")):
            try:
                on = float(r['iptm_on']) if r['iptm_on'] else None
                if on is None: continue
                rows[(r['cx'], r['c1'], r['c2'])].append((float(r['dockq']), on, float(r['iptm_off'])))
            except: pass
    except FileNotFoundError: pass

ifaces = [k for k, d in sorted(rows.items()) if sum(1 for x in d if x[0] > 0.4) >= 3]
n = len(ifaces); ncol = 4; nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.4*nrow)); axes = np.array(axes).ravel()
for ax, key in zip(axes, ifaces):
    d = rows[key]
    good = np.array([(on, off) for dq, on, off in d if dq > 0.4])
    dec = sorted([(on, off) for dq, on, off in d if dq < 0.1], key=lambda x: -x[0])[:20]
    dec = np.array(dec) if dec else np.empty((0, 2))
    for arr, col, lab in [(good, '#2ca02c', f'good (n={len(good)})'), (dec, '#d62728', 'confident decoy (top20)')]:
        if len(arr) == 0: continue
        med = np.median(arr, 0); q1 = np.percentile(arr, 25, 0); q3 = np.percentile(arr, 75, 0)
        ax.plot([0, 1], med, '-o', color=col, lw=2.5, label=lab, zorder=3)
        ax.fill_between([0, 1], q1, q3, color=col, alpha=0.15, zorder=1)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(['trunk-ON', 'trunk-OFF'])
    ax.set_title(f"{key[0]} {key[1]}-{key[2]}", fontsize=10); ax.set_ylabel("pair ipTM")
    ax.legend(fontsize=7, loc='best')
for ax in axes[n:]: ax.axis('off')
fig.suptitle("Type-2: ipTM shift trunk-ON → trunk-OFF, good docks (green) vs confident decoys (red)  |  line=median, band=IQR", fontsize=12)
fig.tight_layout()
fig.savefig("results/figures/trunkoff_shift.png", dpi=110); print("saved results/figures/trunkoff_shift.png")

# also a compact numeric table
print(f"\n{'iface':>10} {'good_on':>7} {'good_off':>8} {'dec_on':>7} {'dec_off':>8} {'gap_on':>7} {'gap_off':>7}")
for key in ifaces:
    d = rows[key]
    g = np.array([(on, off) for dq, on, off in d if dq > 0.4])
    de = np.array(sorted([(on, off) for dq, on, off in d if dq < 0.1], key=lambda x: -x[0])[:20])
    go, gf = np.median(g, 0); do, df = np.median(de, 0)
    print(f"{key[0]+' '+key[1]+'-'+key[2]:>10} {go:>7.3f} {gf:>8.3f} {do:>7.3f} {df:>8.3f} {go-do:>+7.3f} {gf-df:>+7.3f}")
