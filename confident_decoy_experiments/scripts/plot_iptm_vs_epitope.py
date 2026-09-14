#!/usr/bin/env python
"""Is the model genuinely CONFIDENT in the wrong epitope, or just indifferent?

One panel per interface, all 250 poses: x = epitope Jaccard against the native epitope, y = interface
pair_iptm. The interfaces are the worst selection failures on PXM-2025, ranked by
(oracle DockQ - ipTM-selected DockQ).

WHAT THE SHAPES MEAN
  a high-ipTM cloud sitting at x ~ 0        -> the model is confidently committed to a WRONG surface
  ipTM flat across x                        -> confidence carries no epitope information at all
  ipTM rising with x                        -> confidence works here and the failure is elsewhere
Distinguishing the first two matters: the first is a confident-decoy problem (the head is wrong), the
second is an uninformative-head problem (the head is blind). They call for different fixes.

The two poses that decide the outcome are marked: the one ipTM SHIPS (red) and the one with the best
DockQ (green). The vertical distance between them is the selector's error in its own units.

pair_iptm is the PER-INTERFACE value, never the global iptm, which blends in intra-antibody
confidence and hides exactly this signal.

Sources: artifacts/pxm2025/dockq_pxm2025.csv (PXMeter DockQ + chain_pair_iptm) joined to
artifacts/pxm2025/epitope_overlap_pxm.csv on (pdb, unordered chain pair, seed, sample) -- the two
files store the pair in different orders, so the key is sorted.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/plot_iptm_vs_epitope.py [--top 10]
Writes artifacts/pxm2025/iptm_vs_epitope_worst.png
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    A = ap.parse_args()

    epi = {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        epi[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))),
             int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])

    poses = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not (r["dockq"] and r["pair_iptm"]):
            continue
        k = (r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))),
             int(r["seed"]), int(r["sample"]))
        if k in epi:
            poses[f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'].append(
                (epi[k], float(r["pair_iptm"]), float(r["dockq"])))

    rows = [v for v in csv.DictReader(open(f"{P}/pxm2025_selection_protocol.csv"))
            if v.get("sel50_epi_jac") and poses.get(v["interface"])]
    for v in rows:
        v["gap"] = float(v["oracle_dockq"]) - float(v["sel50_dockq"])
    rows.sort(key=lambda v: -v["gap"])
    rows = rows[:A.top]

    ncol = 5
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow), squeeze=False)
    print(f"{'interface':<13}{'n':>5}{'r(ipTM,epiJ)':>14}{'ipTM at epiJ<0.1':>18}"
          f"{'ipTM at epiJ>0.5':>18}")
    for ax, v in zip(axes.ravel(), rows):
        a = np.array(poses[v["interface"]])
        x, y, d = a[:, 0], a[:, 1], a[:, 2]
        ax.scatter(x, y, s=14, c="0.55", alpha=0.55, lw=0, zorder=2)
        si = int(np.argmax(y))
        bi = int(np.argmax(d))
        ax.scatter([x[si]], [y[si]], s=90, marker="X", c="#c0392b", zorder=5,
                   label=f"ipTM pick  DockQ {d[si]:.2f}")
        ax.scatter([x[bi]], [y[bi]], s=90, marker="*", c="#1e8449", zorder=5,
                   label=f"best pose  DockQ {d[bi]:.2f}")
        ax.set_title(v["interface"], fontsize=11)
        ax.set_xlabel("epitope Jaccard to native", fontsize=9)
        ax.set_ylabel("interface pair_iptm", fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        lo, hi = x < 0.1, x > 0.5
        r = np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan
        print(f'{v["interface"]:<13}{len(x):>5}{r:>+14.3f}'
              f'{(y[lo].mean() if lo.any() else np.nan):>18.3f}'
              f'{(y[hi].mean() if hi.any() else np.nan):>18.3f}')
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{P}/iptm_vs_epitope_worst.png", dpi=150)
    print(f"\nwrote {P}/iptm_vs_epitope_worst.png")


if __name__ == "__main__":
    main()
