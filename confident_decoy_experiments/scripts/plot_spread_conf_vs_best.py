#!/usr/bin/env python
"""Per interface: how scattered is the MOST CONFIDENT seed vs the seed holding the BEST pose?

x  spread within the most-confident seed   (highest mean pair_iptm over its 5 samples)
y  spread within the best seed             (contains the highest-DockQ pose)
One point per interface; the diagonal is "equally scattered". Above the diagonal means the good pose
came from a seed whose samples DISAGREED, while confidence went to a seed whose samples AGREED.

SPREAD is the standard deviation across a seed's 5 samples -- left panel of epitope Jaccard (which
basin), right panel of DockQ (pose quality). Epitope is the more interpretable of the two since it is
the quantity that actually differs between docking sites; DockQ is shown because it needs no epitope
computation and would be the cheaper signal to deploy.

COLOURED BY OUTCOME, which is the point of the figure. On the 85 selection FAILURES (oracle >= 0.23,
ipTM-selected < 0.23) the best seed is more scattered than the confident seed in 80/85 cases, paired
Wilcoxon p = 3.1e-13, median spreads 0.164 vs 0.016, and 40/85 confident seeds have EXACTLY zero
spread. But that is measured only where confidence already failed. If the SUCCESSES sit in the same
place, scatter is just a marker of a hard target and is useless as a selection signal; if they
separate, "prefer seeds whose samples disagree" is a usable rule. That comparison is the figure.

EXACT POSE SET enforced (50 seeds x 5 samples per interface); short interfaces are excluded, counted,
never partially averaged.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/plot_spread_conf_vs_best.py
Writes artifacts/pxm2025/spread_conf_vs_best.png
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
T = 0.23


def main():
    epi = {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        epi[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))),
             int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])

    poses = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not (r["dockq"] and r["pair_iptm"]):
            continue
        k = (r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))), int(r["seed"]), int(r["sample"]))
        if k in epi:
            poses[f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'][int(r["seed"])].append(
                (float(r["pair_iptm"]), float(r["dockq"]), epi[k]))

    rows, skipped = [], 0
    for iface, byseed in poses.items():
        byseed = {s: v for s, v in byseed.items() if len(v) == 5}
        if len(byseed) < 40:
            skipped += 1
            continue
        allp = [p for v in byseed.values() for p in v]
        oracle = max(p[1] for p in allp)
        sel = max(allp, key=lambda p: p[0])[1]
        if oracle < T:                    # coverage failure: nothing to select, excluded
            continue
        best = max(byseed, key=lambda s: max(p[1] for p in byseed[s]))
        conf = max(byseed, key=lambda s: np.mean([p[0] for p in byseed[s]]))
        sd = lambda s, i: float(np.std([p[i] for p in byseed[s]]))
        rows.append(dict(iface=iface, ok=sel >= T, same=best == conf,
                         e_best=sd(best, 2), e_conf=sd(conf, 2),
                         d_best=sd(best, 1), d_conf=sd(conf, 1)))

    F = [r for r in rows if not r["ok"]]
    S = [r for r in rows if r["ok"]]
    print(f"{len(rows)} interfaces with an acceptable pose available "
          f"({len(F)} selection FAILED, {len(S)} succeeded; {skipped} excluded)\n")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.3))
    for ax, (kb, kc, lab, hi) in zip(axes, (("e_best", "e_conf", "epitope Jaccard", 0.5),
                                            ("d_best", "d_conf", "DockQ", 0.5))):
        for grp, col, name in ((S, "#31688e", "selection SUCCEEDED"),
                               (F, "#c0392b", "selection FAILED")):
            x = np.array([r[kc] for r in grp])
            y = np.array([r[kb] for r in grp])
            ax.scatter(x, y, s=34, c=col, alpha=0.7, lw=0,
                       label=f"{name}  (n={len(grp)}, {100*(y>x).mean():.0f}% above line)")
        ax.plot([0, hi], [0, hi], color="0.5", lw=1.2)
        ax.set_xlabel(f"spread of {lab} within the MOST CONFIDENT seed", fontsize=12)
        ax.set_ylabel(f"spread within the BEST seed", fontsize=12)
        ax.set_title(f"{lab} spread  (SD over a seed's 5 samples)", fontsize=13)
        ax.set_xlim(-0.02, hi)
        ax.set_ylim(-0.02, hi)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(f"{P}/spread_conf_vs_best.png", dpi=150)

    print(f"{'group':<22}{'n':>5}{'med spread CONF':>18}{'med spread BEST':>18}"
          f"{'% best more scattered':>23}{'best==conf seed':>17}")
    for grp, name in ((F, "selection FAILED"), (S, "selection SUCCEEDED")):
        if not grp:
            continue
        ec = np.array([r["e_conf"] for r in grp]); eb = np.array([r["e_best"] for r in grp])
        print(f"{name:<22}{len(grp):>5}{np.median(ec):>18.3f}{np.median(eb):>18.3f}"
              f"{100*(eb>ec).mean():>22.0f}%{sum(r['same'] for r in grp):>13}/{len(grp)}")
    print(f"\nwrote {P}/spread_conf_vs_best.png")


if __name__ == "__main__":
    main()
