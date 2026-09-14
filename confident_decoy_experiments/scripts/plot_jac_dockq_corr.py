#!/usr/bin/env python
"""WITHIN an interface, does epitope (or paratope) Jaccard track DockQ across its 250 poses?

For each interface separately, correlate the per-pose Jaccard against the per-pose DockQ over its
50 seeds x 5 samples, then histogram those correlations across interfaces. One count per interface.

WHY WITHIN-INTERFACE. Every earlier regression here was BETWEEN interfaces (one point per interface,
gain vs gain), which answers "does finding the epitope explain why scaling helps this target". This
asks the different and more direct question: inside one target, are the good poses the ones on the
right epitope? Between-interface structure cannot leak in, because each correlation is computed on a
single interface's own poses.

BOTH SIDES ARE PLOTTED because they are not redundant and not interchangeable:
  - epitope Jaccard  = which ANTIGEN residues are contacted -- where the antibody docked.
  - paratope Jaccard = which ANTIBODY residues are contacted. NOT a measure of antibody internal
    geometry (that is paratope RMSD); it moves when the docking site moves, so it is largely
    inherited from epitope choice. Measured: per-interface epitope SD and paratope SD correlate at
    r = +0.50 to +0.57.

DEGENERATE INTERFACES ARE DROPPED, NOT ZEROED. A correlation is undefined when either variable is
constant over the 250 poses -- e.g. an interface that fails identically every time. Those are
excluded and counted; writing them as r = 0 would invent a "no relationship" finding out of a
"no variation" one.

Pearson is plotted (the question is linear co-movement) and Spearman is printed alongside, since
DockQ's pose distribution is strongly bimodal and a rank measure is the robustness check.

EXACT SEED SET enforced per dataset, as in scripts/plot_dockq_variance_hist.py: an interface missing
any of its 50 seeds, or any of the 5 samples in a seed, is dropped and reported.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/plot_jac_dockq_corr.py
Writes artifacts/jac_dockq_corr.png
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
FB_SEEDS = frozenset(range(101, 151))
PX_SEEDS = frozenset(range(300000, 300050))
N_SAMPLE = 5

DATASETS = [
    ("FoldBench", FB_SEEDS,
     f"{EXP}/artifacts/dockq_per_pose_foldbench.csv",
     lambda r: (r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"])))),
     f"{EXP}/artifacts/epitope_overlap.csv", f"{EXP}/artifacts/paratope_overlap.csv"),
    ("PXM-2025", PX_SEEDS,
     f"{P}/dockq_pxm2025.csv",
     lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"])))),
     f"{P}/epitope_overlap_pxm.csv", f"{P}/paratope_overlap_pxm.csv"),
]
COLS = {"epitope": "#b3452c", "paratope": "#8c5a2b"}


def read(path, key, field, seeds):
    out = {}
    for r in csv.DictReader(open(path)):
        s = int(r["seed"])
        if s in seeds and r.get(field):
            out[(key(r), s, int(r["sample"]))] = float(r[field])
    return out


def main():
    jkey = lambda r: (r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))))
    results = {}
    for name, seeds, dqp, dqkey, epip, parap in DATASETS:
        dq = read(dqp, dqkey, "dockq", seeds)
        epi = read(epip, jkey, "jaccard", seeds)
        para = read(parap, jkey, "jaccard", seeds)
        for side, jac in (("epitope", epi), ("paratope", para)):
            g = collections.defaultdict(list)
            for k, v in jac.items():
                if k in dq:
                    g[k[0]].append((v, dq[k]))
            rows, flat = [], 0
            for iface, pairs in g.items():
                if len(pairs) != len(seeds) * N_SAMPLE:
                    continue
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                if x.std() < 1e-9 or y.std() < 1e-9:
                    flat += 1
                    continue
                rows.append((iface, pearsonr(x, y)[0], spearmanr(x, y)[0]))
            results[(name, side)] = (rows, flat)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    bins = np.linspace(-1, 1, 41)
    print(f"{'set':<12}{'side':<10}{'n':>5}{'flat':>6}{'median r':>10}"
          f"{'mean r':>9}{'median rho':>12}{'r>0.5':>8}{'r<0':>7}")
    for i, (name, _, _, _, _, _) in enumerate(DATASETS):
        for j, side in enumerate(("epitope", "paratope")):
            rows, flat = results[(name, side)]
            r = np.array([x[1] for x in rows])
            rho = np.array([x[2] for x in rows])
            ax = axes[j][i]
            ax.hist(r, bins=bins, color=COLS[side], alpha=0.55)
            ax.hist(r, bins=bins, histtype="step", lw=1.8, color=COLS[side])
            ax.axvline(0, color="0.6", lw=1.0)
            ax.axvline(np.median(r), color="0.20", lw=1.5, ls="--",
                       label=f"median {np.median(r):.3f}")
            ax.axvline(r.mean(), color="0.20", lw=1.5, ls="-", label=f"mean {r.mean():.3f}")
            ax.set_title(f"{side} Jaccard vs DockQ  --  {name}\n"
                         f"n={len(r)} interfaces ({flat} dropped, no variation)", fontsize=12)
            ax.set_xlabel("Pearson r within the interface, across its 250 poses", fontsize=12)
            ax.set_ylabel("number of interfaces", fontsize=12)
            ax.set_xlim(-1, 1)
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=11, loc="upper left")
            print(f"{name:<12}{side:<10}{len(r):>5}{flat:>6}{np.median(r):>10.3f}"
                  f"{r.mean():>9.3f}{np.median(rho):>12.3f}"
                  f"{100*(r>0.5).mean():>7.0f}%{100*(r<0).mean():>6.0f}%")
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/jac_dockq_corr.png", dpi=150)
    print("\nwrote artifacts/jac_dockq_corr.png")


if __name__ == "__main__":
    main()
