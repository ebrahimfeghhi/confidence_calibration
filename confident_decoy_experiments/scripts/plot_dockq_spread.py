#!/usr/bin/env python
"""How spread out is DockQ across an interface's 250 poses, and does that depend on its level?

One point per interface. Left panel x = MEDIAN DockQ over all of that interface's poses, right panel
x = MEAN over the same poses; y on both = VARIANCE of DockQ over those poses. No selection and no
oracle anywhere -- this is the raw pose distribution.

WHY THE BOUND MATTERS. DockQ is confined to [0, 1], so its variance is not free of its level: for any
variable on [0, 1], var <= mean * (1 - mean), with equality only when all the mass sits at 0 and 1.
That parabola is drawn on the MEAN panel, where it is exactly valid. It is deliberately NOT drawn on
the median panel -- the bound is a statement about the mean, and median and mean come apart badly here
precisely because these distributions are bimodal. An interface hugging the parabola is one whose
poses are all either failures or successes with nothing in between, which is the shape a
basin-selection process produces; an interface far below it has poses of genuinely intermediate
quality.

TWO DATASETS, two DockQ implementations: FoldBench is OST, PXM-2025 is PXMeter's own. Absolute values
are therefore not strictly comparable between them, but the SHAPE of the variance-vs-level relation is.

Sources, both per-pose and traceable:
  artifacts/dockq_per_pose_foldbench.csv     (scripts/dump_dockq_per_pose.py, from the OST jsons)
  artifacts/pxm2025/dockq_pxm2025.csv        (scripts/score_pxm2025.py, from PXMeter)
Deliberately NOT artifacts/dockq_sd_data.npz: no script in the repo produces it and git has no record,
so its pose set is unknown.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/plot_dockq_spread.py [--min-poses 100]
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
FB, PX, PP = "#b3452c", "#31688e", "#3f7d4e"


def load(path, key, dockq="dockq"):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        v = r.get(dockq)
        if v in (None, ""):
            continue
        g[key(r)].append(float(v))
    return g


# DockQ's two RMSD terms are turned into [0,1] scores by DockQ itself before averaging:
#   f(x) = 1 / (1 + (x/d)^2),   d = 1.5 A for iRMSD, 8.5 A for LRMSD
# and DockQ = (fnat + f(iRMSD) + f(LRMSD)) / 3. Plotting the raw RMSDs would be meaningless against
# the m(1-m) bound, which only applies on [0,1]; plotting the scaled versions puts all three
# components on the same footing as DockQ. Verified: recombining the extracted components reproduces
# the reported DockQ to 0.0e+00 over 37,000 PXM poses, so these are the exact transforms in use.
SCALE = {"irmsd": 1.5, "lrmsd": 8.5}


def bounded(name, v):
    d = SCALE.get(name)
    return v if d is None else 1.0 / (1.0 + (v / d) ** 2)


def components(A):
    import collections as C
    # FoldBench components sit in the per-pose dump; PXM's come from the pose jsons (the scorer's
    # csv never carried them). PXM's component file covers EVERY interface in each assembly, so it is
    # restricted to the ab-ag interfaces that dockq_pxm2025.csv actually scored.
    keep = {(r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))), r["seed"], r["sample"])
            for r in csv.DictReader(open(f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv"))}
    src = [
        ("FoldBench (OST)", FB, f"{EXP}/artifacts/dockq_per_pose_foldbench.csv",
         lambda r: (r["pdb"], r["ab_chain"], r["ag_chain"]), None),
        ("PXM-2025 (PXMeter)", PX, f"{EXP}/artifacts/pxm2025/dockq_components_pxm2025.csv",
         lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"])))),
         lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))), r["seed"], r["sample"])
         in keep),
        ("FoldBench prot-prot", PP, f"{EXP}/artifacts/pp/dockq_pp.csv",
         lambda r: (r["pdb"], r["chain1"], r["chain2"]), None),
    ]
    metrics = [("fnat", "fnat"), ("irmsd", "f(iRMSD),  d = 1.5 A"),
               ("lrmsd", "f(LRMSD),  d = 8.5 A")]

    fig, axes = plt.subplots(len(src), 3, figsize=(17.4, 5.3 * len(src)))
    print(f"one point per interface, >= {A.min_poses} poses each\n")
    print(f"{'set':<20}{'metric':<10}{'n':>5}{'mean':>9}{'mean var':>11}"
          f"{'max var':>10}{'% of bound':>12}")
    for row, (name, col, path, key, keepfn) in enumerate(src):
        g = C.defaultdict(lambda: C.defaultdict(list))
        for r in csv.DictReader(open(path)):
            if keepfn and not keepfn(r):
                continue
            for m, _ in metrics:
                if r.get(m):
                    g[m][key(r)].append(bounded(m, float(r[m])))
        for j, (m, lab) in enumerate(metrics):
            ax = axes[row][j]
            kept = {k: np.array(v) for k, v in g[m].items() if len(v) >= A.min_poses}
            c = np.array([v.mean() for v in kept.values()])
            var = np.array([v.var() for v in kept.values()])
            ax.scatter(c, var, s=30, c=col, alpha=0.7, lw=0)
            mm = np.linspace(0, 1, 200)
            ax.plot(mm, mm * (1 - mm), color="0.35", lw=1.3, ls="--")
            ax.set_title(f"{name}  --  {lab}   (n={len(kept)})", fontsize=12)
            ax.set_xlabel(f"mean {lab.split(',')[0]} across the interface's poses", fontsize=12)
            ax.set_ylabel(f"variance of {lab.split(',')[0]}", fontsize=12)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.012, 0.27)
            ax.tick_params(labelsize=11)
            ax.grid(alpha=0.2)
            frac = np.median(var / np.maximum(c * (1 - c), 1e-9))
            print(f"{name:<20}{m:<10}{len(kept):>5}{c.mean():>9.3f}{var.mean():>11.4f}"
                  f"{var.max():>10.4f}{100*frac:>11.0f}%")
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/dockq_spread_components.png", dpi=150)
    print("\nwrote artifacts/dockq_spread_components.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-poses", type=int, default=100)
    ap.add_argument("--components", action="store_true",
                    help="plot the three DockQ COMPONENTS instead of DockQ itself")
    A = ap.parse_args()
    if A.components:
        return components(A)

    sets = [
        ("FoldBench (OST DockQ)", FB,
         load(f"{EXP}/artifacts/dockq_per_pose_foldbench.csv",
              lambda r: (r["pdb"], r["ab_chain"], r["ag_chain"]))),
        ("PXM-2025 (PXMeter DockQ)", PX,
         load(f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv",
              lambda r: (r["pdb"], r["chain1"], r["chain2"]))),
        # protein-protein, the contrast class: no antibody, so no epitope-search story
        ("FoldBench protein-protein (OST DockQ)", PP,
         load(f"{EXP}/artifacts/pp/dockq_pp.csv",
              lambda r: (r["pdb"], r["chain1"], r["chain2"]))),
    ]

    # One ROW per dataset so the two are not overplotted, columns = median / mean.
    fig, axes = plt.subplots(len(sets), 2, figsize=(13.0, 5.2 * len(sets)))
    stats = []
    for row, (name, col, g) in enumerate(sets):
        kept = {k: np.array(v) for k, v in g.items() if len(v) >= A.min_poses}
        var = np.array([v.var() for v in kept.values()])
        for j, centre in enumerate(("median", "mean")):
            ax = axes[row][j]
            c = np.array([np.median(v) if centre == "median" else v.mean()
                          for v in kept.values()])
            ax.scatter(c, var, s=34, c=col, alpha=0.72, lw=0)
            if centre == "mean":
                m = np.linspace(0, 1, 200)
                ax.plot(m, m * (1 - m), color="0.35", lw=1.4, ls="--",
                        label="var = m(1-m), the [0,1] bound")
                ax.legend(frameon=False, fontsize=11, loc="upper right")
                stats.append((name, len(kept), c, var,
                              np.array([len(v) for v in kept.values()])))
            ax.set_title(f"{name}   (n={len(kept)})", fontsize=13)
            ax.set_xlabel(f"{centre} DockQ across the interface's poses", fontsize=13)
            ax.set_ylabel("variance of DockQ", fontsize=13)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.012, 0.27)
            ax.tick_params(labelsize=11)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/dockq_spread.png", dpi=150)

    print(f"one point per interface, >= {A.min_poses} poses each\n")
    print(f"{'set':<26}{'n':>5}{'poses/iface':>13}{'mean DockQ':>12}"
          f"{'mean var':>10}{'max var':>9}{'% of bound':>12}")
    for name, n, c, var, np_ in stats:
        frac = var / np.maximum(c * (1 - c), 1e-9)
        print(f"{name:<26}{n:>5}{int(np.median(np_)):>13}{c.mean():>12.3f}"
              f"{var.mean():>10.4f}{var.max():>9.4f}{100*np.median(frac):>11.0f}%")
    print("\n'% of bound' is the MEDIAN over interfaces of var / [m(1-m)] -- how close a typical "
          "interface\nsits to the all-poses-are-0-or-1 extreme.")
    print("\nwrote artifacts/dockq_spread.png")


if __name__ == "__main__":
    main()
