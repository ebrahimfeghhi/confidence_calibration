#!/usr/bin/env python
"""Per interface: the best pose available vs the pose ipTM actually ships, at 50 seeds.

x = oracle DockQ  (best of the interface's 250 poses)
y = ipTM-selected DockQ (the single pose pooled ipTM ranks first)
One point per interface. y can never exceed x, so every point lies on or below the diagonal, and the
VERTICAL DROP from the diagonal is exactly what selection loses on that interface.

THE QUADRANT THAT MATTERS is bottom-right: oracle >= 0.23 but selected < 0.23. Those are interfaces
where an acceptable pose was generated and then not chosen -- the selection gap, pose by pose, rather
than as a difference of two averages. Interfaces on the left of the vertical threshold line never had
an acceptable pose at all; no selector could have helped them, and they are a COVERAGE problem.

Reading the two failure modes off one plot is the point: coverage failures move points left, selection
failures move points down.

Sources: artifacts/abag_selection_protocol.csv, artifacts/pxm2025/pxm2025_selection_protocol.csv,
artifacts/pp_selection_protocol.csv. FoldBench/pp are OST DockQ, PXM is PXMeter's own, so compare the
SHAPES rather than absolute positions.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/plot_oracle_vs_selected.py
Writes artifacts/oracle_vs_selected.png
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
T = 0.23
SETS = [
    ("FoldBench ab-ag", "#b3452c", f"{EXP}/artifacts/abag_selection_protocol.csv"),
    ("PXM-2025 ab-ag", "#31688e", f"{EXP}/artifacts/pxm2025/pxm2025_selection_protocol.csv"),
    ("FoldBench protein-protein", "#3f7d4e", f"{EXP}/artifacts/pp_selection_protocol.csv"),
]

# --mean swaps the x axis from ORACLE (max over the 250 poses) to the MEAN over them.
# The distinction matters: oracle asks "does a good pose exist", the mean is a proxy for how MANY
# there are. Selection at a fixed percentile is driven by the second, so the mean is the better
# control -- an interface with oracle 0.7 and 40% of poses acceptable behaves nothing like one with
# oracle 0.7 and a single lucky pose. NOTE the y <= x bound and the diagonal only apply to the oracle
# version; a selected pose can and does beat its pool's mean.
POSE_SRC = {
    "FoldBench ab-ag": (f"{EXP}/artifacts/dockq_per_pose_foldbench.csv",
                        lambda r: (r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"])))),
                        frozenset(range(101, 151))),
    "PXM-2025 ab-ag": (f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv",
                       lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"])))),
                       frozenset(range(300000, 300050))),
    "FoldBench protein-protein": (f"{EXP}/artifacts/pp/dockq_pp.csv",
                                  lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"])))),
                                  frozenset(range(101, 151))),
}


def pool_means(name):
    """{interface_key: mean DockQ over its poses} on the exact seed set."""
    path, key, seeds = POSE_SRC[name]
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if int(r["seed"]) in seeds and r.get("dockq"):
            g[key(r)].append(float(r["dockq"]))
    return {k: np.mean(v) for k, v in g.items() if len(v) >= 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean", action="store_true", help="x = mean DockQ over poses, not the max")
    ap.add_argument("--no-pp", action="store_true")
    A = ap.parse_args()
    sets = [s for s in SETS if not (A.no_pp and "protein-protein" in s[0])]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 5.9), squeeze=False)
    axes = axes[0]
    print(f"{'set':<28}{'n':>5}{'sel ok':>9}{'oracle ok':>11}"
          f"{'GENERATED NOT PICKED':>22}{'never generated':>17}{'median drop':>13}")
    for ax, (name, col, path) in zip(axes, sets):
        r = list(csv.DictReader(open(path)))
        if A.mean:
            pm = pool_means(name)
            keyof = lambda v: (v["interface"].rsplit("_", 1)[0],
                               tuple(sorted(v["interface"].rsplit("_", 1)[1])))
            r = [v for v in r if keyof(v) in pm]
            x = np.array([pm[keyof(v)] for v in r])
        else:
            x = np.array([float(v["oracle_dockq"]) for v in r])
        y = np.array([float(v["sel50_dockq"]) for v in r])
        miss = (x >= T) & (y < T)          # acceptable pose existed, selector missed it
        nocov = x < T                      # no acceptable pose anywhere
        hit = (x >= T) & (y >= T)

        if not A.mean:
            ax.plot([0, 1], [0, 1], color="0.6", lw=1.1, ls="-", zorder=1)
        ax.axhline(T, color="0.75", lw=1.0, ls="--", zorder=1)
        ax.axvline(T, color="0.75", lw=1.0, ls="--", zorder=1)
        ax.scatter(x[hit], y[hit], s=26, c=col, alpha=0.75, lw=0, zorder=3,
                   label=f"shipped acceptable  ({hit.sum()})")
        ax.scatter(x[miss], y[miss], s=34, facecolors="none", edgecolors="#c0392b", lw=1.3,
                   zorder=4, label=f"generated, NOT picked  ({miss.sum()})")
        ax.scatter(x[nocov], y[nocov], s=22, c="0.6", alpha=0.7, lw=0, zorder=2,
                   label=(f"pool mean < 0.23  ({nocov.sum()})" if A.mean
                          else f"never generated  ({nocov.sum()})"))
        ax.set_title(f"{name}   (n={len(r)})", fontsize=13)
        ax.set_xlabel("mean DockQ over the 250 poses" if A.mean
                      else "oracle DockQ  (best of 250 poses)", fontsize=12)
        ax.set_ylabel("ipTM-selected DockQ", fontsize=12)
        ax.set_xlim(-0.03, (1.03 if not A.mean else max(0.6, x.max() * 1.08)))
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        print(f"{name:<28}{len(r):>5}{100*(y>=T).mean():>8.0f}%{100*(x>=T).mean():>10.0f}%"
              f"{miss.sum():>14} ({100*miss.mean():>3.0f}%){nocov.sum():>10} "
              f"({100*nocov.mean():>3.0f}%){np.median(x - y):>13.3f}")
    fig.tight_layout()
    out = f"{EXP}/artifacts/{'poolmean' if A.mean else 'oracle'}_vs_selected.png"
    fig.savefig(out, dpi=150)
    print("\n'GENERATED NOT PICKED' = oracle >= 0.23 but selected < 0.23 (bottom-right quadrant):")
    print("the interfaces a perfect selector would rescue with no extra compute.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
