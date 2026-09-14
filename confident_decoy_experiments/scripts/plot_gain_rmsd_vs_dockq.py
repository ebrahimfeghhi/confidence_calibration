#!/usr/bin/env python
"""Does seed scaling buy DockQ by improving the ANTIBODY's own geometry?

Companion to scripts/plot_gain_epitope_vs_dockq.py, same construction, different x:
  x  REDUCTION in RMSD = (median RMSD among the 50 per-seed ipTM picks) - (50-seed ipTM pick)
  y  gain in DockQ     = (50-seed ipTM pick) - (median DockQ among those 50 picks)

SIGN. RMSD is better when SMALLER, so x is plotted as a reduction rather than a change. That keeps
"right and up = scaling helped" identical to the epitope/paratope Jaccard figures; a point at x<0
means the shipped pose has a WORSE antibody geometry than a typical single seed.

Both RMSD families are measured by superposing WITHIN the antibody, so neither depends on whether
the antibody is docked correctly -- they ask "is the antibody built right", not "is it placed right".
  paratope = antibody residues within 5 A of the antigen IN THE GROUND TRUTH (a fixed residue set per
             interface, so poses that dock elsewhere are still scored on the same patch)
  CDR H3   = Chothia-numbered H3 loop, heavy chain only; light-chain interface rows have no H3 and
             are dropped, hence the smaller n on those panels
  local    = superpose the patch/loop on itself -> conformation only
  global   = superpose the whole antibody chain -> conformation plus placement on the framework

Usage: .../python scripts/plot_gain_rmsd_vs_dockq.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
COL = "#b3452c"
PANELS = [("paratope RMSD, local", "para_local"), ("paratope RMSD, global", "para_global"),
          ("CDR H3 RMSD, local", "h3_local"), ("CDR H3 RMSD, global", "h3_global"),
          # epitope RMSD carried in BOTH versions: the two source tables disagree on 13k poses and
          # neither is provably canonical, so any claim must hold under both
          ("epitope RMSD local (v1)", "epirm_local"),
          ("epitope RMSD global (v1)", "epirm_global"),
          ("epitope RMSD local (v2)", "epirm2_local"),
          ("epitope RMSD global (v2)", "epirm2_global")]
GROUPS = {"epirmsd": ["epirm_local", "epirm2_local", "epirm_global", "epirm2_global"]}


def main():
    import sys
    # optional arg = one panel key (e.g. para_local) -> single untitled figure matching the style of
    # scripts/plot_gain_epitope_vs_dockq.py, for putting the two side by side
    only = sys.argv[1] if len(sys.argv) > 1 else None
    # second arg swaps the y-axis: "dockq" (default) or "epi" = gain in epitope Jaccard, which asks
    # whether a better-built antibody comes with landing on a better antigen surface
    yv = sys.argv[2] if len(sys.argv) > 2 else "dockq"
    ylab = ("gain in DockQ" if yv == "dockq" else "gain in epitope Jaccard")
    ykey = ("sel1_dockq", "sel50_dockq") if yv == "dockq" else ("sel1med_epi_jac", "sel50_epi_jac")
    if only in GROUPS:
        panels = [p for p in PANELS if p[1] in GROUPS[only]]
        panels.sort(key=lambda p: GROUPS[only].index(p[1]))
    else:
        panels = [p for p in PANELS if p[1] == only] if only else PANELS
    r = list(csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")))
    fig, axes = plt.subplots(1, len(panels),
                             figsize=((7.6, 7.2) if len(panels) == 1
                                      else (5.4 * len(panels), 6.2)), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (title, key) in zip(axes, panels):
        xs, ys = [], []
        for v in r:
            a, b, d1, d50 = (v.get(f"sel1med_{key}"), v.get(f"sel50_{key}"),
                             v[ykey[0]], v[ykey[1]])
            if not a or not b or not d1 or not d50:
                continue
            xs.append(float(a) - float(b))          # reduction: positive = improved
            ys.append(float(d50) - float(d1))
        x, y = np.array(xs), np.array(ys)
        ax.axhline(0, color="0.75", lw=0.9, zorder=0)
        ax.axvline(0, color="0.75", lw=0.9, zorder=0)
        ax.scatter(x, y, s=30, c=COL, alpha=0.72, lw=0, zorder=3)
        m, b = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        gx = np.linspace(x.min(), x.max(), 50)
        ax.plot(gx, m * gx + b, color="0.25", lw=1.4, zorder=4,
                label=f"slope {m:.3f},  r$^2$ = {r2:.3f}")
        if len(panels) > 1:
            ax.set_title(f"{title}\nn = {len(x)} interfaces", fontsize=14)
        ax.set_xlabel(f"reduction in {title} (Å)\n"
                      "(median of the 50 per-seed picks  -  50-seed ipTM pick)",
                      fontsize=15 if len(panels) == 1 else 13)
        ax.tick_params(labelsize=13 if len(panels) == 1 else 12)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", frameon=False, fontsize=14 if len(panels) == 1 else 12)
        print(f"{title:<24} n={len(x):>4}  slope {m:+.3f}  r2 {r2:.3f}  "
              f"mean reduction {x.mean():+.3f} A")
    axes[0].set_ylabel(f"{ylab}\n(50-seed ipTM pick  -  median of the 50 per-seed picks)",
                       fontsize=15 if len(panels) == 1 else 13)
    fig.tight_layout()
    out = (f"gain_rmsd_{only}_vs_{yv}.png" if only else f"gain_rmsd_vs_{yv}.png")
    fig.savefig(f"{EXP}/artifacts/{out}", dpi=150)
    print(f"wrote artifacts/{out}")


if __name__ == "__main__":
    main()
