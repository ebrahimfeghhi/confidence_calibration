#!/usr/bin/env python
"""Joint distribution of DockQ and CDR H3 RMSD, pooled over interfaces.

Reproduces the layout of Figure 2 in Hitawala & Gray (mAbs 2025) on our Protenix data, for BOTH RMSD
variants: local (H3 superposed on itself -> loop conformation) and global (heavy-chain framework
superposed, then H3 measured -> conformation plus placement). The paper does not state which variant
its figure uses, so both are shown.

One point per (pose, interface). H3 is a complex-level quantity, so complexes contributing two
labelled interfaces contribute the same H3 value twice -- that is what "pooled across interfaces"
means here, and it is why the marginal on the x axis is not a sample of independent complexes.

CAPRI bands: DockQ 0.23 acceptable, 0.49 medium, 0.80 high.
"""
import csv, glob, json, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))

h3 = {}
for r in csv.DictReader(open("artifacts/cdr_h3_rmsd_ost.csv")):
    s = int(r["seed"])
    if r["pdb"] in KEEP and 101 <= s <= 150:
        h3[(r["pdb"], s, int(r["sample"]))] = (float(r["h3_local"]), float(r["h3_global"]))

pts = []          # (h3_local, h3_global, dockq)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m:
        continue
    p, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if p not in KEEP or not (101 <= s <= 150):
        continue
    hv = h3.get((p, s, k))
    if hv is None:
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    q, ifs = j.get("dockq") or [], j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        for i, t in enumerate(ifs):
            if i < len(q) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                pts.append((hv[0], hv[1], q[i]))
                break
pts = np.array(pts)
print(f"{len(pts)} (pose, interface) points")

XMAX = 11.0
for idx, (name, lab) in enumerate([(0, "local"), (1, "global")]):
    x, y = pts[:, name], pts[:, 2]
    keep = x <= XMAX
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    # shaded regions, as in the reference figure
    ax.axvspan(0, 1.0, color="#f7c9de", alpha=.55, zorder=0)
    ax.add_patch(plt.Rectangle((1.0, 0.80), XMAX, 0.24, color="#bfe3d0", alpha=.55, zorder=0))
    ax.scatter(x[keep], y[keep], s=7, alpha=.16, c="#3b5f9e", edgecolors="none", zorder=2)
    hi = (x <= 1.0)
    ax.scatter(x[hi], y[hi], s=7, alpha=.22, c="#6b4fa1", edgecolors="none", zorder=3)
    for t in (0.23, 0.49, 0.80):
        ax.axhline(t, ls="--", lw=1.1, color="#222", zorder=4)
    ax.axvline(1.0, lw=1.1, color="#222", zorder=4)
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel(f"CDR H3 {lab} RMSD (Å)", fontsize=12)
    ax.set_ylabel("DockQ", fontsize=12)
    for yy, txt, col in ((0.90, "High accuracy\ndocking", "#1d6b45"),
                         (0.63, "Medium accuracy\ndocking", "#2b3f8c"),
                         (0.35, "Acceptable accuracy\ndocking", "#8a7420"),
                         (0.10, "Incorrect\ndocking", "#b3543f")):
        ax.text(XMAX * 0.62, yy, txt, fontsize=9.5, color=col, ha="center", va="center",
                bbox=dict(fc="white", ec=col, alpha=.75, boxstyle="round,pad=0.3"))
    ax.annotate("<1.0 Å", xy=(1.0, 0.32), xytext=(2.15, 0.32), fontsize=9.5, color="#c2185b",
                va="center", arrowprops=dict(arrowstyle="->", color="#c2185b", lw=1.2),
                bbox=dict(fc="white", ec="#c2185b", alpha=.85, boxstyle="round,pad=0.25"))
    axt.hist(x[keep], bins=44, color="#7d95c9", edgecolor="white", linewidth=.4)
    axr.hist(y[keep], bins=30, orientation="horizontal", color="#7d95c9",
             edgecolor="white", linewidth=.4)
    for a in (axt, axr):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    n_hi = int(((x <= 1.0) & (y >= 0.80)).sum()); n_dq = int((y >= 0.80).sum())
    n_lo = int((x <= 1.0).sum())
    fig.suptitle(f"DockQ vs CDR H3 {lab} RMSD — {len(x)} poses, 161 interfaces\n"
                 f"P(H3≤1Å | DockQ≥0.8) = {n_hi/max(n_dq,1):.1%}   "
                 f"P(DockQ≥0.8 | H3≤1Å) = {n_hi/max(n_lo,1):.1%}", fontsize=11.5, y=0.985)
    fig.savefig(f"artifacts/fig_h3_{lab}_vs_dockq.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote artifacts/fig_h3_{lab}_vs_dockq.png")
