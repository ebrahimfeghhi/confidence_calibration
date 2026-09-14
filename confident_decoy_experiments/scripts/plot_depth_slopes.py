#!/usr/bin/env python
"""Slopegraph: oracle and selected DockQ at 250 poses vs at maximum depth, for the 22 deep-run interfaces.

The point of the pair of panels is the contrast: depth lifts the ORACLE almost everywhere while
SELECTED barely moves. Both panels share a y-axis so the two slopes are directly comparable -- with
independent axes the selected panel would look far more dynamic than it is.

N is POSES (5 diffusion samples per seed), so 11745 poses = 2349 seeds against the reference's 50.

Writes artifacts/depth_slopes.png
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ALL = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))

rows = []
for f in sorted(glob.glob(f"{EXP}/artifacts/free1k*_scores.json")) + \
         sorted(glob.glob(f"{EXP}/artifacts/freedeep*_scores.json")):
    for i, d in json.load(open(f)).items():
        b = ALL.get(i, {})
        rows.append(dict(iface=i, status=b.get("status", "?"),
                         sel0=b.get("sel_dockq"), orc0=b.get("best_dockq"),
                         N=d.get("new_poses"), sel1=d.get("new_sel_dockq"),
                         orc1=d.get("new_best")))
rows = [r for r in rows if None not in (r["sel0"], r["orc0"], r["sel1"], r["orc1"])]
print(f"{len(rows)} interfaces")

COL = {"confident_decoy": "#2E6E8E", "no_acceptable_pose": "#9A3B2E", "solved": "#3F7A56"}
fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

for ax, (k0, k1, title) in zip(axes, [("orc0", "orc1", "Oracle DockQ  (best pose in pool)"),
                                      ("sel0", "sel1", "Selected DockQ  (top pair_ipTM pose)")]):
    for r in rows:
        c = COL.get(r["status"], "#888")
        ax.plot([0, 1], [r[k0], r[k1]], "-o", color=c, alpha=0.75, lw=1.4, ms=4.5)
    ax.axhline(0.23, color="#555", ls="--", lw=1, zorder=0)
    ax.text(1.02, 0.235, "acceptable (0.23)", fontsize=8, color="#555", va="bottom")
    ax.set_xlim(-0.18, 1.30); ax.set_xticks([0, 1])
    ax.set_xticklabels(["250 poses\n(50 seeds)", "max depth\n(3.6k-11.7k poses)"], fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

axes[0].set_ylabel("DockQ", fontsize=11)
axes[0].set_ylim(-0.02, 1.02)

# label the interfaces that actually move, on whichever panel they move in
for r in rows:
    if r["orc1"] - r["orc0"] > 0.25:
        axes[0].annotate(r["iface"], (1, r["orc1"]), xytext=(4, 0),
                         textcoords="offset points", fontsize=7.5, va="center", color="#333")
    if r["sel1"] - r["sel0"] > 0.15:
        axes[1].annotate(r["iface"], (1, r["sel1"]), xytext=(4, 0),
                         textcoords="offset points", fontsize=7.5, va="center", color="#333")

from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([], [], color=COL["confident_decoy"], marker="o", ls="-",
                               label="confident decoy (13)"),
                        Line2D([], [], color=COL["no_acceptable_pose"], marker="o", ls="-",
                               label="no acceptable pose (9)")],
               loc="upper left", fontsize=9, frameon=False)

import statistics as st
fig.suptitle("Depth lifts what the model can generate, not what it selects\n"
             f"mean oracle {st.mean(r['orc0'] for r in rows):.3f} → "
             f"{st.mean(r['orc1'] for r in rows):.3f}   |   "
             f"mean selected {st.mean(r['sel0'] for r in rows):.3f} → "
             f"{st.mean(r['sel1'] for r in rows):.3f}   |   "
             f"rescued 0/22 → 2/22",
             fontsize=12.5, y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = f"{EXP}/artifacts/depth_slopes.png"
fig.savefig(out, dpi=160)
print("wrote", out)
