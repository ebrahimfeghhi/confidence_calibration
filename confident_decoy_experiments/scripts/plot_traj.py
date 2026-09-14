#!/usr/bin/env python
"""Epitope trajectory: how close each seed is to the CRYSTAL epitope at every recycle.

One line per seed, x = recycle, y = jaccard of the decoded pose's epitope to the crystal epitope.
A line that rises then falls is a trajectory that FOUND the right site and then left it.
Seeds were chosen to be maximally diverse, so most start and stay near zero by construction.
"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {r["interface"]: r for r in json.load(open("artifacts/traj_results.json"))}
CLS = {}
for i in json.load(open("artifacts/traj20.json")):
    CLS[i] = None
import csv
P = {r["interface"]: r for r in csv.DictReader(open("artifacts/abag_selection_protocol.csv"))}
for i in CLS:
    r = P.get(i)
    if r:
        sel, orc = float(r["sel50_dockq"]), float(r["oracle_dockq"])
        CLS[i] = "failure" if orc < .23 else "confident decoy" if sel < .23 else "solved"

want = [i for i in sys.argv[1:]] or sorted(R)
n = len(want)
# two rows: A = best-of-5 jaccard to the crystal epitope, B = spread AMONG the 5 samples of the same
# recycle (same trunk state, different diffusion draws) -- i.e. how much of the movement in A is
# just sampling noise.
fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7.2), squeeze=False)
for ax, iface in zip(axes[0], want):
    tr = R[iface]["nat_trace"]
    for seed, v in sorted(tr.items(), key=lambda kv: -max(kv[1])):
        v = np.array(v)
        col = "#2f6d54" if v[-1] >= .5 else "#c1553a" if v.max() >= .5 else "#9a938a"
        ax.plot(range(len(v)), v, marker="o", ms=3.5, lw=1.5, color=col, alpha=.85)
    ax.set_ylim(-.03, 1.03); ax.set_xlim(-.3, 9.3); ax.set_xticks(range(10))
    ax.grid(alpha=.2)
    flat = sum(1 for v in tr.values() if max(v) == 0)
    ax.set_title(f"{iface}  ({CLS.get(iface,'?')})\n{flat}/{len(tr)} seeds flat at zero throughout",
                 fontsize=10, loc="left", pad=7)
    ax.set_xlabel("recycle", fontsize=9, labelpad=5)
    ax.set_ylabel("best-of-5 jaccard to crystal epitope", fontsize=9)
for ax, iface in zip(axes[1], want):
    st = R[iface].get("spread_trace", {})
    for seed, v in sorted(st.items()):
        v = np.array([np.nan if x is None else x for x in v], float)
        ax.plot(range(len(v)), v, marker="o", ms=3.5, lw=1.4, color="#4a6fa5", alpha=.75)
    ax.set_ylim(-.03, 1.03); ax.set_xlim(-.3, 9.3); ax.set_xticks(range(10))
    ax.grid(alpha=.2)
    ax.set_title("disagreement among the 5 samples of the SAME recycle", fontsize=9.5,
                 loc="left", pad=7)
    ax.set_xlabel("recycle", fontsize=9, labelpad=5)
    ax.set_ylabel("mean pairwise jaccard distance", fontsize=9)
h = [plt.Line2D([], [], color="#2f6d54", lw=2, label="ends on the native epitope"),
     plt.Line2D([], [], color="#c1553a", lw=2, label="visits it, then leaves"),
     plt.Line2D([], [], color="#9a938a", lw=2, label="never finds it")]
fig.legend(handles=h, fontsize=9.5, frameon=False, loc="lower left",
           bbox_to_anchor=(0.01, 1.0), ncol=3)
fig.tight_layout()
out = "artifacts/traj_epitope.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
