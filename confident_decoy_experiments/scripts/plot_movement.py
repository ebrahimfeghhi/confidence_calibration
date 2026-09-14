#!/usr/bin/env python
"""Is the epitope moving between recycles, or is that just diffusion sampling?

Both curves are the SAME quantity -- mean Jaccard distance between two decoded epitopes -- and
differ only in whether the trunk advanced between the two draws:
  within-loop   the 5 samples of recycle k against each other      (pure diffusion sampling)
  across-loop   the 5 samples of recycle k against the 5 of k+1    (sampling + one trunk step)
If the curves sit on top of each other, recycling is not relocating the epitope and the apparent
"hopping" is a sampling artefact. A gap is the part attributable to the trunk.

Usage: .../python scripts/plot_movement.py 7txw_AC 7txw_BC 7yxu_BA 7yxu_CA
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W, A = "#3b6ea5", "#c0562c"


def main():
    want = sys.argv[1:]
    res = {x["interface"]: x for x in json.load(open("artifacts/traj_results.json"))}
    ifs = [i for i in want if i in res and res[i].get("move_trace")]
    if not ifs:
        raise SystemExit("no interfaces with move_trace; re-run traj_analyze.py")

    fig, axes = plt.subplots(1, len(ifs), figsize=(4.2 * len(ifs), 4.6), squeeze=False)
    for ax, iface in zip(axes[0], ifs):
        x = res[iface]
        sp = np.array([[v for v in t] for t in x["spread_trace"].values()], dtype=float)
        mv = np.array([[v for v in t] for t in x["move_trace"].values()], dtype=float)
        ks = np.arange(sp.shape[1])
        # across-loop values sit between recycles; plot at k+0.5
        for m, c, lab, xs in ((sp, W, "within loop (5 samples, same trunk state)", ks),
                              (mv, A, "across loops (k vs k+1, all 5x5 pairs)", ks[:-1] + 0.5)):
            mu, sd = np.nanmean(m, 0), np.nanstd(m, 0)
            ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
            ax.plot(xs, mu, color=c, lw=2, marker="o", ms=4, label=lab)
        gap = np.nanmean(mv) - np.nanmean(sp)
        ax.set_title(f'{iface}   n={x["n_seeds"]} seeds\n'
                     f'within {np.nanmean(sp):.2f}   across {np.nanmean(mv):.2f}   '
                     f'gap {gap:+.2f}', fontsize=10)
        ax.set_xlabel("recycle")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("mean Jaccard distance\nbetween two decoded epitopes")
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=[0.015, 0, 1, 0.80])
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("epitope movement: how much of it survives holding the trunk fixed?",
                 y=0.975, fontsize=12)
    fig.savefig("artifacts/traj_movement.png", dpi=150)
    print("wrote artifacts/traj_movement.png")


if __name__ == "__main__":
    main()
