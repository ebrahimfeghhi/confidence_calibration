"""Trajectory shape: do future-G seeds go DEEPER into B1 and then snap out
(commit-then-correct), while B1 seeds steadily deepen? Measures each seed's
W1 distance to the committed B1 / G / B2 centroids (loop-16 group means) at
EVERY loop, from saved pairformer_z. Labels are DIAGNOSTIC only (describe the
shape), not used by any algorithm.

Saves CSV + PNG only. Run: CUDA_VISIBLE_DEVICES=N ESM_PYTHON ... trajectory_depth.py
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import load, cross_distogram_from_z, w1, GOOD, N_LOOPS, REPS  # noqa: E402
from basins import B1, B2  # noqa: E402

OUT = Path(__file__).resolve().parent
SEEDS = list(range(60))
GROUPS = {"G": GOOD, "B1": B1, "B2": B2}
label = {s: ("G" if s in GOOD else "B1" if s in B1 else "B2" if s in B2 else "other")
         for s in SEEDS}


def distogram(model, pair_mask, seed, loop):
    z = torch.from_numpy(np.load(
        REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
    ).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)
    del z
    return P


def main():
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]

    # --- pass 1: committed (loop-16) centroids per group (diagnostic labels) ---
    print("computing loop-16 centroids...", flush=True)
    cents = {}
    for g, seeds in GROUPS.items():
        acc = None
        for s in seeds:
            P = distogram(model, pair_mask, s, N_LOOPS)
            acc = P.clone() if acc is None else acc + P
        P = acc / len(seeds)
        cents[g] = (P / P.sum(-1, keepdim=True)).detach()
    print(f"centroids: {list(cents)}", flush=True)

    # --- pass 2: every seed, every loop -> W1 to each centroid ---
    rows = []
    for loop in range(N_LOOPS + 1):
        t0 = time.time()
        for s in SEEDS:
            P = distogram(model, pair_mask, s, loop)
            rows.append(dict(seed=s, label=label[s], loop=loop,
                             d_B1=round(float(w1(P, cents["B1"])), 5),
                             d_G=round(float(w1(P, cents["G"])), 5),
                             d_B2=round(float(w1(P, cents["B2"])), 5)))
        print(f"loop {loop:2d} done ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()

    with open(OUT / "trajectory_depth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- summarize the commit-then-correct shape for G seeds ---
    def series(seed, key):
        return [r[key] for r in rows if r["seed"] == seed]
    loops = list(range(N_LOOPS + 1))
    print("\nG-seed distance-to-B1-centroid (does it dip then rise?):", flush=True)
    for s in GOOD:
        d = series(s, "d_B1")
        lo = int(np.argmin(d))
        print(f"  seed {s:2d}: dB1 L0={d[0]:.3f}  min={d[lo]:.3f}@L{lo}  L16={d[-1]:.3f}  "
              f"{'DIP-THEN-RISE' if (lo not in (0, 16) and d[lo] < d[0] and d[-1] > d[lo]) else 'no clear dip'}",
              flush=True)

    def group_mean(g, key):
        return [np.mean([r[key] for r in rows if r["loop"] == lp and r["label"] == g])
                for lp in loops]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"B1": "tab:blue", "G": "tab:green", "B2": "tab:orange"}
    for g in ("B1", "G", "B2"):
        ax[0].plot(loops, group_mean(g, "d_B1"), "-o", color=colors[g], label=g)
        ax[1].plot(loops, group_mean(g, "d_G"), "-o", color=colors[g], label=g)
    ax[0].set_title("mean W1 to B1 centroid (depth into B1)")
    ax[0].set_xlabel("loop"); ax[0].set_ylabel("W1 to B1"); ax[0].legend()
    ax[1].set_title("mean W1 to G centroid")
    ax[1].set_xlabel("loop"); ax[1].set_ylabel("W1 to G"); ax[1].legend()
    for s in GOOD:
        ax[2].plot(loops, series(s, "d_B1"), "-", label=f"s{s} ->B1")
        ax[2].plot(loops, series(s, "d_G"), "--", color=ax[2].lines[-1].get_color())
    ax[2].set_title("individual G seeds: ->B1 (solid), ->G (dashed)")
    ax[2].set_xlabel("loop"); ax[2].set_ylabel("W1"); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "trajectory_depth.png", dpi=130)
    print("\nsaved", OUT / "trajectory_depth.csv", "and .png", flush=True)


if __name__ == "__main__":
    main()
