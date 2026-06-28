"""Self-calibrating timing signal test: is "commitment" observable as per-particle
VELOCITY (W1 between consecutive loops) collapsing while NEAR the dense default
basin? Tests whether B1 seeds' velocity collapses early (settle into the core)
while G seeds stay plastic / SPIKE at the snap (loops ~4-6) — a label-free,
loop-number-free timing cue.

Per-loop distograms from saved pairformer_z. Labels DIAGNOSTIC only (to read the
signal); the cue itself (velocity + distance-to-dense-centroid) uses no labels.

Saves CSV + PNG only. Run: CUDA_VISIBLE_DEVICES=N ESM_PYTHON ... velocity_probe.py
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

    # dense-default centroid (B1, committed L16) — identified by density in the real
    # algorithm; here we use the B1 label as a diagnostic stand-in.
    print("computing B1 (dense default) centroid...", flush=True)
    acc = None
    for s in B1:
        P = distogram(model, pair_mask, s, N_LOOPS)
        acc = P.clone() if acc is None else acc + P
    P = acc / len(B1)
    dense = (P / P.sum(-1, keepdim=True)).detach()

    rows = []
    for s in SEEDS:
        t0 = time.time()
        prev = None
        for loop in range(N_LOOPS + 1):
            P = distogram(model, pair_mask, s, loop)
            vel = float(w1(P, prev)) if prev is not None else float("nan")
            rows.append(dict(seed=s, label=label[s], loop=loop,
                             velocity=round(vel, 5) if prev is not None else "",
                             d_dense=round(float(w1(P, dense)), 5)))
            prev = P
        if s % 10 == 0:
            print(f"seed {s:2d} done ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()

    with open(OUT / "velocity_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    loops = list(range(N_LOOPS + 1))

    def gmean(grp, key, lp):
        vals = [r[key] for r in rows if int(r["loop"]) == lp and r["label"] == grp and r[key] != ""]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nloop   vel:B1   vel:G    vel:B2  |  d_dense:B1  d_dense:G", flush=True)
    for lp in loops:
        print(f"{lp:3d}  {gmean('B1','velocity',lp):7.4f} {gmean('G','velocity',lp):7.4f} "
              f"{gmean('B2','velocity',lp):7.4f}  | {gmean('B1','d_dense',lp):9.3f} "
              f"{gmean('G','d_dense',lp):9.3f}", flush=True)

    # peak-velocity loop per G seed (the snap)
    print("\nG-seed peak-velocity loop (the snap):", flush=True)
    for s in GOOD:
        v = [(int(r["loop"]), r["velocity"]) for r in rows if r["seed"] == s and r["velocity"] != ""]
        pl, pv = max(v, key=lambda t: t[1])
        print(f"  seed {s:2d}: peak vel {pv:.4f} @ loop {pl}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"B1": "tab:blue", "G": "tab:green", "B2": "tab:orange"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for g in ("B1", "G", "B2"):
        ax[0].plot(loops, [gmean(g, "velocity", lp) for lp in loops], "-o", color=colors[g], label=g)
    ax[0].set_title("mean velocity (W1 loop i vs i-1)")
    ax[0].set_xlabel("loop"); ax[0].set_ylabel("velocity"); ax[0].legend()
    # velocity vs distance-to-dense: commitment = low vel + low d_dense (bottom-left)
    for g in ("B1", "G", "B2"):
        xs = [gmean(g, "d_dense", lp) for lp in loops]
        ys = [gmean(g, "velocity", lp) for lp in loops]
        ax[1].plot(xs, ys, "-o", color=colors[g], label=g, ms=3)
    ax[1].set_title("velocity vs distance-to-dense (group means, loops 1..16)")
    ax[1].set_xlabel("distance to dense default"); ax[1].set_ylabel("velocity"); ax[1].legend()
    for s in GOOD:
        v = [r["velocity"] if r["velocity"] != "" else np.nan for r in rows if r["seed"] == s]
        ax[2].plot(loops, v, "-", label=f"s{s}")
    ax[2].set_title("individual G-seed velocity (the snap spike)")
    ax[2].set_xlabel("loop"); ax[2].set_ylabel("velocity"); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "velocity_probe.png", dpi=130)
    print("\nsaved", OUT / "velocity_probe.csv", "and .png", flush=True)


if __name__ == "__main__":
    main()
