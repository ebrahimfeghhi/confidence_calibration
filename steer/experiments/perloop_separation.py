"""Beam-probe reality check (no trunk run): cluster the 60 saved 7txw seeds by their
cross-block distogram at EVERY recycling loop and measure when the good basin G
separates from the rest. Uses only saved pairformer_z_loop{00..16} -> distogram
(forward_grad.cross_distogram_from_z). Tests the "early uninformative / late snap"
claim and validates the W1 + agglomerative clustering the beam relies on.

Saves CSV + PNG only (disk quota). Run: ESM_PYTHON steer/core... see launch cmd.
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import load, cross_distogram_from_z, AB, C, GOOD, N_LOOPS, REPS  # noqa: E402
from basins import B1, B2  # noqa: E402

OUT = Path(__file__).resolve().parent
SEEDS = list(range(60))
NPAIRS = (AB.stop - AB.start) * (C.stop - C.start)   # 438 * 171

label = {}
for s in SEEDS:
    label[s] = "G" if s in GOOD else "B1" if s in B1 else "B2" if s in B2 else "other"
labels = np.array([label[s] for s in SEEDS])
print("label counts:", {k: int((labels == k).sum()) for k in ("G", "B1", "B2", "other")},
      flush=True)


def distogram_cdf(model, pair_mask, seed, loop):
    z = torch.from_numpy(np.load(
        REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
    ).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)        # [NPAIRS, nbins], softmaxed
    cdf = P.cumsum(-1).reshape(-1).half().cpu()                # flattened CDF
    del z, P
    return cdf


def pairwise_w1(V):
    """V [n,D] CDF vectors on GPU -> symmetric W1 matrix (mean L1 over pairs)."""
    n = V.shape[0]
    W = torch.zeros(n, n)
    for i in range(n):
        W[i] = (V - V[i]).abs().sum(1).cpu() / NPAIRS
    return W.numpy()


def main():
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]

    g_mask = labels == "G"
    idx3 = np.where(np.isin(labels, ["G", "B1", "B2"]))[0]      # drop 'other' for 3-way

    rows = []
    for loop in range(N_LOOPS + 1):
        t0 = time.time()
        V = torch.stack([distogram_cdf(model, pair_mask, s, loop) for s in SEEDS]).cuda().float()
        if loop == 0:
            print(f"CDF vec dim D={V.shape[1]} (nbins={V.shape[1] // NPAIRS})", flush=True)
        W = pairwise_w1(V)
        del V
        torch.cuda.empty_cache()

        sil_gr = silhouette_score(W, g_mask.astype(int), metric="precomputed")
        sil_3 = silhouette_score(W[np.ix_(idx3, idx3)], labels[idx3], metric="precomputed")

        gi = np.where(g_mask)[0]
        ri = np.where(~g_mask)[0]
        w_gg = W[np.ix_(gi, gi)][np.triu_indices(len(gi), 1)].mean()
        w_grest = W[np.ix_(gi, ri)].mean()
        b1i = np.where(labels == "B1")[0]
        w_b1b1 = W[np.ix_(b1i, b1i)][np.triu_indices(len(b1i), 1)].mean()

        off = W[np.triu_indices(len(SEEDS), 1)]
        tau = float(np.median(off) * 0.5)                       # relative threshold
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=tau,
                                     metric="precomputed", linkage="average").fit_predict(W)
        nclu = len(set(cl))
        g_clusters = set(cl[g_mask])
        g_one = len(g_clusters) == 1
        purity = float(g_mask[cl == next(iter(g_clusters))].mean()) if g_one else float("nan")

        row = dict(loop=loop, sil_G_vs_rest=round(sil_gr, 4), sil_3way=round(sil_3, 4),
                   w1_GG=round(float(w_gg), 5), w1_G_rest=round(float(w_grest), 5),
                   w1_B1B1=round(float(w_b1b1), 5), gap_ratio=round(float(w_grest / (w_gg + 1e-9)), 3),
                   n_clusters=nclu, tau=round(tau, 5), G_in_one_cluster=g_one,
                   G_cluster_purity=round(purity, 3))
        rows.append(row)
        print(f"loop {loop:2d}  silGR={sil_gr:+.3f} sil3={sil_3:+.3f} "
              f"w1GG={w_gg:.4f} w1G->rest={w_grest:.4f} gap={row['gap_ratio']:.2f} "
              f"nclu={nclu} Gtogether={g_one} purity={purity:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)

    with open(OUT / "perloop_separation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lp = [r["loop"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(lp, [r["sil_G_vs_rest"] for r in rows], "-o", label="G vs rest")
    ax[0].plot(lp, [r["sil_3way"] for r in rows], "-s", label="3-way G/B1/B2")
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_xlabel("recycling loop"); ax[0].set_ylabel("silhouette (precomputed W1)")
    ax[0].set_title("When does G separate?"); ax[0].legend()
    ax[1].plot(lp, [r["w1_GG"] for r in rows], "-o", label="within G")
    ax[1].plot(lp, [r["w1_G_rest"] for r in rows], "-o", label="G -> rest")
    ax[1].plot(lp, [r["w1_B1B1"] for r in rows], "-o", label="within B1")
    ax[1].set_xlabel("recycling loop"); ax[1].set_ylabel("mean cross-block W1")
    ax[1].set_title("Distogram distances by loop"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "perloop_separation.png", dpi=130)
    print("saved", OUT / "perloop_separation.csv", "and .png", flush=True)


if __name__ == "__main__":
    main()
