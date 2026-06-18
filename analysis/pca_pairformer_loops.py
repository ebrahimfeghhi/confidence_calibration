"""Trajectory of the good-seed cluster across the 17 pairformer-z recycling loops.

For each loop 00..16, load all 60 seeds' full pairformer_z_loop{ii} [1,609,609,256],
flatten, PCA via the 60x60 Gram matrix, and measure how separated/compact the 4 good
seeds {10,14,33,39} are. Silhouette is a ratio of distances within a loop, so it is
comparable across loops even though the rep magnitude grows with recycling.

Outputs:
  analysis/figs/pca_pairformer_loops_separation.png  (silhouette + var vs loop)
  analysis/figs/pca_pairformer_loops_grid.png        (PC1-PC2 scatter per loop)

numpy + matplotlib only.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPS_DIR = Path("/home/ebrahim/esm/foldbench_predictions/7txw_reps/7txw")
CSV = Path("/home/ebrahim/tts_abag/results/csv_files/dockq_ost_7txw_reps_all_samples.csv")
OUT = Path("/home/ebrahim/tts_abag/analysis/figs")
N_SEEDS, N_LOOPS = 60, 17
GOOD_T, MARG_T = 0.5, 0.23


def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def silhouette(D, good_mask):
    """Mean silhouette of the good group (good vs the rest) given a distance matrix."""
    other = ~good_mask
    s = []
    for i in np.where(good_mask)[0]:
        same = good_mask.copy(); same[i] = False
        a = D[i, same].mean()                 # cohesion (to other good)
        b = D[i, other].mean()                # separation (to bad)
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    marg = sorted(s for s in range(N_SEEDS) if MARG_T <= dvals[s] < GOOD_T)
    assert good == [10, 14, 33, 39], f"unexpected good set: {good}"
    good_mask = np.zeros(N_SEEDS, bool); good_mask[good] = True

    P = 609 * 609 * 256
    X = np.empty((N_SEEDS, P), dtype=np.float32)   # ~23 GB, reused across loops

    sil = np.zeros(N_LOOPS)
    var2 = np.zeros(N_LOOPS)
    pcs = np.zeros((N_LOOPS, N_SEEDS, 2))
    for L in range(N_LOOPS):
        for s in range(N_SEEDS):
            z = np.load(REPS_DIR / f"seed_{s}" / "reps" / f"pairformer_z_loop{L:02d}.npy",
                        mmap_mode="r")
            X[s] = np.asarray(z[0], dtype=np.float32).reshape(-1)
            del z
        X -= X.mean(0)                              # center in place
        G = X @ X.T                                 # [60,60]
        diag = np.diag(G)
        D = np.sqrt(np.clip(diag[:, None] + diag[None, :] - 2 * G, 0, None))
        eval_, evec = np.linalg.eigh(G)
        eval_, evec = np.clip(eval_[::-1], 0, None), evec[:, ::-1]
        scores = evec * np.sqrt(eval_)
        # sign-fix so the good centroid sits in the (+,+) quadrant (visual continuity)
        for k in (0, 1):
            if scores[good, k].mean() < 0:
                scores[:, k] *= -1
        pcs[L] = scores[:, :2]
        var2[L] = (eval_[:2] / eval_.sum()).sum()
        sil[L] = silhouette(D, good_mask)
        print(f"loop{L:02d}  silhouette(good)={sil[L]:+.3f}  PC1+2 var={var2[L]:.1%}")

    # ---- figure 1: separation vs loop ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(range(N_LOOPS), sil, "-o", color="#d62728", label="silhouette of good seeds")
    ax.set_xlabel("pairformer recycling loop")
    ax.set_ylabel("silhouette (good vs bad)", color="#d62728")
    ax.set_xticks(range(N_LOOPS))
    ax2 = ax.twinx()
    ax2.plot(range(N_LOOPS), var2 * 100, "--s", color="#1f77b4", alpha=0.7,
             label="PC1+PC2 variance")
    ax2.set_ylabel("PC1+PC2 variance (%)", color="#1f77b4")
    ax.set_title("7txw — when does the good-seed cluster form across recycling?")
    fig.tight_layout()
    fig.savefig(OUT / "pca_pairformer_loops_separation.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "pca_pairformer_loops_separation.png")

    # ---- figure 2: scatter grid per loop ----
    cls = np.array(["bad"] * N_SEEDS, dtype=object)
    cls[good] = "good"; cls[marg] = "marginal"
    style = {"bad": ("#7fa8d0", 16, 0.7), "marginal": ("#ff8c00", 45, 0.95),
             "good": ("#d62728", 60, 0.95)}
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    for L in range(N_LOOPS):
        ax = axes.flat[L]
        for c in ("bad", "marginal", "good"):
            m = cls == c
            col, sz, al = style[c]
            ax.scatter(pcs[L, m, 0], pcs[L, m, 1], s=sz, c=col, alpha=al, edgecolors="none")
        ax.set_title(f"loop{L:02d}  sil={sil[L]:+.2f}  var={var2[L]:.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(N_LOOPS, axes.size):
        axes.flat[j].axis("off")
    axes.flat[0].scatter([], [], c="#d62728", label="good");
    axes.flat[0].scatter([], [], c="#ff8c00", label="marginal")
    axes.flat[0].scatter([], [], c="#7fa8d0", label="bad")
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("7txw — PC1-PC2 of full pairformer-z per recycling loop "
                 "(good seeds red; sign-fixed to +,+)", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "pca_pairformer_loops_grid.png", dpi=130, bbox_inches="tight")
    print("wrote", OUT / "pca_pairformer_loops_grid.png")


if __name__ == "__main__":
    main()
