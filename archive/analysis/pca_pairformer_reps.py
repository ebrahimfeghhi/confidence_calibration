"""PCA of the FULL final-loop pairformer-z across the 60 7txw seeds.

Each seed's final trunk rep is pairformer_z_loop16.npy [1, 609, 609, 256] (~190 MB f16).
We flatten the whole thing per seed (no interface slicing), stack all 60 -> [60, ~95M]
(~23 GB f32, fits easily in RAM), and PCA via the 60x60 Gram matrix. Plot PC1-PC2 colored
by DockQ outcome to see whether the 4 good seeds (10, 14, 33, 39) separate.

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
N_SEEDS = 60
LOOP = 16  # final trunk iteration (feeds diffusion)
GOOD_T, MARG_T = 0.5, 0.23


def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()

    P = 609 * 609 * 256
    X = np.empty((N_SEEDS, P), dtype=np.float32)  # ~23 GB
    for s in range(N_SEEDS):
        z = np.load(REPS_DIR / f"seed_{s}" / "reps" / f"pairformer_z_loop{LOOP:02d}.npy",
                    mmap_mode="r")
        X[s] = np.asarray(z[0], dtype=np.float32).reshape(-1)
        del z
    print(f"X {X.shape} ({X.nbytes/1e9:.1f} GB)")

    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    marg = sorted(s for s in range(N_SEEDS) if MARG_T <= dvals[s] < GOOD_T)
    print(f"good: {good}  marginal: {marg}")
    assert good == [10, 14, 33, 39], f"unexpected good set: {good}"

    # PCA via the 60x60 Gram matrix of centered data (avoids SVD on the huge axis)
    X -= X.mean(0)
    G = X @ X.T                                   # [60, 60]
    eval_, evec = np.linalg.eigh(G)               # ascending
    eval_, evec = eval_[::-1], evec[:, ::-1]
    eval_ = np.clip(eval_, 0, None)
    scores = evec * np.sqrt(eval_)                # [60, k] PC scores
    evr = eval_ / eval_.sum()
    pc1, pc2 = scores[:, 0], scores[:, 1]
    print(f"explained variance PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}")

    cls = np.array(["bad"] * N_SEEDS, dtype=object)
    cls[good] = "good"
    cls[marg] = "marginal"
    style = {"bad": ("#7fa8d0", 28, 0.7), "marginal": ("#ff8c00", 70, 0.95),
             "good": ("#d62728", 90, 0.95)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for c in ("bad", "marginal", "good"):
        m = cls == c
        col, sz, al = style[c]
        ax.scatter(pc1[m], pc2[m], s=sz, c=col, alpha=al, edgecolors="none", label=c)
    for s in good + marg:
        ax.annotate(str(s), (pc1[s], pc2[s]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title(f"pairformer-z loop{LOOP} (full, flattened) — by DockQ class")
    ax.legend(frameon=False)

    ax = axes[1]
    sc = ax.scatter(pc1, pc2, s=55, c=dvals, cmap="viridis",
                    edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="A|C DockQ (per-seed max)")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("same PCA — colored by DockQ")

    fig.suptitle("7txw — PCA of full final-loop pairformer-z across 60 seeds", y=1.02)
    fig.tight_layout()
    out = OUT / "pca_pairformer_reps_7txw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
