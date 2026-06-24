"""PCA of ESMC representations across the 60 7txw seeds (first look).

Each seed's saved ESMC hidden states are [1, 609, 81, 2560] (residues x layers x dim).
We collapse each seed to one vector (final layer, mean-pooled over residues), PCA across
all 60 seeds, and plot PC1-PC2 colored by DockQ outcome to see whether the 4 good seeds
(10, 14, 33, 39) separate from the rest.

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
LAYER = 80  # final ESMC layer (0=embedding .. 80=last); one line to change

GOOD_T, MARG_T = 0.5, 0.23  # A|C per-seed max DockQ thresholds


def seed_dockq_ac():
    """Per-seed max DockQ on the A|C interface."""
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def load_pooled(seed):
    """Final-layer ESMC rep, mean-pooled over residues -> (2560,) float32."""
    arr = np.load(REPS_DIR / f"seed_{seed}" / "reps" / "esmc_hidden_states.npy",
                  mmap_mode="r")
    return np.asarray(arr[0, :, LAYER, :], dtype=np.float32).mean(axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()

    X = np.stack([load_pooled(s) for s in range(N_SEEDS)])  # [60, 2560]
    print(f"X {X.shape}; seeds loaded {N_SEEDS}")
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    marg = sorted(s for s in range(N_SEEDS) if MARG_T <= dvals[s] < GOOD_T)
    print(f"good (>= {GOOD_T}): {good}")
    print(f"marginal [{MARG_T}, {GOOD_T}): {marg}")
    assert good == [10, 14, 33, 39], f"unexpected good set: {good}"

    # PCA via SVD on centered features
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S            # [60, k] PC scores
    evr = (S ** 2) / (S ** 2).sum()
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
    ax.set_title("ESMC reps (final layer, mean-pooled) — by DockQ class")
    ax.legend(frameon=False)

    ax = axes[1]
    sc = ax.scatter(pc1, pc2, s=55, c=dvals, cmap="viridis",
                    edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="A|C DockQ (per-seed max)")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("same PCA — colored by DockQ")

    fig.suptitle("7txw — PCA of ESMC representations across 60 seeds", y=1.02)
    fig.tight_layout()
    out = OUT / "pca_esmc_reps_7txw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
