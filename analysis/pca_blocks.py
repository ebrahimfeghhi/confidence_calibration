"""Localization test: does the good-seed separation live specifically in the A x C
(antibody-antigen) block of the pairformer-z rep, or is it global?

Unsupervised — restrict PCA to one chain-block at a time, then measure the good-seed
silhouette. Compare A x C against control blocks A x A, A x B, B x B, B x C, C x C.
Final loop (loop16) only, for speed.

  A x C wins / highest silhouette -> separation is in the antibody-antigen cross-terms.
  all blocks tie                  -> it's a global rep difference, A x C not special.

Token spans (from predicted cif CA order):  A: 0-218, B: 219-437, C: 438-608.

Output: analysis/figs/pca_blocks_7txw.png
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
N_SEEDS, N_CH, LOOP = 60, 256, 16
GOOD_T, MARG_T = 0.5, 0.23
SPANS = {"A": slice(0, 219), "B": slice(219, 438), "C": slice(438, 609)}
BLOCKS = [("AxC", "A", "C"), ("AxA", "A", "A"), ("AxB", "A", "B"),
          ("BxB", "B", "B"), ("BxC", "B", "C"), ("CxC", "C", "C")]


def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def silhouette(D, good_mask):
    other = ~good_mask
    s = []
    for i in np.where(good_mask)[0]:
        same = good_mask.copy(); same[i] = False
        a = D[i, same].mean()
        b = D[i, other].mean()
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def pca_block(X):
    """Gram-matrix PCA of centered [60, feat]. Returns (D, scores2, var2)."""
    X = X - X.mean(0)
    G = X @ X.T
    diag = np.diag(G)
    D = np.sqrt(np.clip(diag[:, None] + diag[None, :] - 2 * G, 0, None))
    eval_, evec = np.linalg.eigh(G)
    eval_, evec = np.clip(eval_[::-1], 0, None), evec[:, ::-1]
    scores = evec * np.sqrt(eval_)
    var2 = (eval_[:2] / eval_.sum()).sum()
    return D, scores[:, :2], var2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    marg = sorted(s for s in range(N_SEEDS) if MARG_T <= dvals[s] < GOOD_T)
    assert good == [10, 14, 33, 39], good
    good_mask = np.zeros(N_SEEDS, bool); good_mask[good] = True

    # one pass over seeds: slice every block from each seed's full rep
    mats = {name: np.empty((N_SEEDS, (SPANS[r].stop - SPANS[r].start)
                            * (SPANS[c].stop - SPANS[c].start) * N_CH), dtype=np.float32)
            for name, r, c in BLOCKS}
    for s in range(N_SEEDS):
        z = np.load(REPS_DIR / f"seed_{s}" / "reps" / f"pairformer_z_loop{LOOP:02d}.npy",
                    mmap_mode="r")
        zf = np.asarray(z[0], dtype=np.float32)          # [609,609,256]
        for name, r, c in BLOCKS:
            mats[name][s] = zf[SPANS[r], SPANS[c], :].reshape(-1)
        del z, zf
        if s % 15 == 0:
            print(f"  loaded seed {s}")

    res = {}
    for name, r, c in BLOCKS:
        D, sc, v2 = pca_block(mats[name])
        # sign-fix to good centroid in (+,+)
        for k in (0, 1):
            if sc[good, k].mean() < 0:
                sc[:, k] *= -1
        sil = silhouette(D, good_mask)
        res[name] = (sil, sc, v2)
        print(f"{name}: silhouette(good)={sil:+.3f}  PC1+2 var={v2:.1%}  "
              f"({mats[name].shape[1]:,} feat)")

    order = sorted(res, key=lambda n: -res[n][0])
    print("\nblocks ranked by good-seed silhouette:")
    for n in order:
        print(f"  {n}: {res[n][0]:+.3f}")

    # ---------- figure ----------
    cls = np.array(["bad"] * N_SEEDS, dtype=object)
    cls[good] = "good"; cls[marg] = "marginal"
    style = {"bad": ("#7fa8d0", 22, 0.7), "marginal": ("#ff8c00", 55, 0.95),
             "good": ("#d62728", 75, 0.95)}

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4)

    # bar chart of silhouettes
    axb = fig.add_subplot(gs[:, 0])
    names = order
    vals = [res[n][0] for n in names]
    cols = ["#d62728" if n == "AxC" else "#7fa8d0" for n in names]
    axb.barh(range(len(names)), vals, color=cols)
    axb.set_yticks(range(len(names))); axb.set_yticklabels(names)
    axb.invert_yaxis()
    axb.axvline(0, color="0.6", lw=0.8)
    axb.set_xlabel("good-seed silhouette")
    axb.set_title("separation by chain block\n(loop16, A x C highlighted)")

    # scatter per block
    cells = [gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 1], gs[1, 2], gs[1, 3]]
    for cell, name in zip(cells, ["AxC", "AxA", "AxB", "BxB", "BxC", "CxC"]):
        ax = fig.add_subplot(cell)
        sil, sc, v2 = res[name]
        for cl in ("bad", "marginal", "good"):
            m = cls == cl
            col, sz, al = style[cl]
            ax.scatter(sc[m, 0], sc[m, 1], s=sz, c=col, alpha=al, edgecolors="none")
        ec = "#d62728" if name == "AxC" else "0.6"
        for sp in ax.spines.values():
            sp.set_edgecolor(ec); sp.set_linewidth(2 if name == "AxC" else 1)
        ax.set_title(f"{name}  sil={sil:+.2f}  var={v2:.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=style[c][0], label=c)
                        for c in ("good", "marginal", "bad")],
               frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01))
    fig.suptitle("7txw — is the good-seed separation localized to the A x C "
                 "(antibody-antigen) block? [loop16]", y=1.0, fontsize=14)
    fig.tight_layout()
    out = OUT / "pca_blocks_7txw.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
