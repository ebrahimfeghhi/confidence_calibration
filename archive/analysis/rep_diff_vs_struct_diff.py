"""Does the loop16 pairformer-z difference between orientation basins live ONLY in the
residue-pairs whose PREDICTED geometry actually differs?

7txw is an antigen-orientation problem: the antibody (A,B) scaffold folds the same in
every seed, only the antigen (C) docks in different poses. For a rigid antigen
reorientation the *structure* changes only in the antibody<->antigen cross distances;
antigen-internal (C×C) and antibody-internal distances are conserved. The question is
whether the *representation* difference is confined to that same cross region, or whether
it leaks into pairs whose geometry didn't change (notably antigen-internal C×C, which the
earlier block-PCA showed still separates good from bad).

Basins (from loop16 pairformer-z, all 60 seeds; marginal dropped):
  G        = good seeds (A|C DockQ >= 0.5)                    -> correct orientation
  B1, B2   = two sub-clusters of the bad cloud (k-means k=2 on bad PC1-PC2 scores)
             == the lower / upper blue clouds in pca_pairformer_loops_grid.png

Per basin-pair contrast (G-B1, G-B2, B1-B2), over residue pairs (i,j):
  struct_diff[i,j] = | mean_g d_CA(i,j) - mean_h d_CA(i,j) |     (pose-invariant; no align)
  rep_diff_z[i,j]  = || (mean_g z - mean_h z)(i,j,:) / sd ||_2   (256 ch, per-feature z-scored)
  rep_diff_raw     = same without the sd standardization

Deliverables:
  - block-mean table of struct_diff vs rep_diff (the C×C "leakage" is the headline number)
  - antigen-internal conservation check (is C×C structure really conserved?)
  - figs/rep_vs_struct_maps.png      side-by-side heatmaps per contrast
  - figs/rep_vs_struct_scatter.png   rep_diff vs struct_diff (hexbin + per-block centroids)

Reps: single loop16 stack, 60 x [609,609,256] f32 ~= 23 GB held once (fine on this box).
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
DIVIDERS = (219, 438)
BLOCKS = [("AxA", "A", "A"), ("AxB", "A", "B"), ("AxC", "A", "C"),
          ("BxB", "B", "B"), ("BxC", "B", "C"), ("CxC", "C", "C")]
# structural classification of each block: cross = geometry changes under reorientation
CROSS = {"AxC", "BxC"}            # antibody <-> antigen  (distance changes)
AG_INT = {"CxC"}                 # antigen-internal      (distance conserved)


def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def parse_ca(path):
    """CA xyz in file (token) order -> ([N,3], chain_per_index list)."""
    cols, idx, inl = {}, 0, False
    xyz, chains = [], []
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = idx; idx += 1; inl = True; continue
        if inl and s.startswith(("ATOM", "HETATM")):
            p = s.split()
            if p[cols["label_atom_id"]] != "CA":
                continue
            chains.append(p[cols["label_asym_id"]])
            xyz.append((float(p[cols["Cartn_x"]]), float(p[cols["Cartn_y"]]),
                        float(p[cols["Cartn_z"]])))
        elif inl and s.startswith("#"):
            inl = False
    return np.asarray(xyz, np.float32), chains


def dist_matrix(coords):
    g = coords @ coords.T
    d2 = np.diag(g)[:, None] + np.diag(g)[None, :] - 2 * g
    return np.sqrt(np.clip(d2, 0, None)).astype(np.float32)


def kmeans2(X, n_restarts=20, n_iter=100, seed=0):
    """Minimal k=2 k-means; returns labels minimizing inertia over restarts."""
    rng = np.random.default_rng(seed)
    best_lab, best_in = None, np.inf
    for _ in range(n_restarts):
        c = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(n_iter):
            lab = (((X[:, None] - c[None]) ** 2).sum(2)).argmin(1)
            newc = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k]
                             for k in (0, 1)])
            if np.allclose(newc, c):
                c = newc; break
            c = newc
        inertia = sum(((X[lab == k] - c[k]) ** 2).sum() for k in (0, 1))
        if inertia < best_in:
            best_in, best_lab = inertia, lab
    return best_lab


def block_mean(mat, r, c):
    return float(mat[SPANS[r], SPANS[c]].mean())


def spearman(a, b, n=50000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(a), min(n, len(a)), replace=False)
    ra = np.argsort(np.argsort(a[i])).astype(np.float64)
    rb = np.argsort(np.argsort(b[i])).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    marg = [s for s in range(N_SEEDS) if MARG_T <= dvals[s] < GOOD_T]
    bad = [s for s in range(N_SEEDS) if dvals[s] < MARG_T]
    assert good == [10, 14, 33, 39], good

    # ---- load loop16 reps once: X [60, P] f32 (centered later) ----
    P = 609 * 609 * N_CH
    print(f"loading {N_SEEDS} loop{LOOP} reps -> X [{N_SEEDS},{P:,}] (~{N_SEEDS*P*4/1e9:.0f} GB)")
    X = np.empty((N_SEEDS, P), np.float32)
    for s in range(N_SEEDS):
        z = np.load(REPS_DIR / f"seed_{s}" / "reps" / f"pairformer_z_loop{LOOP:02d}.npy",
                    mmap_mode="r")
        X[s] = np.asarray(z[0], np.float32).reshape(-1)
        del z
        if s % 15 == 0:
            print(f"  loaded seed {s}")
    mu = X.mean(0); X -= mu                      # center in place (for PCA + basin diffs)
    sd = X.std(0); sd_eps = sd + 1e-6            # per-feature std (z-scoring of rep diff)

    # ---- PCA via 60x60 Gram, get PC scores ----
    G = X @ X.T
    eval_, evec = np.linalg.eigh(G)
    eval_, evec = np.clip(eval_[::-1], 0, None), evec[:, ::-1]
    scores = evec * np.sqrt(eval_)               # [60, 60]

    # ---- split bad cloud into B1/B2 by k-means on its PC1-PC2 ----
    bad = np.array(bad)
    lab = kmeans2(scores[bad, :2])
    B1 = sorted(bad[lab == 0]); B2 = sorted(bad[lab == 1])
    # order B1 = lower PC2 (the "lower blue"), B2 = upper, for naming stability
    if scores[B1, 1].mean() > scores[B2, 1].mean():
        B1, B2 = B2, B1
    basins = {"G": good, "B1": B1, "B2": B2}
    print("\nbasins (loop16 pairformer-z):")
    for name, mem in basins.items():
        print(f"  {name:3s} n={len(mem):2d}  seeds={mem}")
    named_bad = {0, 12, 32, 42}
    print(f"  user-named bad {sorted(named_bad)} -> "
          f"B1:{sorted(named_bad & set(B1))}  B2:{sorted(named_bad & set(B2))}")

    # ---- per-basin mean rep (from centered X) and mean distance matrix ----
    mean_rep = {n: X[mem].mean(0) for n, mem in basins.items()}   # [P] each
    Dmats = {}
    for s in range(N_SEEDS):
        coords, chains = parse_ca(REPS_DIR / f"seed_{s}" / "predictions" / "7txw_sample_0.cif")
        if s == 0:
            assert chains[:1] == ["A"] and len(coords) == 609, (len(coords), chains[0])
        Dmats[s] = dist_matrix(coords)
    mean_dist = {n: np.mean([Dmats[s] for s in mem], 0) for n, mem in basins.items()}

    # antigen-internal conservation: spread of C×C distances across all seeds
    cc = SPANS["C"]
    cc_std = np.std([Dmats[s][cc, cc] for s in range(N_SEEDS)], 0).mean()
    print(f"\nantigen-internal (C×C) CA-dist std across all 60 seeds: {cc_std:.2f} A "
          f"(small => antigen fold conserved; C×C is a clean structure-same control)")

    sd_grid = sd_eps.reshape(609, 609, N_CH)

    def repz_map(idx_g, idx_h):
        diff = (X[idx_g].mean(0) - X[idx_h].mean(0)).reshape(609, 609, N_CH)
        return np.linalg.norm(diff / sd_grid, axis=2)

    contrasts = [("G", "B1"), ("G", "B2"), ("B1", "B2")]
    struct, repz, repraw = {}, {}, {}
    for g, h in contrasts:
        struct[(g, h)] = np.abs(mean_dist[g] - mean_dist[h])                    # [609,609]
        diff = (mean_rep[g] - mean_rep[h]).reshape(609, 609, N_CH)
        repraw[(g, h)] = np.linalg.norm(diff, axis=2)
        repz[(g, h)] = np.linalg.norm(diff / sd_grid, axis=2)

    # ---- permutation null: how much block rep_diff_z is just sampling noise? ----
    # For each contrast, randomly re-split the pooled seeds into same-size groups and
    # recompute block-mean rep_diff_z. Observed >> null => the basin difference in that
    # block is real (not an artifact of averaging few/noisy seeds).
    N_PERM = 100
    rng = np.random.default_rng(0)
    null = {}                          # (g,h) -> {block: (mean, sd)}
    for g, h in contrasts:
        pool = np.array(list(basins[g]) + list(basins[h]))
        ng = len(basins[g])
        acc = {nm: [] for nm, _, _ in BLOCKS}
        for _ in range(N_PERM):
            perm = rng.permutation(pool)
            m = repz_map(perm[:ng], perm[ng:])
            for nm, r, c in BLOCKS:
                acc[nm].append(block_mean(m, r, c))
        null[(g, h)] = {nm: (float(np.mean(v)), float(np.std(v))) for nm, v in acc.items()}

    # ---- block-mean table ----
    print("\n" + "=" * 78)
    print("block-mean STRUCT_diff (A) and REP_diff_z (sd units), per contrast")
    print("  cross = AxC,BxC (geometry changes)   ag-int = CxC (geometry conserved)")
    print("=" * 78)
    for g, h in contrasts:
        print(f"\n{g} vs {h}:   (rep_z null = same-size random re-split; z = (obs-null)/null_sd)")
        print(f"  {'block':5s} {'class':7s} {'struct_dA':>10s} {'rep_z':>8s} "
              f"{'null':>7s} {'z':>6s}")
        sd_struct, sd_rep = struct[(g, h)], repz[(g, h)]
        for nm, r, c in BLOCKS:
            cls = "cross" if nm in CROSS else ("ag-int" if nm in AG_INT else "ab-int")
            obs = block_mean(sd_rep, r, c)
            nmu, nsd = null[(g, h)][nm]
            z = (obs - nmu) / nsd if nsd > 0 else float("nan")
            print(f"  {nm:5s} {cls:7s} {block_mean(sd_struct, r, c):10.2f} "
                  f"{obs:8.2f} {nmu:7.2f} {z:6.1f}")
        # headline leakage ratios
        cross_struct = np.mean([block_mean(sd_struct, *b[1:]) for b in BLOCKS if b[0] in CROSS])
        cc_struct = block_mean(sd_struct, "C", "C")
        cross_rep = np.mean([block_mean(sd_rep, *b[1:]) for b in BLOCKS if b[0] in CROSS])
        cc_rep = block_mean(sd_rep, "C", "C")
        print(f"  -> struct CxC/cross = {cc_struct/cross_struct:.3f}   "
              f"rep_z CxC/cross = {cc_rep/cross_rep:.3f}   "
              f"(rep >> struct ratio => rep leaks into geometry-conserved CxC)")
        print(f"  -> Spearman(rep_z, struct) over all pairs = "
              f"{spearman(sd_rep.ravel(), sd_struct.ravel()):.3f}")

    _figures(contrasts, struct, repz)
    _null_figure(contrasts, repz, null)
    print("\ndone.")


def _null_figure(contrasts, repz, null):
    fig, axes = plt.subplots(1, len(contrasts), figsize=(5.5 * len(contrasts), 4.5),
                             sharey=True)
    names = [nm for nm, _, _ in BLOCKS]
    for ax, (g, h) in zip(np.atleast_1d(axes), contrasts):
        obs = [block_mean(repz[(g, h)], r, c) for _, r, c in BLOCKS]
        nmu = [null[(g, h)][nm][0] for nm in names]
        nsd = [null[(g, h)][nm][1] for nm in names]
        cols = ["#d62728" if nm in CROSS else ("#1f77b4" if nm in AG_INT else "0.6")
                for nm in names]
        x = np.arange(len(names))
        ax.bar(x, obs, color=cols, label="observed")
        ax.errorbar(x, nmu, yerr=nsd, fmt="_k", ms=18, lw=1.5, capsize=4,
                    label="null (random re-split)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45)
        ax.set_title(f"{g} vs {h}")
    np.atleast_1d(axes)[0].set_ylabel("block-mean rep_diff_z (sd units)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.suptitle("rep difference vs sampling-noise null. Bars above the null floor = real "
                 "basin difference; ag-int CxC & ab-int above null = rep leaks beyond geometry.",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "rep_vs_struct_null.png", dpi=130, bbox_inches="tight")
    print("wrote", OUT / "rep_vs_struct_null.png")


def _figures(contrasts, struct, repz):
    # ---- heatmap maps ----
    fig, axes = plt.subplots(len(contrasts), 2, figsize=(11, 5 * len(contrasts)))
    for row, (g, h) in enumerate(contrasts):
        for col, (mat, title) in enumerate(
                ((struct[(g, h)], f"{g}-{h}  struct_diff |Δd_CA| (Å)"),
                 (repz[(g, h)], f"{g}-{h}  rep_diff_z (sd units)"))):
            ax = axes[row, col]
            vmax = np.percentile(mat, 99)
            im = ax.imshow(mat, cmap="magma", vmax=vmax, origin="upper", aspect="equal")
            for d in DIVIDERS:
                ax.axhline(d - 0.5, color="cyan", lw=0.5); ax.axvline(d - 0.5, color="cyan", lw=0.5)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([109, 328, 523]); ax.set_xticklabels(["A", "B", "C"])
            ax.set_yticks([109, 328, 523]); ax.set_yticklabels(["A", "B", "C"])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("7txw loop16 — structural vs representational difference between "
                 "antigen-orientation basins", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "rep_vs_struct_maps.png", dpi=130, bbox_inches="tight")
    print("\nwrote", OUT / "rep_vs_struct_maps.png")

    # ---- scatter: rep_diff_z vs struct_diff, per-block centroids overlaid ----
    fig, axes = plt.subplots(1, len(contrasts), figsize=(6 * len(contrasts), 5.5))
    for ax, (g, h) in zip(np.atleast_1d(axes), contrasts):
        smat, rmat = struct[(g, h)], repz[(g, h)]
        ax.hexbin(smat.ravel(), rmat.ravel(), gridsize=60, bins="log", cmap="Greys", mincnt=1)
        for nm, r, c in BLOCKS:
            cls = "cross" if nm in CROSS else ("ag-int" if nm in AG_INT else "ab-int")
            col = {"cross": "#d62728", "ag-int": "#1f77b4", "ab-int": "0.5"}[cls]
            ax.scatter(block_mean(smat, r, c), block_mean(rmat, r, c), s=140,
                       c=col, edgecolors="k", zorder=5)
            ax.annotate(nm, (block_mean(smat, r, c), block_mean(rmat, r, c)),
                        fontsize=9, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("struct_diff |Δd_CA| (Å)")
        ax.set_ylabel("rep_diff_z (sd units)")
        ax.set_title(f"{g} vs {h}   ρ={spearman(rmat.ravel(), smat.ravel()):.2f}")
    handles = [plt.Line2D([], [], marker="o", ls="", mec="k", color=c, label=l)
               for l, c in (("cross AxC/BxC", "#d62728"), ("ag-int CxC", "#1f77b4"),
                            ("ab-int", "0.5"))]
    fig.legend(handles=handles, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0.07, 1.0))
    fig.suptitle("rep_diff vs struct_diff per residue-pair (block centroids highlighted). "
                 "ag-int CxC sitting high = rep differs where geometry is conserved.", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "rep_vs_struct_scatter.png", dpi=130, bbox_inches="tight")
    print("wrote", OUT / "rep_vs_struct_scatter.png")


if __name__ == "__main__":
    main()
