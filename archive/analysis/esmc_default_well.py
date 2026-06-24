"""Do the good 7txw seeds' ESMC (pre-trunk LM) reps separate, and do they sit AWAY
from the no-mask/no-dropout default well in a consistent direction?

Hypothesis: the antigen-orientation signal that the pairformer-z trunk shows is already
present, pre-trunk, in the ESMC representation as *displacement from the default basin*.
If good seeds leave the default well (the lm_mask=0/dropout=0 baseline rep) farther and
along a shared direction, that direction is a candidate steering axis for a compute-cheap
intervention (no 2000-seed dump needed).

Data (read-only, already on disk):
  reps      7txw_reps/7txw/seed_{s}/reps/esmc_hidden_states.npy   [1,609,81,2560] f16
            (residues x layers x dim;  60 seeds with deterministic input masks)
  baseline  7txw_reps_baseline/7txw/seed_0/reps/esmc_hidden_states.npy
            (lm_mask_pct=0, lm_dropout=0  ->  the "default well" reference)
  labels    dockq_ost_7txw_reps_all_samples.csv, A|C per-seed max DockQ
            good = {10,14,33,39} (>=0.5), bad = the other 56.

Chains: A=0:219  B=219:438  C(antigen)=438:609.  Epitope window = C48-C50 -> 485:488
(the residues the powered mask analysis flagged: mask_pattern_7txw_per_residue.csv).

Four tests (numpy only; permutation p-values, mirroring pca_blocks_perm.py):
  1. Does it separate?   good-set silhouette (Euclid) vs random-4-subset null,
     swept over layer x pooling (mean-pool vs per-residue).  Beat trunk-z (+0.71).
  2. Where?              same, per region (all / antigen-C / epitope / antibody).
  3. Default-basin geometry (the core idea, best-powered):
       a. distance-from-well ||rep_s - rep_base||: good farther than bad? (rank AUC + perm p)
       b. one good direction?  mean pairwise cosine of good (rep_s-rep_base) vs null.
       c. separability along d=mean(good)-mean(bad): leave-one-out projection AUC vs mask 0.62.
  4. Supervised probe (caveated, noisy at 4 good): numpy L1-logistic (ISTA) CV-AUC vs
     label-shuffle null on mean-pooled features.

Output: printed tables, results/csv_files/esmc_default_well_7txw.csv, figs/esmc_default_well.png.
Gate: green = good seeds separate AND sit away from the well along a consistent direction.

Run:  python analysis/esmc_default_well.py     (tts_abag venv: numpy + matplotlib)
"""
from pathlib import Path
import csv
import sys

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, CSV_DIR  # noqa: E402

REPS = ESM_PREDICTIONS / "7txw_reps" / "7txw"
BASE = ESM_PREDICTIONS / "7txw_reps_baseline" / "7txw" / "seed_0" / "reps" / "esmc_hidden_states.npy"
CSV = CSV_DIR / "dockq_ost_7txw_reps_all_samples.csv"
OUT_FIG = _p / "analysis" / "figs" / "esmc_default_well.png"
OUT_CSV = CSV_DIR / "esmc_default_well_7txw.csv"

N_SEEDS = 60
GOOD_T = 0.5
LAYERS = list(range(0, 81, 5))           # 0,5,...,80  (mid-stack + final)
REGIONS = {                              # name -> (lo, hi) residue span
    "all":      (0, 609),
    "antigenC": (438, 609),
    "epitope":  (485, 488),              # C48-C50, the mask-flagged window
    "antibody": (0, 438),
}
N_PERM = 5000          # silhouette / distance / direction null
N_PERM_PROBE = 200     # supervised-probe label-shuffle null
RNG = np.random.default_rng(0)

# benchmarks to beat (from prior analyses / memory)
TRUNK_SIL = 0.71       # pairformer-z trunk good-set silhouette
MASK_AUC = 0.62        # powered input-mask omnibus LOO/CV AUC


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def load_layer(path, layer):
    """ESMC hidden state at one layer, all residues -> [609, 2560] f32."""
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr[0, :, layer, :], dtype=np.float32)


# --------------------------------------------------------------------------- #
# stats (numpy-only; permutation nulls)
# --------------------------------------------------------------------------- #
def silhouette(D, mask):
    """One-sided silhouette of the masked set against the rest (pca_blocks_perm)."""
    other = ~mask
    s = []
    for i in np.where(mask)[0]:
        same = mask.copy(); same[i] = False
        a = D[i, same].mean(); b = D[i, other].mean()
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def euclid_D(mats):
    X = mats - mats.mean(0)
    G = X @ X.T
    diag = np.diag(G)
    return np.sqrt(np.clip(diag[:, None] + diag[None, :] - 2 * G, 0, None))


def rank_auc(scores, pos_mask):
    """AUC = P(score_pos > score_neg) via mid-ranks (== Mann-Whitney U / n1 n0)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = pos_mask.sum(); n0 = (~pos_mask).sum()
    return float((ranks[pos_mask].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_subsets(n, k, n_perm):
    return [RNG.choice(n, k, replace=False) for _ in range(n_perm)]


# --------------------------------------------------------------------------- #
# test 3 helpers
# --------------------------------------------------------------------------- #
def mean_pairwise_cos(V):
    """Mean pairwise cosine among rows of V."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = Vn @ Vn.T
    iu = np.triu_indices(len(V), 1)
    return float(C[iu].mean())


def loo_projection_auc(Z, good_mask):
    """Leave-one-out projection onto d=mean(good)-mean(bad); AUC over all seeds.
    Z is per-dim standardized features (transductive)."""
    n = len(Z); scores = np.empty(n)
    for i in range(n):
        g = good_mask.copy(); g[i] = False
        b = (~good_mask).copy(); b[i] = False
        d = Z[g].mean(0) - Z[b].mean(0)
        scores[i] = Z[i] @ d
    return rank_auc(scores, good_mask), scores


# --------------------------------------------------------------------------- #
# test 4: numpy L1-logistic (ISTA) + CV-AUC + label-shuffle null
# --------------------------------------------------------------------------- #
def l1_logistic_ista(X, y, lam=0.02, n_iter=400):
    """Class-balanced L1-logistic by proximal gradient. X standardized, y in {0,1}."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    w0 = y.mean()
    sw = np.where(y == 1, 0.5 / max(w0, 1e-6), 0.5 / max(1 - w0, 1e-6))  # balanced
    sw = sw / sw.mean()
    L = (np.linalg.norm(X, 2) ** 2) / n * 0.25 + 1e-6                    # Lipschitz
    step = 1.0 / L
    for _ in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (sw * (p - y))
        gw = X.T @ g / n
        gb = g.mean()
        b -= step * gb
        w -= step * gw
        thr = step * lam
        w = np.sign(w) * np.maximum(np.abs(w) - thr, 0.0)               # soft-threshold
    return w, b


def cv_auc_l1(X, y, folds, lam):
    scores = np.empty(len(y))
    for tr, te in folds:
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        w, b = l1_logistic_ista(Xtr, y[tr], lam=lam)
        scores[te] = Xte @ w + b
    return rank_auc(scores, y.astype(bool))


def make_folds(good_idx, bad_idx, k=4):
    """Stratified k folds (one good seed per fold)."""
    gi = RNG.permutation(good_idx); bi = RNG.permutation(bad_idx)
    g_sp = np.array_split(gi, k); b_sp = np.array_split(bi, k)
    folds = []
    n = len(good_idx) + len(bad_idx)
    allidx = np.arange(n)
    for f in range(k):
        te = np.concatenate([g_sp[f], b_sp[f]])
        tr = np.setdiff1d(allidx, te)
        folds.append((tr, te))
    return folds


# --------------------------------------------------------------------------- #
def main():
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    dq = seed_dockq_ac()
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    assert good == [10, 14, 33, 39], good
    good_mask = np.zeros(N_SEEDS, bool); good_mask[good] = True
    bad_idx = np.where(~good_mask)[0]; good_idx = np.array(good)
    print(f"good {good}  ({good_mask.sum()} good / {(~good_mask).sum()} bad)\n")

    # pre-draw shared null subsets (fair across layers/regions)
    sub4 = perm_subsets(N_SEEDS, 4, N_PERM)
    sub_masks = []
    for idx in sub4:
        m = np.zeros(N_SEEDS, bool); m[idx] = True
        sub_masks.append(m)

    rows = []          # one per (layer, region, pooling) for the silhouette sweep
    geo = []           # one per (layer, region) for the distance/direction/LOO sweep
    probe_cache = {}   # (layer, region) -> mean-pooled X for test 4

    for L in LAYERS:
        X = np.stack([load_layer(REPS / f"seed_{s}" / "reps" / "esmc_hidden_states.npy", L)
                      for s in range(N_SEEDS)])          # [60, 609, 2560]
        xb = load_layer(BASE, L)                          # [609, 2560]

        for rname, (lo, hi) in REGIONS.items():
            block = X[:, lo:hi, :]                        # [60, r, 2560]
            base_block = xb[lo:hi, :]

            # ---- tests 1 & 2: silhouette x pooling ----
            for pool in ("mean", "perres"):
                if pool == "mean":
                    feat = block.mean(1)                  # [60, 2560]
                else:
                    feat = block.reshape(N_SEEDS, -1)     # [60, r*2560]
                D = euclid_D(feat)
                sil = silhouette(D, good_mask)
                null = np.fromiter((silhouette(D, m) for m in sub_masks),
                                   float, N_PERM)
                p = (1 + (null >= sil).sum()) / (1 + N_PERM)
                rows.append(dict(layer=L, region=rname, pool=pool, sil=sil,
                                 null_mean=null.mean(), null_p95=np.percentile(null, 95),
                                 excess=sil - null.mean(), p=p))

            # ---- test 3: default-well geometry (mean-pooled) ----
            v = block.mean(1)                             # [60, 2560]
            vb = base_block.mean(0)                       # [2560]
            diff = v - vb
            dist = np.linalg.norm(diff, axis=1)           # [60]
            dist_auc = rank_auc(dist, good_mask)          # good farther -> >0.5
            dnull = np.fromiter((rank_auc(dist, m) for m in sub_masks), float, N_PERM)
            dist_p = (1 + (dnull >= dist_auc).sum()) / (1 + N_PERM)

            cos_true = mean_pairwise_cos(diff[good])
            cnull = np.fromiter((mean_pairwise_cos(diff[m]) for m in sub_masks),
                                float, N_PERM)
            cos_p = (1 + (cnull >= cos_true).sum()) / (1 + N_PERM)

            Z = (v - v.mean(0)) / (v.std(0) + 1e-6)
            loo_auc, _ = loo_projection_auc(Z, good_mask)

            geo.append(dict(layer=L, region=rname, dist_auc=dist_auc, dist_p=dist_p,
                            cos_true=cos_true, cos_null=cnull.mean(), cos_p=cos_p,
                            loo_auc=loo_auc,
                            dist_good=dist[good_mask].mean(), dist_bad=dist[~good_mask].mean()))
            probe_cache[(L, rname)] = v
        del X
        print(f"  layer {L:2d} done", flush=True)

    # ---- report tests 1 & 2 ----
    print("\n=== TEST 1&2: good-set silhouette (Euclid) vs random-4 null ===")
    print(f"  benchmark: trunk-z good-set silhouette ~ +{TRUNK_SIL:.2f}\n")
    print(f"  {'layer':>5} {'region':9} {'pool':7} {'sil':>7} {'null':>7} "
          f"{'p95':>7} {'excess':>7} {'p':>7}")
    for r in sorted(rows, key=lambda x: -x["excess"])[:18]:
        print(f"  {r['layer']:>5} {r['region']:9} {r['pool']:7} {r['sil']:+7.3f} "
              f"{r['null_mean']:+7.3f} {r['null_p95']:+7.3f} {r['excess']:+7.3f} "
              f"{r['p']:7.4f}")
    best = max(rows, key=lambda x: x["excess"])
    print(f"\n  best: layer {best['layer']} {best['region']}/{best['pool']}  "
          f"sil {best['sil']:+.3f}  excess {best['excess']:+.3f}  p {best['p']:.4f}")

    # ---- report test 3 ----
    print("\n=== TEST 3: default-well geometry (mean-pooled) ===")
    print("  dist_auc = P(good farther from well than bad);  cos = mean pairwise cosine "
          "of good (rep-base) dirs;  loo_auc = leave-one-out projection")
    print(f"  benchmark: mask omnibus AUC ~ {MASK_AUC:.2f}\n")
    print(f"  {'layer':>5} {'region':9} {'distAUC':>8} {'dist_p':>7} {'cosTrue':>8} "
          f"{'cosNull':>8} {'cos_p':>7} {'looAUC':>7}")
    for g in sorted(geo, key=lambda x: -x["loo_auc"])[:18]:
        print(f"  {g['layer']:>5} {g['region']:9} {g['dist_auc']:8.3f} {g['dist_p']:7.4f} "
              f"{g['cos_true']:+8.3f} {g['cos_null']:+8.3f} {g['cos_p']:7.4f} "
              f"{g['loo_auc']:7.3f}")
    best_geo = max(geo, key=lambda x: x["loo_auc"])
    print(f"\n  best LOO: layer {best_geo['layer']} {best_geo['region']}  "
          f"loo_auc {best_geo['loo_auc']:.3f}  dist_auc {best_geo['dist_auc']:.3f} "
          f"(p {best_geo['dist_p']:.4f})  cos {best_geo['cos_true']:+.3f} "
          f"(p {best_geo['cos_p']:.4f})")

    # ---- test 4: supervised probe at a couple of configs ----
    print("\n=== TEST 4: L1-logistic CV-AUC vs label-shuffle null (caveated, 4 good) ===")
    y = good_mask.astype(int)
    probe_rows = []
    cfgs = {(best_geo["layer"], best_geo["region"]),
            (best["layer"], best["region"]),
            (80, "all"), (80, "antigenC")}
    for (L, rname) in sorted(cfgs):
        Xp = probe_cache[(L, rname)]
        folds = make_folds(good_idx, bad_idx, k=4)
        obs = cv_auc_l1(Xp, y, folds, lam=0.02)
        null = np.empty(N_PERM_PROBE)
        for k in range(N_PERM_PROBE):
            yk = RNG.permutation(y)
            null[k] = cv_auc_l1(Xp, yk, folds, lam=0.02)
        pv = (1 + (null >= obs).sum()) / (1 + N_PERM_PROBE)
        probe_rows.append(dict(layer=L, region=rname, auc=obs, null=null.mean(), p=pv))
        print(f"  layer {L:>2} {rname:9}  CV-AUC {obs:.3f}  null {null.mean():.3f}"
              f"+/-{null.std():.3f}  p {pv:.4f}")

    # ---- CSV ----
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "layer", "region", "pool", "metric", "value", "p"])
        for r in rows:
            w.writerow(["sil", r["layer"], r["region"], r["pool"], "silhouette",
                        f"{r['sil']:.4f}", f"{r['p']:.4f}"])
        for g in geo:
            w.writerow(["geo", g["layer"], g["region"], "mean", "dist_auc",
                        f"{g['dist_auc']:.4f}", f"{g['dist_p']:.4f}"])
            w.writerow(["geo", g["layer"], g["region"], "mean", "cos_good",
                        f"{g['cos_true']:.4f}", f"{g['cos_p']:.4f}"])
            w.writerow(["geo", g["layer"], g["region"], "mean", "loo_auc",
                        f"{g['loo_auc']:.4f}", ""])
        for pr in probe_rows:
            w.writerow(["probe", pr["layer"], pr["region"], "mean", "cv_auc",
                        f"{pr['auc']:.4f}", f"{pr['p']:.4f}"])
    print(f"\nsaved -> {OUT_CSV}")

    # ---- gate ----
    sep_green = best["p"] < 0.05 and best["sil"] > 0
    dist_green = best_geo["dist_p"] < 0.05 and best_geo["dist_auc"] > 0.5
    dir_green = best_geo["cos_p"] < 0.05
    print("\n=== GATE ===")
    print(f"  [{'Y' if sep_green else 'N'}] separates (best sil {best['sil']:+.3f}, "
          f"p {best['p']:.4f}; trunk ~ +{TRUNK_SIL:.2f})")
    print(f"  [{'Y' if dist_green else 'N'}] sits away from well "
          f"(best dist_auc {best_geo['dist_auc']:.3f}, p {best_geo['dist_p']:.4f})")
    print(f"  [{'Y' if dir_green else 'N'}] consistent direction "
          f"(best cos {best_geo['cos_true']:+.3f}, p {best_geo['cos_p']:.4f})")
    verdict = "GREEN" if (dist_green and dir_green) else (
        "AMBER" if (sep_green or dist_green or dir_green) else "RED")
    print(f"  -> {verdict}")

    _plot(rows, geo, dvals, good_mask, probe_cache, best_geo)


def _plot(rows, geo, dvals, good_mask, probe_cache, best_geo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg_order = ["all", "antigenC", "epitope", "antibody"]
    colors = {"all": "#333333", "antigenC": "#d62728", "epitope": "#2ca02c",
              "antibody": "#1f77b4"}
    layers = sorted({r["layer"] for r in rows})

    def series(table, region, key):
        d = {r["layer"]: r[key] for r in table if r["region"] == region}
        return [d.get(L, np.nan) for L in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) silhouette vs layer (mean-pool)
    ax = axes[0, 0]
    mean_rows = [r for r in rows if r["pool"] == "mean"]
    for rg in reg_order:
        ax.plot(layers, series(mean_rows, rg, "sil"), "-o", ms=4,
                color=colors[rg], label=rg)
    ax.axhline(TRUNK_SIL, ls="--", color="purple", lw=1, label=f"trunk ~+{TRUNK_SIL}")
    ax.axhline(0, color="gray", lw=0.7)
    p95 = np.nanmean([r["null_p95"] for r in mean_rows])
    ax.axhspan(-abs(p95), abs(p95), color="gray", alpha=0.12, label="null +/-p95")
    ax.set_xlabel("ESMC layer"); ax.set_ylabel("good-set silhouette (mean-pool)")
    ax.set_title("TEST 1&2: does ESMC separate the good seeds?")
    ax.legend(fontsize=8)

    # (B) distance-from-well AUC vs layer
    ax = axes[0, 1]
    for rg in reg_order:
        ax.plot(layers, series(geo, rg, "dist_auc"), "-o", ms=4,
                color=colors[rg], label=rg)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("ESMC layer"); ax.set_ylabel("P(good farther from well)")
    ax.set_title("TEST 3a: do good seeds sit away from the default well?")
    ax.legend(fontsize=8)

    # (C) LOO projection AUC + direction cosine
    ax = axes[1, 0]
    for rg in reg_order:
        ax.plot(layers, series(geo, rg, "loo_auc"), "-o", ms=4,
                color=colors[rg], label=f"{rg} LOO-AUC")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.axhline(MASK_AUC, ls="--", color="orange", lw=1, label=f"mask ~{MASK_AUC}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("ESMC layer"); ax.set_ylabel("leave-one-out projection AUC")
    ax.set_title("TEST 3c: separable along mean(good)-mean(bad)?")
    ax.legend(fontsize=8)

    # (D) at best config: dist-from-well, good vs bad
    ax = axes[1, 1]
    L, rg = best_geo["layer"], best_geo["region"]
    lo, hi = REGIONS[rg]
    v = probe_cache[(L, rg)]
    xb = load_layer(BASE, L)[lo:hi, :].mean(0)
    dist = np.linalg.norm(v - xb, axis=1)
    gb = good_mask
    jitter = RNG.normal(0, 0.04, size=len(dist))
    ax.scatter(0 + jitter[~gb], dist[~gb], s=28, c="#7fa8d0", alpha=0.7, label="bad")
    ax.scatter(1 + jitter[gb], dist[gb], s=80, c="#d62728", label="good")
    for s in np.where(gb)[0]:
        ax.annotate(str(s), (1 + jitter[s], dist[s]), fontsize=7,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["bad", "good"])
    ax.set_ylabel("||rep - default well||")
    ax.set_title(f"TEST 3a detail: layer {L} {rg}  "
                 f"(AUC {best_geo['dist_auc']:.2f}, p {best_geo['dist_p']:.3f})")
    ax.legend(fontsize=8)

    fig.suptitle("7txw: is the orientation signal in the ESMC rep as displacement from "
                 "the default well?", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
