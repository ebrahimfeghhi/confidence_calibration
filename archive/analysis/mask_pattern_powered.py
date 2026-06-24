"""
Do "good" 7txw seeds share an input-mask pattern? (powered first pass)

The new 7txw_tts2000 run + the 7txw_reps set give 2060 seeds, each with a
deterministic, reproducible input mask (~10% of 609 residues drawn i.i.d.
Bernoulli) and a scored A|C DockQ outcome. The old n=4-good analysis was
underpowered; this tests *with power* whether masking specific residues (or any
combination of residues) makes a seed reach the good antigen-orientation basin.

Why not PCA: the masks are i.i.d. across residues, so mask-mask covariance is
~0.09*I (flat spectrum, no real PCs) and PCA is unsupervised. The signal we want
is a mask<->outcome correlation -> a supervised feature-selection problem.

Pipeline (esm venv + a CUDA device, so the RNG matches the real runs):
  0. Confound check: total / per-chain masked fraction, good vs bad (Mann-Whitney).
  1. Per-residue enrichment: f_good vs f_bad per residue -> Fisher's exact + BH-FDR.
  2. Omnibus: L1-logistic M->good, 5-fold CV AUC vs a label-shuffle null.

Outcome per seed = ORACLE best-of-5 (max DockQ over the seed's 5 samples), A|C.
Labels: good = DockQ >= --good (0.7), bad = DockQ < --bad (0.23); middle dropped.

Outputs:
  results/csv_files/mask_pattern_7txw_per_residue.csv
  analysis/figs/mask_pattern_7txw.png
  (+ printed summary)

Usage:
    CUDA_VISIBLE_DEVICES=3 ~/esm/.venv/bin/python analysis/mask_pattern_powered.py
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))

from paths import CSV_DIR, RESULTS_DIR, GROUND_TRUTHS
import analyze_masks as am

CHAIN_LETTER = {0: "A", 1: "B", 2: "C"}


# ---------------------------------------------------------------------------
# Outcomes: oracle best-of-5 A|C DockQ per seed, straight from the combined CSV
# ---------------------------------------------------------------------------

def oracle_dockq(csv_path: Path, complex_prefix: str, c1: str, c2: str) -> dict:
    """{seed_int: max DockQ over that seed's samples} for the c1|c2 interface."""
    best = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["assembly_id"].startswith(complex_prefix):
                continue
            if {row["chain1"], row["chain2"]} != {c1, c2}:
                continue
            try:
                seed = int(row["seed"])
                dq = float(row["dockq"])
            except (ValueError, KeyError):
                continue
            if seed not in best or dq > best[seed]:
                best[seed] = dq
    return best


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a 1-D array of p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DIR / "dockq_ost_7txw_tts2060_all_samples.csv"))
    ap.add_argument("--complex", default="7txw", help="assembly_id prefix")
    ap.add_argument("--chain1", default="A", help="antibody chain (interface side 1)")
    ap.add_argument("--chain2", default="C", help="antigen chain (interface side 2)")
    ap.add_argument("--good", type=float, default=0.7)
    ap.add_argument("--bad", type=float, default=0.23)
    ap.add_argument("--n-perm", type=int, default=200, help="label shuffles for the omnibus null")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L1 strength for logistic")
    args = ap.parse_args()

    if am.torch is None:
        sys.exit("torch/esm not importable -- run with ~/esm/.venv/bin/python on a GPU.")

    from scipy.stats import fisher_exact, mannwhitneyu

    # --- outcomes + good/bad split ---
    csv_path = Path(args.csv)
    dq = oracle_dockq(csv_path, args.complex, args.chain1, args.chain2)
    seeds = sorted(dq)
    good = [s for s in seeds if dq[s] >= args.good]
    bad = [s for s in seeds if dq[s] < args.bad]
    mid = [s for s in seeds if args.bad <= dq[s] < args.good]
    print(f"{args.complex} {args.chain1}|{args.chain2}  oracle best-of-5 over {len(seeds)} seeds")
    print(f"  good (>= {args.good}) = {len(good)}   bad (< {args.bad}) = {len(bad)}   "
          f"dropped middle = {len(mid)}")
    if len(good) < 5 or len(bad) < 5:
        sys.exit("need >= 5 seeds in each group.")

    # --- regenerate masks for every scored seed (deterministic, CUDA RNG) ---
    builder = am.ESMFold2InputBuilder()
    spi, model_seqs = am.build_spi(args.complex)
    features, _ = builder.prepare_input(spi, seed=0, device=am.DEVICE)
    lm_input_ids, expand_maps = am.build_lm_input_ids(features)
    seed_to_masked = am.generate_masks(lm_input_ids, features, expand_maps, seeds=seeds)

    residues = sorted(am.all_residues(features))          # fixed column order
    n_res = len(residues)
    col_of = {r: i for i, r in enumerate(residues)}
    chain_arr = np.array([r[0] for r in residues])        # asym per column
    print(f"residues: {n_res}   per-chain: "
          + "  ".join(f"{CHAIN_LETTER.get(a,a)}={int((chain_arr==a).sum())}"
                      for a in sorted(set(chain_arr.tolist()))))

    def build_M(seed_list):
        M = np.zeros((len(seed_list), n_res), dtype=np.int8)
        for i, s in enumerate(seed_list):
            for r in seed_to_masked[s]:
                M[i, col_of[r]] = 1
        return M

    Mg, Mb = build_M(good), build_M(bad)
    p_global = float(np.r_[Mg, Mb].mean())
    print(f"global masked fraction (chance baseline) = {p_global:.4f}")

    # --- native interface residues (A|C and B|C) for region annotation ---
    region = {}  # col -> 'epitope' | 'paratope'
    try:
        gt_cif = GROUND_TRUTHS / f"{args.complex}-assembly1.cif"
        for ab in (args.chain1, "B"):
            s1, s2 = am.get_interface_ref(gt_cif, model_seqs, ab, args.chain2)
            for lbl, pos in s1:                       # antibody side -> paratope
                r = (ord(lbl) - ord("A"), pos)
                if r in col_of:
                    region.setdefault(col_of[r], "paratope")
            for lbl, pos in s2:                       # antigen side -> epitope
                r = (ord(lbl) - ord("A"), pos)
                if r in col_of:
                    region[col_of[r]] = "epitope"     # epitope wins over paratope
    except Exception as e:
        print(f"(interface annotation skipped: {e})")

    # =====================================================================
    # 0. Confound check
    # =====================================================================
    print("\n=== 0. Global confounds (good vs bad) ===")
    print(f"{'feature':22s} {'good':>8s} {'bad':>8s} {'MWU-p':>9s}")

    def mwu(gv, bv):
        try:
            return float(mannwhitneyu(gv, bv, alternative="two-sided").pvalue)
        except ValueError:
            return float("nan")

    ng_tot, nb_tot = Mg.sum(1), Mb.sum(1)
    print(f"{'n_masked_total':22s} {ng_tot.mean():8.2f} {nb_tot.mean():8.2f} "
          f"{mwu(ng_tot, nb_tot):9.3g}")
    for a in sorted(set(chain_arr.tolist())):
        cols = chain_arr == a
        gv = Mg[:, cols].mean(1)
        bv = Mb[:, cols].mean(1)
        print(f"{'frac_masked_chain_' + CHAIN_LETTER.get(a, str(a)):22s} "
              f"{gv.mean():8.4f} {bv.mean():8.4f} {mwu(gv, bv):9.3g}")

    # =====================================================================
    # 1. Per-residue enrichment: Fisher's exact + BH-FDR
    # =====================================================================
    f_good = Mg.mean(0)
    f_bad = Mb.mean(0)
    ng, nb = len(good), len(bad)
    a_cnt = Mg.sum(0)            # good & masked
    c_cnt = Mb.sum(0)           # bad  & masked
    pvals = np.ones(n_res)
    for i in range(n_res):
        table = [[a_cnt[i], ng - a_cnt[i]], [c_cnt[i], nb - c_cnt[i]]]
        pvals[i] = fisher_exact(table, alternative="two-sided")[1]
    qvals = bh_fdr(pvals)
    with np.errstate(divide="ignore", invalid="ignore"):
        risk_ratio = np.where(f_bad > 0, f_good / f_bad, np.inf)
    diff = f_good - f_bad

    n_hit = int((qvals < 0.05).sum())
    print(f"\n=== 1. Per-residue enrichment ({n_res} residues, Fisher + BH-FDR) ===")
    print(f"  mean f_good = {f_good.mean():.4f}   mean f_bad = {f_bad.mean():.4f}   "
          f"(chance {p_global:.4f})")
    print(f"  residues with FDR q < 0.05: {n_hit}")
    show = np.argsort(qvals)[:15]
    print(f"  {'chain':5s} {'res':>5s} {'region':9s} {'f_good':>7s} {'f_bad':>7s} "
          f"{'RR':>6s} {'fisher_p':>10s} {'fdr_q':>9s}")
    for i in show:
        a, pos = residues[i]
        print(f"  {CHAIN_LETTER.get(a, a):5s} {pos + 1:5d} {region.get(i, ''):9s} "
              f"{f_good[i]:7.3f} {f_bad[i]:7.3f} {risk_ratio[i]:6.2f} "
              f"{pvals[i]:10.2e} {qvals[i]:9.3f}")

    # write per-residue table
    out_csv = CSV_DIR / "mask_pattern_7txw_per_residue.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["asym_idx", "chain", "residue_1based", "region",
                    "f_good", "f_bad", "diff", "risk_ratio", "fisher_p", "fdr_q"])
        for i, (a, pos) in enumerate(residues):
            w.writerow([a, CHAIN_LETTER.get(a, a), pos + 1, region.get(i, ""),
                        f"{f_good[i]:.4f}", f"{f_bad[i]:.4f}", f"{diff[i]:.4f}",
                        f"{risk_ratio[i]:.3f}", f"{pvals[i]:.3e}", f"{qvals[i]:.3e}"])
    print(f"  saved -> {out_csv}")

    # =====================================================================
    # 2. Omnibus: L1-logistic CV-AUC vs label-shuffle null
    # =====================================================================
    print(f"\n=== 2. Omnibus L1-logistic M->good (CV AUC vs {args.n_perm}-shuffle null) ===")
    import warnings
    # sklearn 1.9 deprecates penalty= in favour of l1_ratio=; penalty="l1" still
    # applies a true L1 here (most coefficients end up exactly 0), so silence the
    # noisy per-fit FutureWarning / "Inconsistent values" UserWarning.
    warnings.filterwarnings("ignore", message="'penalty' was deprecated")
    warnings.filterwarnings("ignore", message="Inconsistent values: penalty")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    X = np.r_[Mg, Mb].astype(np.float64)
    y = np.r_[np.ones(ng), np.zeros(nb)].astype(int)
    clf = LogisticRegression(penalty="l1", solver="liblinear",
                             class_weight="balanced", C=args.C, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    obs = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())

    rng = np.random.default_rng(0)
    null = np.empty(args.n_perm)
    for k in range(args.n_perm):
        yk = rng.permutation(y)
        cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=k + 1)
        null[k] = cross_val_score(clf, X, yk, cv=cvk, scoring="roc_auc").mean()
    pval = (np.sum(null >= obs) + 1) / (args.n_perm + 1)
    print(f"  observed 5-fold CV AUC = {obs:.4f}")
    print(f"  shuffle null AUC       = {null.mean():.4f} +/- {null.std():.4f}  "
          f"(95th pct {np.percentile(null, 95):.4f})")
    print(f"  permutation p          = {pval:.4f}  "
          f"({'beats null' if pval < 0.05 else 'NOT beyond chance'})")

    # nonzero coefficients from a full-data fit (descriptive)
    clf.fit(X, y)
    coef = clf.coef_.ravel()
    nz = np.nonzero(coef)[0]
    print(f"  nonzero L1 coefficients: {len(nz)} / {n_res}")
    if len(nz):
        top = nz[np.argsort(-np.abs(coef[nz]))][:15]
        print(f"  {'chain':5s} {'res':>5s} {'region':9s} {'coef':>8s} "
              f"{'f_good':>7s} {'f_bad':>7s}")
        for i in top:
            a, pos = residues[i]
            print(f"  {CHAIN_LETTER.get(a, a):5s} {pos + 1:5d} {region.get(i, ''):9s} "
                  f"{coef[i]:8.3f} {f_good[i]:7.3f} {f_bad[i]:7.3f}")

    # =====================================================================
    # Figure
    # =====================================================================
    try:
        _plot(residues, chain_arr, region, f_good, f_bad, diff, qvals,
              obs, null, ng, nb, args)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(residues, chain_arr, region, f_good, f_bad, diff, qvals,
          obs, null, ng, nb, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_res = len(residues)
    hit = qvals < 0.05
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) f_bad vs f_good scatter, region-colored, FDR hits ringed
    ax = axes[0]
    base = np.array([region.get(i, "other") for i in range(n_res)])
    colors = {"epitope": "seagreen", "paratope": "royalblue", "other": "lightgray"}
    for rl, col in colors.items():
        m = base == rl
        ax.scatter(f_bad[m], f_good[m], s=14, alpha=0.5, color=col, label=rl)
    if hit.any():
        ax.scatter(f_bad[hit], f_good[hit], s=70, facecolors="none",
                   edgecolors="red", lw=1.3, label="FDR<0.05")
    lim = max(0.3, f_good.max() * 1.1, f_bad.max() * 1.1)
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("P(masked | bad)"); ax.set_ylabel("P(masked | good)")
    ax.set_title(f"per-residue mask frequency\ngood={ng}  bad={nb}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) Delta along sequence, banded by chain, interface + FDR marked
    ax = axes[1]
    x = np.arange(n_res)
    ax.bar(x, diff, width=1.0, color="silver")
    band = {0: "#fde0dd", 1: "#e0ecf4", 2: "#e5f5e0"}
    a0 = 0
    for a in sorted(set(chain_arr.tolist())):
        cols = np.nonzero(chain_arr == a)[0]
        lo, hi = cols.min(), cols.max()
        ax.axvspan(lo - 0.5, hi + 0.5, color=band.get(a, "#eee"), alpha=0.5, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], f"chain {CHAIN_LETTER.get(a, a)}",
                ha="center", va="top", fontsize=9)
    epi = np.array([region.get(i) == "epitope" for i in range(n_res)])
    par = np.array([region.get(i) == "paratope" for i in range(n_res)])
    ymin = diff.min()
    ax.scatter(x[epi], np.full(epi.sum(), ymin), s=6, color="seagreen", marker="^",
               label="epitope")
    ax.scatter(x[par], np.full(par.sum(), ymin), s=6, color="royalblue", marker="v",
               label="paratope")
    if hit.any():
        ax.scatter(x[hit], diff[hit], s=40, facecolors="none", edgecolors="red",
                   lw=1.2, label="FDR<0.05")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("residue (concatenated A|B|C)")
    ax.set_ylabel("f_good - f_bad")
    ax.set_title("per-residue enrichment along sequence")
    ax.legend(fontsize=8, loc="lower right")

    # (c) omnibus null vs observed
    ax = axes[2]
    ax.hist(null, bins=30, color="lightsteelblue", edgecolor="gray")
    ax.axvline(obs, color="red", lw=2, label=f"observed {obs:.3f}")
    ax.axvline(0.5, color="black", ls=":", lw=1, label="chance 0.5")
    ax.set_xlabel("5-fold CV AUC")
    ax.set_ylabel(f"# of {len(null)} label shuffles")
    ax.set_title("omnibus L1-logistic: mask -> good\n(observed vs shuffle null)")
    ax.legend(fontsize=8)

    fig.suptitle(f"{args.complex} {args.chain1}|{args.chain2}: does the input mask "
                 f"predict the good basin? (oracle best-of-5)", y=1.02)
    out_png = _p / "analysis" / "figs" / "mask_pattern_7txw.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure -> {out_png}")


if __name__ == "__main__":
    main()
