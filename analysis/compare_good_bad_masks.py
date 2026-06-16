"""
Do "good" seeds mask similar regions to each other, and differently from "bad" seeds?

For one interface we split the 50 seeds into good (per-seed DockQ >= --good) and
bad (< --bad), regenerate each seed's input mask deterministically (same path as
analyze_masks.py -> needs the esm venv + a CUDA device so the RNG matches the
real runs), then quantify mask similarity two ways:

  (1) Pairwise overlap (Jaccard).  Each seed's mask = the SET of masked residues.
      For two seeds, similarity = |A & B| / |A | B|. We compare the mean Jaccard of
      good-good pairs vs bad-bad pairs vs good-bad (cross) pairs, against the
      chance level. Because only ~10% of residues are masked, two independent
      masks already overlap ~p/(2-p) ~ 0.05 by chance, so the all-pairs mean is the
      chance reference. "Good seeds mask similar regions" => within-good > cross,
      and we get a permutation p-value by shuffling the good/bad labels.

  (2) Per-residue masking frequency.  For each residue, f_good = fraction of good
      seeds that mask it, f_bad = fraction of bad seeds. Residues with large
      f_good - f_bad are the regions good seeds consistently mask (or, if negative,
      consistently spare). This says WHICH regions, not just whether they overlap.

Usage (esm venv + GPU; pick a device with CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=3 ~/esm/.venv/bin/python analysis/compare_good_bad_masks.py \
        --complex 8q6r --chain1 A --chain2 D
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "plot"))
sys.path.insert(0, str(_p / "analysis"))

from paths import CSV_DIR, RESULTS_DIR
from plot_scaling_ost import load_data
import analyze_masks as am  # build_spi, build_lm_input_ids, generate_masks, all_residues


def per_seed_dockq(complex_id, c1, c2, conf_col="pair_iptm"):
    """{seed_int: confidence-selected DockQ} for one interface from the tts CSV."""
    by, _ = load_data(CSV_DIR / "dockq_ost_esmfold2fast_tts_all_samples.csv", conf_col)
    out = {}
    for (aid, x1, x2), recs in by.items():
        if aid.split("-")[0] == complex_id and {x1, x2} == {c1, c2}:
            for seed, _score, dq in recs:
                try:
                    out[int(seed)] = dq
                except ValueError:
                    pass
    return out


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def mean_jaccard(masks, pairs):
    vals = [jaccard(masks[i], masks[j]) for i, j in pairs]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True)
    ap.add_argument("--chain1", required=True)
    ap.add_argument("--chain2", required=True)
    ap.add_argument("--good", type=float, default=0.5, help="DockQ >= this = good seed")
    ap.add_argument("--bad", type=float, default=0.23, help="DockQ < this = bad seed")
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()

    if am.torch is None:
        sys.exit("torch/esm not importable — run this with ~/esm/.venv/bin/python on a GPU.")

    # --- 1. good/bad split from DockQ ---
    dq = per_seed_dockq(args.complex, args.chain1, args.chain2)
    good = sorted(s for s, v in dq.items() if v >= args.good)
    bad = sorted(s for s, v in dq.items() if v < args.bad)
    print(f"{args.complex} {args.chain1}|{args.chain2}: "
          f"{len(good)} good (DockQ>={args.good}) / {len(bad)} bad (<{args.bad}) "
          f"of {len(dq)} seeds")
    if len(good) < 2 or len(bad) < 2:
        sys.exit("need >=2 seeds in each group.")

    # --- 2. regenerate masks for all seeds (deterministic, CUDA RNG) ---
    builder = am.ESMFold2InputBuilder()
    spi, _model_seqs = am.build_spi(args.complex)
    features, _ = builder.prepare_input(spi, seed=0, device=am.DEVICE)
    lm_input_ids, expand_maps = am.build_lm_input_ids(features)
    seed_to_masked = am.generate_masks(lm_input_ids, features, expand_maps)
    n_res = len(am.all_residues(features))
    sizes = [len(seed_to_masked[s]) for s in dq]
    print(f"residues: {n_res}   masked/seed: mean={np.mean(sizes):.1f} "
          f"({100*np.mean(sizes)/n_res:.1f}%)")

    # --- 3. pairwise Jaccard: within-good vs within-bad vs cross vs chance ---
    gg = list(itertools.combinations(good, 2))
    bb = list(itertools.combinations(bad, 2))
    cross = [(i, j) for i in good for j in bad]
    allp = list(itertools.combinations(list(dq), 2))
    m_gg = mean_jaccard(seed_to_masked, gg)
    m_bb = mean_jaccard(seed_to_masked, bb)
    m_cx = mean_jaccard(seed_to_masked, cross)
    m_all = mean_jaccard(seed_to_masked, allp)
    p = np.mean(sizes) / n_res
    print("\n=== Pairwise mask overlap (Jaccard) ===")
    print(f"  within-good : {m_gg:.4f}   ({len(gg)} pairs)")
    print(f"  within-bad  : {m_bb:.4f}   ({len(bb)} pairs)")
    print(f"  cross g/b   : {m_cx:.4f}   ({len(cross)} pairs)")
    print(f"  all pairs   : {m_all:.4f}   (chance reference)")
    print(f"  chance ~p/(2-p) = {p/(2-p):.4f}")

    # permutation test: is within-good - cross beyond random label assignment?
    labels = np.array([1] * len(good) + [0] * len(bad))
    seeds_gb = good + bad
    obs = m_gg - m_cx
    rng = np.random.default_rng(0)
    null = np.empty(args.n_perm)
    for k in range(args.n_perm):
        perm = rng.permutation(labels)
        g = [seeds_gb[i] for i in range(len(perm)) if perm[i] == 1]
        b = [seeds_gb[i] for i in range(len(perm)) if perm[i] == 0]
        gg_p = list(itertools.combinations(g, 2))
        cx_p = [(i, j) for i in g for j in b]
        null[k] = mean_jaccard(seed_to_masked, gg_p) - mean_jaccard(seed_to_masked, cx_p)
    pval = (np.sum(null >= obs) + 1) / (args.n_perm + 1)
    print(f"\n  within-good - cross = {obs:+.4f}   permutation p = {pval:.4f} "
          f"({'good seeds ARE more mutually similar' if pval < 0.05 else 'not beyond chance'})")

    # --- 4. per-residue masking frequency contrast ---
    res = sorted(set().union(*seed_to_masked.values()))
    f_good = {r: np.mean([r in seed_to_masked[s] for s in good]) for r in res}
    f_bad = {r: np.mean([r in seed_to_masked[s] for s in bad]) for r in res}
    diff = {r: f_good[r] - f_bad[r] for r in res}
    fg = np.array([f_good[r] for r in res]); fb = np.array([f_bad[r] for r in res])
    print("\n=== Per-residue masking frequency ===")
    print(f"  corr(f_good, f_bad) across {len(res)} residues = {np.corrcoef(fg, fb)[0,1]:+.3f}")
    top = sorted(res, key=lambda r: diff[r], reverse=True)[:10]
    bot = sorted(res, key=lambda r: diff[r])[:10]
    print("  residues masked MORE by good seeds (asym,res: f_good vs f_bad):")
    for r in top:
        print(f"    chain{r[0]} res{r[1]:>4}: good {f_good[r]:.2f}  bad {f_bad[r]:.2f}  Δ {diff[r]:+.2f}")
    print("  residues masked LESS by good seeds:")
    for r in bot:
        print(f"    chain{r[0]} res{r[1]:>4}: good {f_good[r]:.2f}  bad {f_bad[r]:.2f}  Δ {diff[r]:+.2f}")

    # save per-residue table + scatter
    out_csv = CSV_DIR / f"mask_goodbad_{args.complex}_{args.chain1}{args.chain2}.csv"
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["asym_idx", "residue_index", "f_good", "f_bad", "diff"])
        for r in res:
            w.writerow([r[0], r[1], f"{f_good[r]:.4f}", f"{f_bad[r]:.4f}", f"{diff[r]:.4f}"])
    print(f"\nsaved per-residue table -> {out_csv}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(fb, fg, s=14, alpha=0.5, color="steelblue")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("P(masked | bad seed)"); ax.set_ylabel("P(masked | good seed)")
        ax.set_title(f"{args.complex} {args.chain1}|{args.chain2}: per-residue mask frequency\n"
                     f"good={len(good)} bad={len(bad)} seeds")
        ax.set_xlim(0, max(0.4, fb.max()*1.1)); ax.set_ylim(0, max(0.4, fg.max()*1.1))
        ax.grid(True, alpha=0.3)
        out_png = RESULTS_DIR / f"mask_goodbad_{args.complex}_{args.chain1}{args.chain2}.png"
        plt.tight_layout(); plt.savefig(out_png, dpi=150)
        print(f"saved scatter -> {out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
