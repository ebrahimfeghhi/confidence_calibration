"""
Did the few "good" seeds mask KEY residues (and the bad seeds not)?

Simple, interpretable method for one interface (default 7txw A|C):

  1. Reproduce every seed's deterministic input mask (same path as analyze_masks;
     needs the esm venv + a CUDA device so the RNG matches the real runs).
  2. Define a handful of structurally meaningful residue sets:
       - true epitope  : antigen residues at the NATIVE interface (GT, 8A)
       - true paratope : antibody residues at the NATIVE interface (GT, 8A)
       - decoy epitope : antigen residues at the interface in the BAD seeds'
                         predicted structures (the wrong face the failures dock
                         through; majority vote over bad seeds)
  3. For each set S, compare the mean FRACTION of S masked by good vs bad seeds.
     That single contrast answers "did good seeds mask these key residues?".
  4. Also rank individual residues by f_good - f_bad and tag each by role.

Because only ~10% of residues are masked, the chance fraction-masked of any set is
~0.10; a region masked much more (or less) by good seeds than bad is the signal.
With only a few good seeds this is descriptive — a label-shuffle p-value is printed
as a sanity check, not a heavy test.

Usage (esm venv + GPU):
    CUDA_VISIBLE_DEVICES=3 ~/esm/.venv/bin/python analysis/key_residue_masking.py \
        --complex 7txw-assembly1 --chain1 A --chain2 C
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "plot"))
sys.path.insert(0, str(_p / "analysis"))

from paths import CSV_DIR, RESULTS_DIR, GROUND_TRUTHS, ESM_PREDICTIONS
import analyze_masks as am
from compare_good_bad_masks import per_seed_dockq
from compare_7txw_poses import parse_cif_ca


def group_mean_frac(seed_to_masked, seeds, S):
    if not S:
        return float("nan")
    return float(np.mean([len(seed_to_masked[s] & S) / len(S) for s in seeds]))


def perm_pvalue(seed_to_masked, good, bad, S, n_perm=20000, seed=0):
    """Two-sided label-shuffle p for (good-mean - bad-mean) fraction-masked of S."""
    if not S:
        return float("nan")
    fr = {s: len(seed_to_masked[s] & S) / len(S) for s in good + bad}
    obs = np.mean([fr[s] for s in good]) - np.mean([fr[s] for s in bad])
    alls = good + bad
    ng = len(good)
    rng = np.random.default_rng(seed)
    vals = np.array([fr[s] for s in alls])
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(alls))
        null[k] = vals[perm[:ng]].mean() - vals[perm[ng:]].mean()
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True, help="e.g. 7txw-assembly1")
    ap.add_argument("--chain1", required=True, help="antibody chain (ref label)")
    ap.add_argument("--chain2", required=True, help="antigen chain (ref label)")
    ap.add_argument("--good", type=float, default=0.5, help="DockQ >= this = good seed")
    ap.add_argument("--bad", type=float, default=0.23, help="DockQ < this = bad seed")
    ap.add_argument("--cutoff", type=float, default=8.0, help="interface CA distance (A)")
    ap.add_argument("--decoy-frac", type=float, default=0.5,
                    help="antigen residue is 'decoy epitope' if contacted in >= this "
                         "fraction of bad seeds")
    args = ap.parse_args()

    if am.torch is None:
        sys.exit("torch/esm not importable — run with ~/esm/.venv/bin/python on a GPU.")

    base = args.complex.split("-")[0]
    antibody_lbl, antigen_lbl = args.chain1, args.chain2

    # --- 1. good/bad split from DockQ ---
    dq = per_seed_dockq(base, args.chain1, args.chain2)
    good = sorted(s for s, v in dq.items() if v >= args.good)
    bad = sorted(s for s, v in dq.items() if v < args.bad)
    print(f"{args.complex} {args.chain1}|{args.chain2}: "
          f"good (DockQ>={args.good}) = {good}  ({len(good)})   "
          f"bad (<{args.bad}) = {len(bad)} of {len(dq)} seeds")
    if len(good) < 1 or len(bad) < 2:
        sys.exit("need >=1 good and >=2 bad seeds.")

    # --- 2. reproduce masks deterministically (CUDA RNG) ---
    builder = am.ESMFold2InputBuilder()
    spi, model_seqs = am.build_spi(base)
    features, _ = builder.prepare_input(spi, seed=0, device=am.DEVICE)
    lm_input_ids, expand_maps = am.build_lm_input_ids(features)
    seed_to_masked = am.generate_masks(lm_input_ids, features, expand_maps)
    all_res = am.all_residues(features)
    n_total = len(all_res)
    p_global = float(np.mean([len(seed_to_masked[s]) for s in dq]) / n_total)
    print(f"residues: {n_total}   global masked frac (chance baseline) = {p_global:.3f}")

    # chain_mapping (ref label -> model letter); identity for 7txw
    per_seed, _baseline = am.load_dockq(args.complex, args.chain1, args.chain2)
    cm = per_seed[good[0]][1] if good and good[0] in per_seed else {c: c for c in "ABCDEFGH"}
    asym_of = {lbl: am.cm_asym(cm, lbl) for lbl in cm}
    label_of = {a: lbl for lbl, a in asym_of.items() if a is not None}
    antibody_asym = asym_of.get(antibody_lbl)
    antigen_asym = asym_of.get(antigen_lbl)

    antigen_all = {r for r in all_res if r[0] == antigen_asym}
    antibody_all = {r for r in all_res if r[0] == antibody_asym}

    # --- 3a. true epitope / paratope from ground truth (reused mapping) ---
    gt_cif = GROUND_TRUTHS / f"{args.complex}.cif"
    side1, side2 = am.get_interface_ref(gt_cif, model_seqs, args.chain1, args.chain2,
                                        cutoff=args.cutoff)
    by_label = {args.chain1: side1, args.chain2: side2}

    def to_keys(ref_set):
        out = set()
        for ref_label, pos in ref_set:
            a = am.cm_asym(cm, ref_label)
            if a is not None:
                out.add((a, pos))
        return out

    true_epitope = to_keys(by_label[antigen_lbl])
    true_paratope = to_keys(by_label[antibody_lbl])

    # --- 3b. decoy epitope from BAD seeds' predicted structures ---
    # antigen residues whose CA is within cutoff of ANY antibody-chain CA, in
    # >= decoy_frac of bad seeds. predicted CIFs number each chain 1..N, so
    # model_pos0based = auth_seq_id - 1 (same space as get_interface_ref positions).
    pred_dir = ESM_PREDICTIONS / "esmfold2fast_tts" / base
    contact_count = defaultdict(int)
    n_bad_used = 0
    maxres_seen = 0
    for s in bad:
        f = pred_dir / f"seed_{s}" / "predictions" / f"{base}_sample_0.cif"
        if not f.exists():
            continue
        st = parse_cif_ca(f)
        if antigen_lbl not in st:
            continue
        ab_xyz = np.array([v for ch, res in st.items() if ch != antigen_lbl
                           for v in res.values()])
        if len(ab_xyz) == 0:
            continue
        for k, v in st[antigen_lbl].items():
            maxres_seen = max(maxres_seen, k)
            if np.sqrt(((ab_xyz - v) ** 2).sum(1)).min() < args.cutoff:
                contact_count[k] += 1
        n_bad_used += 1
    decoy_resids = {k for k, c in contact_count.items()
                    if c >= args.decoy_frac * max(n_bad_used, 1)}
    decoy_epitope = {(antigen_asym, k - 1) for k in decoy_resids}
    # sanity: antigen residue numbering should be contiguous 1..len
    print(f"decoy built from {n_bad_used} bad seeds; antigen max auth_seq_id={maxres_seen} "
          f"(antigen_all={len(antigen_all)})")

    regions = {
        "true_epitope": true_epitope,
        "true_paratope": true_paratope,
        "decoy_epitope": decoy_epitope,
        "antigen_all": antigen_all,
        f"antibody_{antibody_lbl}_all": antibody_all,
        "all_residues": all_res,
    }
    print("\nkey-set sizes:  "
          + "  ".join(f"{k}={len(v)}" for k, v in regions.items()))
    print(f"  overlap true_epitope ∩ decoy_epitope = "
          f"{len(true_epitope & decoy_epitope)} residues")

    # --- 4. group-mean fraction masked: good vs bad ---
    print(f"\n=== Fraction of each region masked (mean over seeds) ===")
    print(f"{'region':22s} {'size':>5s} {'good':>7s} {'bad':>7s} {'Δ(g-b)':>8s} {'perm_p':>7s}")
    for name, S in regions.items():
        g = group_mean_frac(seed_to_masked, good, S)
        b = group_mean_frac(seed_to_masked, bad, S)
        pv = perm_pvalue(seed_to_masked, good, bad, S) if name != "all_residues" else float("nan")
        print(f"{name:22s} {len(S):5d} {g:7.3f} {b:7.3f} {g-b:+8.3f} {pv:7.3f}")

    # --- 5. per-residue contrast (over all residues; tag key roles) ---
    role = {}
    for r in true_paratope:
        role[r] = "true_paratope"
    for r in true_epitope:
        role[r] = "true_epitope"
    for r in decoy_epitope:
        role.setdefault(r, "decoy_epitope")  # don't overwrite a true-iface label

    f_good = {r: float(np.mean([r in seed_to_masked[s] for s in good])) for r in all_res}
    f_bad = {r: float(np.mean([r in seed_to_masked[s] for s in bad])) for r in all_res}
    diff = {r: f_good[r] - f_bad[r] for r in all_res}

    def show(rs, label):
        print(f"\n  {label}")
        print(f"    {'chain':5s} {'res':>4s} {'role':14s} {'f_good':>6s} {'f_bad':>6s} {'Δ':>6s}")
        for r in rs:
            a, pos = r
            print(f"    {label_of.get(a, '?'):5s} {pos+1:4d} {role.get(r,''):14s} "
                  f"{f_good[r]:6.2f} {f_bad[r]:6.2f} {diff[r]:+6.2f}")

    order = sorted(all_res, key=lambda r: diff[r])
    show(order[::-1][:12], "residues masked MORE by good seeds (top 12):")
    show(order[:12], "residues masked LESS by good seeds (top 12):")

    # also: of residues masked by ALL good seeds, which do bad seeds rarely mask?
    all_good_masked = [r for r in all_res if f_good[r] == 1.0]
    standout = sorted([r for r in all_good_masked if f_bad[r] <= 0.2],
                      key=lambda r: f_bad[r])
    print(f"\n  residues masked by ALL {len(good)} good seeds but <=20% of bad seeds: "
          f"{len(standout)}")
    show(standout[:12], "  (chance of all-good by luck ~ p^n_good)")

    # --- save per-residue table ---
    out_csv = CSV_DIR / f"key_residue_masking_{base}_{args.chain1}{args.chain2}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["asym_idx", "chain", "residue_1based", "role", "f_good", "f_bad", "diff"])
        for r in sorted(all_res):
            a, pos = r
            w.writerow([a, label_of.get(a, "?"), pos + 1, role.get(r, ""),
                        f"{f_good[r]:.4f}", f"{f_bad[r]:.4f}", f"{diff[r]:.4f}"])
    print(f"\nsaved per-residue table -> {out_csv}")

    # --- plots: region bar contrast + per-residue scatter ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["true_epitope", "true_paratope", "decoy_epitope",
                 "antigen_all", f"antibody_{antibody_lbl}_all"]
        gv = [group_mean_frac(seed_to_masked, good, regions[n]) for n in names]
        bv = [group_mean_frac(seed_to_masked, bad, regions[n]) for n in names]
        x = np.arange(len(names))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(x - 0.2, gv, 0.4, label=f"good ({len(good)})", color="seagreen")
        ax1.bar(x + 0.2, bv, 0.4, label=f"bad ({len(bad)})", color="firebrick")
        ax1.axhline(p_global, ls="--", color="gray", lw=1, label=f"chance {p_global:.2f}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("mean fraction masked")
        ax1.set_title(f"{args.complex} {args.chain1}|{args.chain2}: region masking, good vs bad")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        colors = {"true_epitope": "seagreen", "true_paratope": "royalblue",
                  "decoy_epitope": "firebrick"}
        fb = np.array([f_bad[r] for r in all_res])
        fg = np.array([f_good[r] for r in all_res])
        ax2.scatter(fb, fg, s=10, alpha=0.25, color="lightgray", label="other")
        for rl, col in colors.items():
            rs = [r for r in all_res if role.get(r) == rl]
            if rs:
                ax2.scatter([f_bad[r] for r in rs], [f_good[r] for r in rs],
                            s=40, color=col, edgecolor="k", lw=0.3, label=rl)
        ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax2.set_xlabel("P(masked | bad seed)")
        ax2.set_ylabel("P(masked | good seed)")
        ax2.set_title("per-residue masking frequency")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        out_png = RESULTS_DIR / f"key_residue_masking_{base}_{args.chain1}{args.chain2}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"saved plot -> {out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
