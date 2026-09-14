#!/usr/bin/env python
"""randopt_stats.py — Phase-1 screen readout.

Reads the per-cell DockQ CSVs (results/csv_files/randopt_screen/*.csv), computes each cell's
per-complex best DockQ (max over interfaces x diffusion samples), pairs each perturbation against
the mask0/dropout0 baseline over the 16 complexes, and prints the 3x4 table
(mean DockQ lift, significance, #complexes improved) plus a per-sigma aggregate.

Reading: weigh effect size (mean lift) and #improved over raw p-values (12 cells -> multiple testing).
"""
import csv, os, collections
import numpy as np
from scipy import stats

OUT = "results/csv_files/randopt_screen"
CX = ["8ts0","8ath","7yxu","7zoz","8a44","8w84","7ua2","8dcm",
      "8oxw","7ucx","8vui","8aok","8oni","8q3j","8ivx","8vzo"]
SIGMAS = ["0.0005","0.001","0.002","0.005","0.01"]
SEEDS = [1,2,3,4]
IMP = 0.05   # a complex "improved" if DockQ rose by more than this

def per_complex_best(csvpath):
    """max DockQ over (interface, sample) per 4-char complex code."""
    best = collections.defaultdict(float)
    if not os.path.exists(csvpath):
        return {}
    for r in csv.DictReader(open(csvpath)):
        if r.get("dockq_status") == "chain_unmapped":
            continue
        try:
            dq = float(r["dockq"])
        except (ValueError, KeyError):
            continue
        cx = r["assembly_id"][:4]
        best[cx] = max(best[cx], dq)
    return dict(best)

def paired(pert, base):
    common = [cx for cx in CX if cx in base and cx in pert]
    d = np.array([pert[cx] - base[cx] for cx in common])
    p_arr = np.array([pert[cx] for cx in common])
    b_arr = np.array([base[cx] for cx in common])
    return common, d, p_arr, b_arr

def sig_marks(d, p_arr, b_arr):
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan"), float("nan")
    t_p = stats.ttest_rel(p_arr, b_arr).pvalue
    try:
        w_p = stats.wilcoxon(d).pvalue
    except ValueError:      # all-zero differences
        w_p = 1.0
    return t_p, w_p

def main():
    base = per_complex_best(f"{OUT}/baseline.csv")
    if not base:
        print("No baseline scores yet (results/csv_files/randopt_screen/baseline.csv). "
              "Run randopt_score.sh after the folds finish.")
        return
    print(f"baseline scored for {len(base)}/16 complexes; "
          f"mean best-DockQ {np.mean(list(base.values())):.3f}\n")

    print("3x4 table — mean DockQ lift (perturbed - baseline);  * = t & Wilcoxon p<0.05;  (#improved>%.2f / n)" % IMP)
    print("sigma".ljust(9) + "".join(f"seed {s}".center(20) for s in SEEDS))
    for sig in SIGMAS:
        row = sig.ljust(9)
        for seed in SEEDS:
            pert = per_complex_best(f"{OUT}/s{sig}_seed{seed}.csv")
            if not pert:
                row += "—".center(20); continue
            common, d, p_arr, b_arr = paired(pert, base)
            t_p, w_p = sig_marks(d, p_arr, b_arr)
            mark = "*" if (t_p < 0.05 and w_p < 0.05) else " "
            nimp = int((d > IMP).sum())
            row += f"{d.mean():+.3f}{mark} ({nimp}/{len(d)})".center(20)
        print(row)

    print("\nper-sigma aggregate (4 draws pooled):")
    for sig in SIGMAS:
        allp, allb = [], []
        for seed in SEEDS:
            pert = per_complex_best(f"{OUT}/s{sig}_seed{seed}.csv")
            for cx in CX:
                if cx in base and cx in pert:
                    allp.append(pert[cx]); allb.append(base[cx])
        if len(allp) < 2:
            print(f"  sigma {sig}: (incomplete)"); continue
        d = np.array(allp) - np.array(allb)
        t_p = stats.ttest_rel(allp, allb).pvalue
        print(f"  sigma {sig}: mean lift {d.mean():+.3f}  n={len(d)}  "
              f"t-test p={t_p:.3g}  #improved>{IMP}={int((d>IMP).sum())}  "
              f"#degraded<-{IMP}={int((d<-IMP).sum())}")

if __name__ == "__main__":
    main()
