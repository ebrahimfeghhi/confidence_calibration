#!/usr/bin/env python
"""WITHIN an interface, does landing on the right epitope go with a better-built antibody?

Across the 250 poses of one interface: correlate epitope Jaccard (did this pose find the right
antigen surface) against paratope RMSD (is this pose's antibody patch built correctly). Then report
the distribution of those per-interface correlations.

WHY WITHIN AND NOT POOLED. A pooled correlation is dominated by between-target differences -- easy
targets have both high epitope overlap and low RMSD -- which says nothing about whether, on ONE
target, the seeds that find the epitope are also the ones with better antibody geometry. Both are
printed so the gap is visible.

SIGN. Lower RMSD is better and higher Jaccard is better, so a NEGATIVE correlation means "poses that
found the epitope also have a better-built antibody". A correlation near zero means the antibody is
built independently of where it docks.

Paratope RMSD is measured by superposing within the antibody on a fixed GT-defined residue set, so it
does not depend on docking success -- that independence is what makes the question well posed.

Usage: .../python scripts/epi_jac_vs_para_rmsd.py [--min-poses 100]
"""
import argparse
import collections
import csv

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SEED_LO, SEED_HI = 101, 150


def load(fn, cols):
    out = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
        if not (SEED_LO <= int(r["seed"]) <= SEED_HI):
            continue
        try:
            out[(r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"])] = \
                tuple(float(r[c]) for c in cols)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-poses", type=int, default=100)
    A = ap.parse_args()
    epi = load("epitope_overlap.csv", ["jaccard"])
    prm = load("paratope_rmsd.csv", ["para_local", "para_global"])

    per = collections.defaultdict(list)
    for k in set(epi) & set(prm):
        per[f"{k[0]}_{k[1]}{k[2]}"].append((epi[k][0], prm[k][0], prm[k][1]))

    rows, pooled = [], []
    for iface, v in sorted(per.items()):
        if len(v) < A.min_poses:
            continue
        a = np.array(v)
        pooled.append(a)
        if a[:, 0].std() < 1e-9:
            continue
        rows.append((iface, len(a),
                     np.corrcoef(a[:, 0], a[:, 1])[0, 1] if a[:, 1].std() > 1e-9 else np.nan,
                     np.corrcoef(a[:, 0], a[:, 2])[0, 1] if a[:, 2].std() > 1e-9 else np.nan))

    P = np.vstack(pooled)
    print(f"{len(rows)} interfaces with >= {A.min_poses} poses "
          f"({len(P)} poses total)\n")
    print("POOLED across all poses (contaminated by between-target differences):")
    for i, nm in ((1, "paratope RMSD local"), (2, "paratope RMSD global")):
        print(f"  epitope jaccard vs {nm:<22} r = {np.corrcoef(P[:,0],P[:,i])[0,1]:+.3f}")

    print("\nWITHIN interface -- distribution of the per-interface correlation r:")
    print(f"{'':<26}{'median':>9}{'mean':>8}{'IQR':>18}{'|r|>0.3':>10}{'n<0':>7}")
    for j, nm in ((2, "vs paratope RMSD local"), (3, "vs paratope RMSD global")):
        x = np.array([r[j] for r in rows], float)
        x = x[~np.isnan(x)]
        q1, q3 = np.percentile(x, [25, 75])
        print(f"  {nm:<24}{np.median(x):>9.3f}{x.mean():>8.3f}"
              f"{q1:>9.2f} to {q3:<6.2f}{100*(np.abs(x)>0.3).mean():>8.0f}%"
              f"{100*(x<0).mean():>6.0f}%")
    print("\n(negative r = poses that found the epitope also have a better-built antibody;"
          "\n r near 0 = antibody geometry is independent of where the pose docked)")


if __name__ == "__main__":
    main()
