#!/usr/bin/env python
"""Which metric moves the most when you re-roll a seed? A unitless variance decomposition.

WHY NOT CV. The coefficient of variation (sd/mean) needs a ratio scale with a true zero. fnat, DockQ
and the recall measures are proportions bounded in [0,1] -- already unitless, so dividing by the mean
adds nothing and explodes as the mean approaches zero, which is exactly where confident-decoy poses
live. It also cannot compare a bounded proportion against an unbounded RMSD; an earlier ranking here
(fnat 0.81 vs iRMSD 0.45 vs LRMSD 0.39) did exactly that and should not be trusted.

WHAT THIS DOES INSTEAD. Every interface has 50 seeds x 5 diffusion samples, so each metric's total
variance splits three ways:
    between targets      -- some complexes are just harder; nothing to do with seeds
    between seeds        -- SAME target, different seed: the test-time-scaling knob
    within seed          -- same target, same seed, different diffusion sample
Reported as FRACTIONS of total variance, which are dimensionless and therefore comparable across
metrics whose raw scales are nothing alike. "Between seeds" is the column that answers the question.

Nested and unbalanced-safe: seed means are averaged per interface before pooling, so an interface with
more poses does not dominate. Also reports the quartile coefficient of dispersion (Q3-Q1)/(Q3+Q1) of
the within-target spread -- robust, ratio-scale, meaningful for the unbounded RMSDs.

Usage: /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/metric_variability.py
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iface_chains import resolve  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SEED_LO, SEED_HI = 101, 150
MIN_POSES = 100

# metric -> (csv, column). Chains in all of these are REFERENCE labels (scripts/iface_chains.py).
SOURCES = {
    "epitope recall": ("epitope_overlap.csv", "recall"),
    "paratope recall": ("paratope_overlap.csv", "recall"),
    "epitope local RMSD": ("epitope_rmsd.csv", "epi_local"),
    "paratope local RMSD": ("paratope_rmsd.csv", "para_local"),
}


def key_of(r):
    return (r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"])


def load():
    vals = collections.defaultdict(dict)          # metric -> pose key -> value
    for name, (fn, col) in SOURCES.items():
        for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
            if SEED_LO <= int(r["seed"]) <= SEED_HI:
                vals[name][key_of(r)] = float(r[col])
    # DockQ from OST, for the poses every other metric also has
    common = set.intersection(*(set(v) for v in vals.values()))
    dq = {}
    for k in common:
        try:
            c = resolve(k[0], k[3], k[4], k[1], k[2])
        except (KeyError, FileNotFoundError):
            continue
        if c["dockq"] is not None:
            dq[k] = c["dockq"]
    vals["DockQ"] = dq
    common &= set(dq)
    return {m: {k: v[k] for k in common} for m, v in vals.items()}, common


def decompose(series):
    """series: pose key -> value. Returns variance fractions + robust spread."""
    per_if = collections.defaultdict(lambda: collections.defaultdict(list))
    for (pdb, ab, ag, seed, smp), v in series.items():
        per_if[f"{pdb}_{ab}{ag}"][seed].append(v)
    per_if = {k: v for k, v in per_if.items()
              if sum(len(x) for x in v.values()) >= MIN_POSES}

    if_means, within_seed_v, between_seed_v, if_sd = [], [], [], []
    for iface, seeds in per_if.items():
        seed_means = np.array([np.mean(x) for x in seeds.values()])
        allv = np.concatenate([np.asarray(x, float) for x in seeds.values()])
        if_means.append(allv.mean())
        if_sd.append(allv.std())
        between_seed_v.append(seed_means.var())
        ws = [np.var(x) for x in seeds.values() if len(x) > 1]
        within_seed_v.append(np.mean(ws) if ws else 0.0)

    between_if = float(np.var(if_means))
    between_seed = float(np.mean(between_seed_v))
    within_seed = float(np.mean(within_seed_v))
    tot = between_if + between_seed + within_seed
    q1, q3 = np.percentile(if_sd, [25, 75])
    qcd = (q3 - q1) / (q3 + q1) if (q3 + q1) > 0 else float("nan")
    return dict(n_iface=len(per_if), tot=tot,
                f_target=between_if / tot, f_seed=between_seed / tot,
                f_sample=within_seed / tot,
                mean=float(np.mean(if_means)), sd_within=float(np.mean(if_sd)), qcd=qcd)


def main():
    vals, common = load()
    print(f"{len(common)} poses carrying all {len(SOURCES) + 1} metrics "
          f"(seeds {SEED_LO}-{SEED_HI}); same poses for every row.\n")
    order = ["DockQ", "epitope recall", "paratope recall",
             "epitope local RMSD", "paratope local RMSD"]
    res = {m: decompose(vals[m]) for m in order}

    print("VARIANCE DECOMPOSITION -- fractions of total variance (unitless, comparable across rows)")
    print(f"{'metric':<22}{'n_if':>6}{'target':>9}{'SEED':>8}{'sample':>8}"
          f"{'mean':>9}{'sd(within)':>12}{'QCD':>7}")
    for m in order:
        r = res[m]
        print(f"{m:<22}{r['n_iface']:>6}{r['f_target']:>8.1%}{r['f_seed']:>8.1%}"
              f"{r['f_sample']:>8.1%}{r['mean']:>9.3f}{r['sd_within']:>12.3f}{r['qcd']:>7.2f}")

    print("\n  target = between complexes (not a seed effect)")
    print("  SEED   = same complex, different seed  <-- the test-time-scaling knob")
    print("  sample = same complex and seed, different diffusion sample")

    rank = sorted(order, key=lambda m: -res[m]["f_seed"])
    print(f"\nMoves most on a seed re-roll: {' > '.join(rank)}")

    with open(f"{EXP}/artifacts/metric_variability.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "n_interfaces", "frac_between_target", "frac_between_seed",
                    "frac_within_seed", "mean", "sd_within_target", "qcd_of_within_target_sd"])
        for m in order:
            r = res[m]
            w.writerow([m, r["n_iface"], round(r["f_target"], 4), round(r["f_seed"], 4),
                        round(r["f_sample"], 4), round(r["mean"], 4), round(r["sd_within"], 4),
                        round(r["qcd"], 4)])
    print("wrote artifacts/metric_variability.csv")


if __name__ == "__main__":
    main()
