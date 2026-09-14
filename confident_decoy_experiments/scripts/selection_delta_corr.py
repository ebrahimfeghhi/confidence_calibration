#!/usr/bin/env python
"""Across interfaces: which change explains the change in fnat and DockQ from 1 seed to 50?

Reads artifacts/abag_selection_protocol.csv (see scripts/selection_protocol.py for the selection
protocol) and correlates the per-interface DELTAS. One point per interface, n=148 -- this is a
between-interface question ("when scaling helps a target, what changed about the pose?"), not a
within-interface one.

CIRCULARITY. fnat is a DockQ component, so delta-fnat vs delta-DockQ is partly true by construction
and is shown only as a reference row. Epitope and paratope overlap are NOT DockQ components, so their
correlations with both fnat and DockQ are honest.

Also fits delta-DockQ on all four overlap deltas jointly, to see how much of the improvement is
accounted for by the contact-side changes alone.
"""
import csv
import os

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
PREDS = [("epitope recall", "epi_recall"), ("epitope jaccard", "epi_jac"),
         ("paratope recall", "para_recall"), ("paratope jaccard", "para_jac")]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None, None, None
    r = float(np.corrcoef(x, y)[0, 1])
    # two-sided p from the t statistic; avoids a scipy dependency
    if abs(r) >= 1.0:
        return r, r * r, 0.0
    t = abs(r) * np.sqrt((n - 2) / (1 - r * r))
    try:
        from math import erfc
        p = erfc(t / np.sqrt(2))          # normal approximation, fine at n=148
    except Exception:
        p = float("nan")
    return r, r * r, p


def main():
    rows = list(csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")))
    d = {}
    for _, key in PREDS + [("fnat", "fnat"), ("dockq", "dockq")]:
        d[key] = np.array([float(r[f"sel50_{key}"]) - float(r[f"sel1_{key}"]) for r in rows])
    n = len(rows)
    print(f"per-interface DELTAS (50 seeds - 1 seed), n = {n} interfaces\n")

    for target in ("dockq", "fnat"):
        print(f"--- vs delta {target.upper()}")
        print(f"{'predictor':<20}{'r':>8}{'r^2':>8}{'p':>12}{'n':>6}")
        for name, key in PREDS:
            r, r2, p = pearson(d[key], d[target])
            print(f"{name:<20}{r:>8.3f}{r2:>8.3f}{p:>12.2e}{n:>6}")
        if target == "dockq":
            r, r2, p = pearson(d["fnat"], d["dockq"])
            print(f"{'fnat (CIRCULAR)':<20}{r:>8.3f}{r2:>8.3f}{p:>12.2e}{n:>6}")
        print()

    # joint fit: how much of delta-DockQ do the four contact deltas explain together?
    X = np.column_stack([d[k] for _, k in PREDS] + [np.ones(n)])
    y = d["dockq"]
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res[0] / ss_tot) if len(res) else float("nan")
    print(f"joint fit, delta DockQ ~ all four overlap deltas:  R^2 = {r2:.3f}  (n={n})")
    print("  coefficients: " + ", ".join(f"{nm} {b:+.3f}" for (nm, _), b in zip(PREDS, beta)))

    # epitope-only vs paratope-only, for the comparison we actually care about
    for label, keys in (("epitope only (recall+jaccard)", ["epi_recall", "epi_jac"]),
                        ("paratope only (recall+jaccard)", ["para_recall", "para_jac"])):
        Xs = np.column_stack([d[k] for k in keys] + [np.ones(n)])
        b2, r2s, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        rr = 1 - (r2s[0] / ss_tot) if len(r2s) else float("nan")
        print(f"  {label:<32} R^2 = {rr:.3f}")


if __name__ == "__main__":
    main()
