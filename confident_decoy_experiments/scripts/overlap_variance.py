#!/usr/bin/env python
"""How much of the DockQ variation is explained by interface overlap (recall / jaccard)?

Runs on either side: --side epitope (antigen surface found) or --side paratope (antibody face used).

Joins artifacts/epitope_overlap.csv to the cached OST DockQ on (pdb, ab, ag, seed, sample) and reports,
per interface, the r^2 of DockQ on each overlap measure -- plus fnat and the paratope/epitope fold
measures for context, so the comparison is like-for-like on the same poses.

WITHIN-INTERFACE, not pooled. A pooled correlation is dominated by between-target differences
(easy targets have both high overlap and high DockQ), which says nothing about whether re-rolling a
seed on ONE target buys you DockQ through finding the surface. Per-interface r^2, then the median
over interfaces, is the quantity that answers the seed-scaling question.

Usage: .../foldbench/bin/python scripts/epitope_overlap_variance.py [--min-poses 100]
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iface_chains import resolve  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["epitope", "paratope"], default="epitope")
    ap.add_argument("--min-poses", type=int, default=100)
    ap.add_argument("--seed-lo", type=int, default=101)
    ap.add_argument("--seed-hi", type=int, default=150)
    A = ap.parse_args()

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    miss = 0
    for r in csv.DictReader(open(f"{EXP}/artifacts/{A.side}_overlap.csv")):
        if not (A.seed_lo <= int(r["seed"]) <= A.seed_hi):
            continue
        # DockQ for THIS pose and THIS interface, from the same OST row the overlap was measured on.
        try:
            ch = resolve(r["pdb"], r["seed"], r["sample"], r["ab_chain"], r["ag_chain"])
        except (KeyError, FileNotFoundError):
            miss += 1
            continue
        dq, fn = ch["dockq"], ch["fnat"]
        if dq is None:
            miss += 1
            continue
        key = f"{r['pdb']}_{r['ab_chain']}{r['ag_chain']}"
        by[key]["recall"].append(float(r["recall"]))
        by[key]["jaccard"].append(float(r["jaccard"]))
        by[key]["dockq"].append(dq)
        by[key]["fnat"].append(fn if fn is not None else np.nan)
    print(f"unjoined overlap rows (no cached DockQ): {miss}", flush=True)

    rows = []
    for key, d in sorted(by.items()):
        if len(d["dockq"]) < A.min_poses:
            continue
        rr, rj = r2(d["recall"], d["dockq"]), r2(d["jaccard"], d["dockq"])
        fn = np.asarray(d["fnat"], float)
        rf = r2(fn, d["dockq"]) if not np.isnan(fn).any() else None
        rows.append(dict(interface=key, n=len(d["dockq"]),
                         mean_dockq=float(np.mean(d["dockq"])),
                         mean_recall=float(np.mean(d["recall"])),
                         sd_recall=float(np.std(d["recall"])),
                         r2_recall=rr, r2_jaccard=rj, r2_fnat=rf))

    with open(f"{EXP}/artifacts/{A.side}_overlap_variance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summ(name, vals):
        v = np.array([x for x in vals if x is not None])
        print(f"  {name:<24} n={len(v):<4} median {np.median(v):.3f}   mean {v.mean():.3f}   "
              f">0.25 in {int((v > .25).sum())}/{len(v)}")

    print(f"\nVARIANCE IN DOCKQ EXPLAINED WITHIN AN INTERFACE, {len(rows)} interfaces "
          f"(seeds {A.seed_lo}-{A.seed_hi}, >={A.min_poses} poses)")
    summ(f"{A.side} recall", [r["r2_recall"] for r in rows])
    summ(f"{A.side} jaccard", [r["r2_jaccard"] for r in rows])
    summ("fnat (DockQ component)", [r["r2_fnat"] for r in rows])
    print(f"\nwrote artifacts/{A.side}_overlap_variance.csv")


if __name__ == "__main__":
    main()
