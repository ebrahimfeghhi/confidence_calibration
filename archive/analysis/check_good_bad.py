"""Confirm 'good' vs 'bad' seed structure for a set of interfaces.

For each (pdb, chain1, chain2) report, from the tts DockQ CSV: baseline DockQ,
per-seed best-sample DockQ distribution, and how many seeds are good/bad. An
interface is interesting (a 'rescue') if the no-mask baseline is low but some
seeds reach high DockQ while most stay low.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CSV = Path("/home/ebrahim/tts_abag/results/csv_files/"
           "dockq_ost_esmfold2fast_tts_all_samples.csv")

INTERFACES = [
    ("7txw", "A", "C"),   # reference (already analyzed)
    ("8ido", "B", "C"),
    ("8vui", "B", "C"),
    ("7yxu", "A", "C"),
    ("8tmz", "B", "C"),
    ("9au1", "C", "E"),
    ("8fdd", "A", "E"),
    # --- corrected chain IDs (what actually exists in the data) ---
    ("8ido", "A", "D"),
    ("8fdd", "A", "C"),
    ("8fdd", "B", "C"),
]

GOOD_T, BAD_T, HIGH_T = 0.5, 0.23, 0.8


def load():
    """{(pdb, frozenset{c1,c2})): {seed: [dockq,...]}, plus baseline list}."""
    rows = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            try:
                dq = float(r["dockq"])
            except (ValueError, KeyError):
                continue
            pdb = r["assembly_id"].split("-")[0]
            key = (pdb, frozenset({r["chain1"], r["chain2"]}))
            rows[key][r["seed"]].append(dq)
    return rows


def main():
    data = load()
    print(f"thresholds: good >= {GOOD_T}, bad < {BAD_T}, high >= {HIGH_T} "
          f"(per-seed = best of its samples)\n")
    hdr = f"{'interface':16s} {'seeds':>5s} {'base':>5s} {'max':>5s} " \
          f"{'#high':>5s} {'#good':>5s} {'#bad':>5s}  good_seeds"
    print(hdr)
    print("-" * len(hdr))
    for pdb, c1, c2 in INTERFACES:
        key = (pdb, frozenset({c1, c2}))
        if key not in data:
            print(f"{pdb} {c1}|{c2:12s} NOT FOUND in CSV")
            continue
        per_seed = data[key]
        baseline = max(per_seed.get("baseline_seed", [float("nan")]))
        best = {s: max(v) for s, v in per_seed.items() if s != "baseline_seed"}
        vals = np.array(list(best.values()))
        n_high = int((vals >= HIGH_T).sum())
        n_good = int((vals >= GOOD_T).sum())
        n_bad = int((vals < BAD_T).sum())
        good_seeds = sorted((int(s) for s, v in best.items() if v >= GOOD_T))
        print(f"{pdb+' '+c1+'|'+c2:16s} {len(best):5d} {baseline:5.2f} "
              f"{vals.max():5.2f} {n_high:5d} {n_good:5d} {n_bad:5d}  {good_seeds}")


if __name__ == "__main__":
    main()
