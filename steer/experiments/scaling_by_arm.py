#!/usr/bin/env python3
"""scaling_by_arm.py — does TEST-TIME SCALING help Protenix more than ESMFold2?

Ebrahim: "Protenix does much better scaling wise than ESMFold2 with MSA, from paper results."

That is a claim about the SLOPE, not the intercept, and it is separable into two very different
curves that this project has been conflating:

  GENERATOR (coverage)  P(the pool contains an acceptable pose) as seeds grow.
                        This can only go UP -- more draws, more chances.
  SELECTOR              P(argmax lands on an acceptable pose | one exists) as seeds grow.
                        This can go DOWN: more seeds means a bigger haystack, and if confidence is
                        miscalibrated at the top, more decoys get a chance to out-score the truth.

The whole confident-decoy story lives in the second curve. On ESMFold2-Fast we already measured the
damning version: 50 -> 1,025 seeds moved coverage 86% -> 94% and moved SELECTION 63% -> 60%. Twenty
times the compute bought three points of top-1, because the selector ate the entire gain.

The question now is whether an MSA fixes the SLOPE, or only the intercept. If ESMFold2+MSA still has
a flat/negative selector curve while Protenix's climbs, then MSAs cure the *level* of the disease
but not its *scaling behaviour* -- and test-time scaling remains a bad bet for ESM.

Bootstrap over seed subsets (100 draws per depth) so each point is an expectation, not one lucky
subsample.
"""
import argparse, collections, csv
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ARMS = [
    ("ESMFold2-Fast, no MSA ", REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"),
    ("base ESMFold2, no MSA ", REPO / "results/csv_files/dockq_ost_esmfold2_nomsa.csv"),
    ("base ESMFold2, +MSA   ", REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"),
    ("Protenix,      +MSA   ", REPO / "results/csv_files/dockq_ost_protenix_full.csv"),
]
SCORE = "pair_iptm"
CX = "7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3".split()


def load(path):
    by = collections.defaultdict(list)
    if not Path(path).exists():
        return by
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq, sc = float(r["dockq"]), float(r[SCORE])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isnan(dq) or np.isnan(sc):
            continue
        by[(r["assembly_id"][:4], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    pools = {n: load(p) for n, p in ARMS}
    keys = sorted(k for k in pools[ARMS[2][0]]
                  if k[0] in CX and all(k in pools[n] for n, _ in ARMS))
    depths = [1, 2, 5, 10, 20]

    print(f"{len(keys)} interfaces present in all four arms\n")
    print("GENERATOR — P(pool contains an acceptable pose).  More seeds can only help.\n")
    print(f"{'arm':<24}" + "".join(f"{f'n={d}':>8}" for d in depths) + f"{'  slope 1->20':>14}")
    print("-" * (24 + 8 * len(depths) + 14))
    gen = {}
    for name, _ in ARMS:
        row = []
        for d in depths:
            vals = []
            for _ in range(a.reps):
                hit = 0
                for k in keys:
                    seeds = sorted({p[0] for p in pools[name][k]})
                    keep = set(rng.choice(seeds, min(d, len(seeds)), replace=False))
                    dq = np.array([p[1] for p in pools[name][k] if p[0] in keep])
                    hit += (dq > ACC).any()
                vals.append(hit / len(keys))
            row.append(np.mean(vals))
        gen[name] = row
        print(f"{name:<24}" + "".join(f"{v:>8.0%}" for v in row) + f"{row[-1]-row[0]:>+14.0%}")

    print("\n\nSELECTOR — P(argmax lands on an acceptable pose | one exists).")
    print("             THIS is where the confident-decoy disease lives. It can go DOWN.\n")
    print(f"{'arm':<24}" + "".join(f"{f'n={d}':>8}" for d in depths) + f"{'  slope 1->20':>14}")
    print("-" * (24 + 8 * len(depths) + 14))
    sel = {}
    for name, _ in ARMS:
        row = []
        for d in depths:
            vals = []
            for _ in range(a.reps):
                hit = tot = 0
                for k in keys:
                    seeds = sorted({p[0] for p in pools[name][k]})
                    keep = set(rng.choice(seeds, min(d, len(seeds)), replace=False))
                    sub = [p for p in pools[name][k] if p[0] in keep]
                    dq = np.array([p[1] for p in sub]); sc = np.array([p[2] for p in sub])
                    if not (dq > ACC).any():
                        continue                       # selector undefined: nothing to find
                    tot += 1
                    hit += dq[int(np.argmax(sc))] > ACC
                if tot:
                    vals.append(hit / tot)
            row.append(np.mean(vals))
        sel[name] = row
        print(f"{name:<24}" + "".join(f"{v:>8.0%}" for v in row) + f"{row[-1]-row[0]:>+14.0%}")

    print("\n\nTOP-1 = generator x selector — what a user actually experiences.\n")
    print(f"{'arm':<24}" + "".join(f"{f'n={d}':>8}" for d in depths) + f"{'  slope 1->20':>14}")
    print("-" * (24 + 8 * len(depths) + 14))
    for name, _ in ARMS:
        row = [g * s for g, s in zip(gen[name], sel[name])]
        print(f"{name:<24}" + "".join(f"{v:>8.0%}" for v in row) + f"{row[-1]-row[0]:>+14.0%}")

    print("\n" + "=" * 76)
    print("READING IT")
    print("=" * 76)
    print("""
  A model SCALES WELL if its SELECTOR curve stays flat or climbs while its GENERATOR curve rises --
  then extra compute converts into extra accuracy. It scales BADLY if the selector curve falls: the
  extra poses it finds are poses it cannot recognise, and top-1 stalls.

  On ESMFold2-Fast at full depth we already measured the bad case: 50 -> 1,025 seeds moved coverage
  86% -> 94% and moved SELECTION 63% -> 60%. Twenty times the compute, three points of top-1.""")


if __name__ == "__main__":
    main()
