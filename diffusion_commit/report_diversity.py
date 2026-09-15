#!/usr/bin/env python
"""Does ranking a complex's seeds by LABEL-FREE pose diversity surface the acceptable seeds?

Reads the merged output of pose_diversity.py and reports, per top-k, how often at least one acceptable
seed (a seed containing a pose with DockQ >= 0.23) appears in the first k when seeds are ordered by:

  diversity   1 - mean(off-diagonal pose-pose DockQ)   LABEL-FREE, computable at inference
  pp_min      min(off-diagonal pose-pose DockQ), ascending -- one dissenting pose among four identical
              ones is the case that matters, and a mean dilutes it
  gt_var      variance of DockQ-to-native  ORACLE, the ceiling -- needs the answer, not deployable
  ipTM        mean pair ipTM, descending -- the deployable baseline anyone would try first

AGAINST A HYPERGEOMETRIC BASELINE, per interface. This matters more than it sounds: an interface with 30
acceptable seeds out of 50 gives a random top-3 a 66% hit rate, so a pooled "100%" can be almost entirely
prevalence rather than skill. Results are therefore also split by how many acceptable seeds exist -- the
hard interfaces (1-3 of 50) are the only ones where a selector can demonstrate anything.

Usage: report_diversity.py [--csv a.csv b.csv ...]
"""
import argparse
import collections
import csv
import glob

KS = [3, 5, 10, 20]
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def hit_rates(groups, keyf):
    """-> {k: (hit_rate, random_rate, n_interfaces)} for one ranking."""
    hits = {k: 0 for k in KS}
    rand = {k: 0.0 for k in KS}
    n = 0
    for rs in groups:
        acc = [r for r in rs if r["acceptable"]]
        if not acc or len(rs) < 20:
            continue
        n += 1
        order = sorted(rs, key=keyf)
        N, A = len(rs), len(acc)
        for k in KS:
            hits[k] += any(r["acceptable"] for r in order[:k])
            p = 1.0                                  # P(no acceptable seed in k draws, no replacement)
            for i in range(k):
                p *= max(0, N - A - i) / max(1, N - i)
            rand[k] += 1 - p
    return {k: (hits[k] / n, rand[k] / n, n) for k in KS} if n else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", default=None)
    A = ap.parse_args()
    files = A.csv or sorted(glob.glob(f"{EXP}/artifacts/pose_diversity_cd*.csv"))
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            for k in ("diversity", "pp_min", "gt_max", "gt_var"):
                r[k] = float(r[k])
            r["acceptable"] = int(r["acceptable"])
            rows.append(r)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["iface"]].append(r)
    print(f"  {len(files)} file(s), {len(rows)} seed-rows, {len(by)} interfaces\n")

    RANKINGS = [
        ("diversity  (LABEL-FREE)", lambda r: -r["diversity"]),
        ("pp_min     (LABEL-FREE)", lambda r: r["pp_min"]),
        ("gt_var     (ORACLE)",     lambda r: -r["gt_var"]),
    ]

    def block(title, groups):
        print(f"  === {title} ===")
        if not groups:
            print("    (no interfaces)\n"); return
        print(f"    {'ranking':<26} " + "  ".join(f"top{k:<3}" for k in KS) + "   (random)")
        for name, keyf in RANKINGS:
            hr = hit_rates(groups, keyf)
            if not hr:
                continue
            cells = "  ".join(f"{hr[k][0]*100:>4.0f}%" for k in KS)
            rnd = " ".join(f"{hr[k][1]*100:>3.0f}%" for k in KS)
            print(f"    {name:<26} {cells}   {rnd}   n={hr[KS[0]][2]}")
        print()

    allg = list(by.values())
    block("ALL confident-decoy interfaces", allg)
    # prevalence split -- where a selector can actually show skill
    for lo, hi, lab in [(1, 3, "HARD: 1-3 acceptable seeds of 50"),
                        (4, 9, "MID:  4-9 acceptable"),
                        (10, 99, "EASY: 10+ acceptable")]:
        g = [rs for rs in allg if lo <= sum(r["acceptable"] for r in rs) <= hi]
        block(f"{lab}   ({len(g)} interfaces)", g)


if __name__ == "__main__":
    main()
