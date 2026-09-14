#!/usr/bin/env python
"""Query the pose index. Use this instead of globbing a prediction directory.

Poses for one complex live in several trees (the base sweep, plus per-arm and CHUNKED deep runs), so
a glob of predictions_template silently returns only seeds 0-150 even when 2500 exist.
Rebuild with scripts/build_pose_index.py after any new inference run.
"""
import collections, csv, gzip, os

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
_IDX = None


def load():
    global _IDX
    if _IDX is None:
        _IDX = []
        with gzip.open(f"{EXP}/artifacts/pose_index.csv.gz", "rt") as fh:
            for r in csv.DictReader(fh):
                _IDX.append((r["pdb"], int(r["seed"]), int(r["sample"]), r["arm"], r["path"]))
    return _IDX


def poses(pdb, arm=None, seed_lo=None, seed_hi=None):
    """-> [(seed, sample, arm, path)] for one complex, optionally restricted."""
    out = []
    for p, s, k, a, f in load():
        if p != pdb.lower():
            continue
        if arm and arm not in a:
            continue
        if seed_lo is not None and s < seed_lo:
            continue
        if seed_hi is not None and s > seed_hi:
            continue
        out.append((s, k, a, f))
    return sorted(out)


def summary(pdb):
    """-> {arm: (n_poses, min_seed, max_seed)}"""
    d = collections.defaultdict(list)
    for s, k, a, f in poses(pdb):
        d[a].append(s)
    return {a: (len(v), min(v), max(v)) for a, v in d.items()}


if __name__ == "__main__":
    import sys
    for pdb in (sys.argv[1:] or ["8aok"]):
        print(f"=== {pdb}")
        tot = set()
        for a, (n, lo, hi) in sorted(summary(pdb).items(), key=lambda kv: -kv[1][0]):
            print(f"   {n:>6} poses  seeds {lo}-{hi}   {a}")
        for s, k, a, f in poses(pdb):
            tot.add(s)
        print(f"   -> {len(tot)} distinct seeds total")
