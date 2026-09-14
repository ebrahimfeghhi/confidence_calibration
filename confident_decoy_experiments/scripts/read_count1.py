#!/usr/bin/env python
"""Paired comparison: 9x1h predicted with ONE copy vs the sweep's TWO copies, same 30 seeds.

The deposited assembly has two copies of the complex (a crystal-packing artifact). Protenix builds
both and they come out near-identical -- 0.50 A between protomers -- so count=2 costs ~4x the pair
compute for a duplicate. The question is whether dropping it changes the ANSWER.

WHAT WOULD MAKE count=1 SAFE TO ADOPT: the DockQ distribution unchanged, and specifically the
confident-decoy structure preserved -- the decoy basin near 0.019 and the rare good basin above 0.23.
If the decoy vanishes at count=1 that is a finding, not a convenience: it would mean the second copy
is part of why the model goes wrong here.

Chain mapping comes from each pose's own PXMeter ref_to_model_chain_mapping. It is NOT assumed to
match the count=2 runs -- a one-copy model has different chain letters, and PXMeter maps the native
onto whichever model chains fit.

Usage: read_count1.py
"""
import collections
import csv
import glob
import json
import os
import re

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
C = f"{EXP}/artifacts/count1"
ACC = 0.23
SEEDS = set(range(300000, 300030))


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    # count=2 arm: the original sweep, same seeds, C-D interface
    two = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] == "9x1h" and r["dockq"] and int(r["seed"]) in SEEDS
                and "".join(sorted((r["chain1"], r["chain2"]))) == "CD"):
            two[int(r["seed"])].append(
                (float(r["dockq"]), float(r["pair_iptm"]) if r["pair_iptm"] else np.nan))

    # count=1 arm
    one = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{C}/pxm_json/*.json")):
        m = re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            J = json.load(open(f))
        except Exception:
            continue
        v = (J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
        if not v or v.get("dockq") is None:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        cf = glob.glob(f"{C}/out/**/seed_{m.group(1)}/predictions/"
                       f"*summary_confidence_sample_{m.group(2)}.json", recursive=True)
        ip = np.nan
        if cf and all(c in r2m for c in ("C", "D")):
            M = json.load(open(cf[0])).get("chain_pair_iptm")
            i, j = chain_idx(r2m["C"]), chain_idx(r2m["D"])
            if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                ip = float(M[i][j])
        one[int(m.group(1))].append((float(v["dockq"]), ip))

    if not one:
        print("no count=1 scores yet")
        return

    def flat(d):
        return (np.array([x[0] for v in d.values() for x in v]),
                np.array([x[1] for v in d.values() for x in v]))
    d2, p2 = flat(two)
    d1, p1 = flat(one)
    print(f"\n{'arm':<12} {'n':>5} {'seeds':>6} {'DockQ mean':>11} {'median':>8} {'max':>7} "
          f"{'acceptable':>11} {'ipTM mean':>10}")
    for lab, d, p, dd in (("count=2", d2, p2, two), ("count=1", d1, p1, one)):
        print(f"{lab:<12} {len(d):>5} {len(dd):>6} {d.mean():>11.3f} {np.median(d):>8.3f} "
              f"{d.max():>7.3f} {str(int((d>=ACC).sum()))+'/'+str(len(d)):>11} {np.nanmean(p):>10.3f}")

    # the decoy basin: is the ~0.019 attractor still there?
    for lab, d in (("count=2", d2), ("count=1", d1)):
        near = ((d > 0.010) & (d < 0.030)).mean()
        print(f"   {lab}: {100*near:.0f}% of poses sit in the 0.01-0.03 decoy band")

    from scipy import stats
    u = stats.mannwhitneyu(d2, d1, alternative="two-sided")
    print(f"\nMann-Whitney on the DockQ distributions: p={u.pvalue:.3f}")
    o, pf = stats.fisher_exact([[int((d2 >= ACC).sum()), len(d2)],
                                [int((d1 >= ACC).sum()), len(d1)]])
    print(f"acceptable-rate Fisher exact: p={pf:.3f}")
    print("\nper-seed max DockQ, paired (only seeds present in both):")
    both = sorted(set(two) & set(one))
    print(f"{'seed':>8} {'count=2':>9} {'count=1':>9}")
    for s in both[:12]:
        print(f"{s:>8} {max(x[0] for x in two[s]):>9.3f} {max(x[0] for x in one[s]):>9.3f}")
    if len(both) > 12:
        print(f"   ... {len(both)-12} more")


if __name__ == "__main__":
    main()
