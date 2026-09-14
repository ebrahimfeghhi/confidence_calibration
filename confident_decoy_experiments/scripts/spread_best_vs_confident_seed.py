#!/usr/bin/env python
"""On the selection failures: is the seed holding the BEST pose more scattered than the MOST CONFIDENT
seed?

MOTIVATION, from two hand-examined cases. On 9x1h all five seeds that produce an acceptable pose are
scattered (within-seed epitope-Jaccard SD 0.13-0.24) and sit at ipTM ~0.13 against a 0.39 mean, while
the top-confidence seed (0.700) has SD 0.000 -- five identical poses, all on the decoy at DockQ 0.02.
On 8w5f the same holds for three of four good seeds, but the fourth (300047) COMMITTED to the correct
epitope, scored an above-average 0.433, and still lost to a decoy-committed seed at 0.610. So the
pattern is suggestive but n=2 interfaces. This runs it over every failure.

DEFINITION OF A FAILURE (the confident-decoy set): oracle DockQ >= 0.23 but the ipTM-selected pose
< 0.23 -- an acceptable pose was generated and not chosen. Coverage failures (no acceptable pose at
all) are excluded; nothing about selection can be learned from them.

TWO SEEDS PER INTERFACE, compared PAIRWISE so each interface is its own control:
  best seed       the seed containing the highest-DockQ pose
  confident seed  the seed with the highest MEAN pair_iptm over its 5 samples
SPREAD is the standard deviation of epitope Jaccard across a seed's 5 samples -- the same measure that
gave r = -0.81 against ipTM on 9x1h. Epitope rather than DockQ because it is the quantity that
actually differs between basins.

Interfaces where the two seeds coincide are reported separately, not silently dropped: those are
cases where confidence DID pick the right seed and the failure is within-seed sample choice.

EXACT POSE SET enforced (50 seeds x 5 samples); anything short is excluded and counted.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/spread_best_vs_confident_seed.py
"""
import collections
import csv

import numpy as np
from scipy.stats import wilcoxon

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
T = 0.23


def main():
    epi = {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        epi[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))),
             int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])

    poses = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not (r["dockq"] and r["pair_iptm"]):
            continue
        k = (r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))), int(r["seed"]), int(r["sample"]))
        if k in epi:
            poses[f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'][int(r["seed"])].append(
                (float(r["pair_iptm"]), float(r["dockq"]), epi[k]))

    rows, same, skipped = [], 0, 0
    for iface, byseed in poses.items():
        byseed = {s: v for s, v in byseed.items() if len(v) == 5}
        if len(byseed) < 40:
            skipped += 1
            continue
        allp = [p for v in byseed.values() for p in v]
        oracle = max(p[1] for p in allp)
        sel = max(allp, key=lambda p: p[0])[1]
        if not (oracle >= T and sel < T):          # keep only selection failures
            continue
        best_seed = max(byseed, key=lambda s: max(p[1] for p in byseed[s]))
        conf_seed = max(byseed, key=lambda s: np.mean([p[0] for p in byseed[s]]))
        spread = lambda s: float(np.std([p[2] for p in byseed[s]]))
        meanip = lambda s: float(np.mean([p[0] for p in byseed[s]]))
        if best_seed == conf_seed:
            same += 1
            continue
        rows.append(dict(iface=iface, oracle=oracle, sel=sel,
                         best_spread=spread(best_seed), conf_spread=spread(conf_seed),
                         best_ip=meanip(best_seed), conf_ip=meanip(conf_seed),
                         best_dq=max(p[1] for p in byseed[best_seed])))

    print(f"{len(rows)} selection failures where the best seed and the most-confident seed DIFFER")
    print(f"({same} failures where they are the SAME seed -- confidence picked the right seed and "
          f"lost on sample choice; {skipped} interfaces excluded for incomplete pose sets)\n")
    if not rows:
        return

    bs = np.array([r["best_spread"] for r in rows])
    cs = np.array([r["conf_spread"] for r in rows])
    bi = np.array([r["best_ip"] for r in rows])
    ci = np.array([r["conf_ip"] for r in rows])
    print(f"{'':<26}{'best seed':>12}{'confident seed':>17}")
    print(f"{'median epitope spread':<26}{np.median(bs):>12.3f}{np.median(cs):>17.3f}")
    print(f"{'mean epitope spread':<26}{bs.mean():>12.3f}{cs.mean():>17.3f}")
    print(f"{'seeds with ZERO spread':<26}{int((bs<0.01).sum()):>9}/{len(bs)}"
          f"{int((cs<0.01).sum()):>14}/{len(cs)}")
    print(f"{'median mean ipTM':<26}{np.median(bi):>12.3f}{np.median(ci):>17.3f}")
    st, p = wilcoxon(bs, cs)
    print(f"\nbest seed is MORE scattered than the confident seed in "
          f"{int((bs>cs).sum())}/{len(bs)} interfaces   (paired Wilcoxon p = {p:.2e})")
    print(f"median paired difference in spread: {np.median(bs-cs):+.3f}")
    print(f"median ipTM penalty paid by the best seed: {np.median(bi-ci):+.3f}")

    rows.sort(key=lambda r: -(r["best_spread"] - r["conf_spread"]))
    print(f"\n{'interface':<13}{'best DockQ':>11}{'shipped':>9}"
          f"{'spread best':>13}{'spread conf':>13}{'ipTM best':>11}{'ipTM conf':>11}")
    for r in rows[:12]:
        print(f'{r["iface"]:<13}{r["best_dq"]:>11.3f}{r["sel"]:>9.3f}'
              f'{r["best_spread"]:>13.3f}{r["conf_spread"]:>13.3f}'
              f'{r["best_ip"]:>11.3f}{r["conf_ip"]:>11.3f}')


if __name__ == "__main__":
    main()
