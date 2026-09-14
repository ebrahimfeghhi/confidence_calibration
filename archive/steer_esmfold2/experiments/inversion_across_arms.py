#!/usr/bin/env python3
"""inversion_across_arms.py — is the confident-decoy INVERSION a property of ESMFold2-Fast, or of
ESM-class models in general?

critical_band.py found the mechanism behind everything this project has been chasing:

    ESMFold2-Fast's confidence discriminates across the pool (Cliff's d +0.404)
    but INVERTS inside its own top decile (-0.011). Protenix's survives there (+0.477).

argmax reads exactly the top. So the confidence works everywhere except where we look.

But the MSA ablation (msa_ablation_result.py) then showed the ENTIRE ESMFold2-Fast -> Protenix
selector gap decomposes into the CHECKPOINT (+17%) and the MSA (+13%), with nothing left for the
confidence head. We ran this whole project on ESMFold2-**Fast**, a distilled model, single-sequence.

So the question that decides how much of this project generalises:

    DOES THE INVERSION SURVIVE ON THE BASE CHECKPOINT? WITH AN MSA?

  It persists in all arms  -> the confident-decoy disease is a property of ESM-class confidence,
                              and the project's central phenomenon is real and general.
  It vanishes with the base checkpoint / MSA -> the phenomenon is an artefact of a handicapped
                              configuration, and essentially everything we concluded is about
                              ESMFold2-Fast specifically, not about structure predictors.

Four arms, identical interfaces, identical seeds, all OST-scored, all ranked by raw pair_iptm.

CAVEAT, stated up front: these pools are 8 seeds x 5 samples = 40 poses, so a "top decile" is 4
poses and Cliff's delta on it is noisy. We therefore report the top QUARTILE (10 poses) as the
primary statistic and the decile alongside it. Protenix and ESMFold2-Fast are subsampled to the
same 8 seeds so every arm is measured on the same footing -- which also means the ESMFold2-Fast
number here is NOT directly comparable to critical_band.py's (that used 250-pose pools).
"""
import collections, csv
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ARMS = [
    ("A   ESMFold2-Fast, no MSA", REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"),
    ("A'  base ESMFold2, no MSA", REPO / "results/csv_files/dockq_ost_esmfold2_nomsa.csv"),
    ("B   base ESMFold2, +MSA  ", REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"),
    ("C   Protenix,      +MSA  ", REPO / "results/csv_files/dockq_ost_protenix_full.csv"),
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


def cliffs(sc, good):
    """+1 = every acceptable pose outscores every bad one. -1 = perfectly backwards. 0 = nothing."""
    g, b = sc[good], sc[~good]
    if len(g) == 0 or len(b) == 0:
        return None
    return float(((g[:, None] > b[None, :]).sum() - (g[:, None] < b[None, :]).sum())
                 / (len(g) * len(b)))


def analyse(poses, frac):
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    good = dq > ACC
    if not good.any():
        return None
    order = np.argsort(-sc)
    k = max(3, int(round(frac * len(dq))))
    top = order[:k]
    return dict(all=cliffs(sc, good), top=cliffs(sc[top], good[top]),
                hit=bool(dq[int(np.argmax(sc))] > ACC), n=len(dq))


def main():
    rng = np.random.default_rng(20260713)
    pools = {name: load(p) for name, p in ARMS}
    abl = [n for n, _ in ARMS if "base ESMFold2" in n]
    keys = sorted(k for k in pools[ARMS[1][0]]
                  if k[0] in CX and all(k in pools[n] for n, _ in ARMS))
    print(f"{len(keys)} interfaces with all four arms "
          f"(the 9 complexes base ESMFold2 can fold)\n")

    for frac, label in ((0.25, "TOP QUARTILE (25 of 100 poses — the primary statistic)"),
                        (0.10, "TOP DECILE (10 of 100 poses)")):
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"{'arm':<28}{'n':>4}{'selector':>10}{'Cliff d ALL':>13}{'Cliff d TOP':>13}"
              f"{'inverted':>10}")
        print("-" * 78)
        for name, _ in ARMS:
            vals = []
            for k in keys:
                # every arm on the SAME 8 seeds. The ablation arms ran seeds 0-7; subsample the
                # two long-running pools to 8 seeds so nothing is compared at unequal depth.
                seeds = sorted({p[0] for p in pools[name][k]})
                if "base ESMFold2" in name:
                    keep = {str(s) for s in range(20)}   # 20 seeds complete in BOTH arms
                else:
                    keep = set(rng.choice(seeds, min(20, len(seeds)), replace=False))
                v = analyse([p for p in pools[name][k] if p[0] in keep], frac)
                if v:
                    vals.append(v)
            if not vals:
                continue
            ca = [v["all"] for v in vals if v["all"] is not None]
            ct = [v["top"] for v in vals if v["top"] is not None]
            inv = np.mean([c < 0 for c in ct]) if ct else float("nan")
            print(f"{name:<28}{len(vals):>4}{np.mean([v['hit'] for v in vals]):>10.0%}"
                  f"{np.mean(ca):>+13.3f}{np.mean(ct):>+13.3f}{inv:>10.0%}")
        print()

    print("=" * 78)
    print("READING IT")
    print("=" * 78)
    print("""
  'Cliff d ALL' is how well confidence separates acceptable from bad poses across the WHOLE pool.
  'Cliff d TOP' is the same thing computed only among the most-confident poses -- the region argmax
  actually reads. A model whose confidence is healthy keeps its discrimination at the top. The
  confident-decoy disease is: ALL is positive, TOP collapses or inverts.

  If TOP stays positive once the base checkpoint has an MSA, then the disease is a property of the
  handicapped configuration we studied, and this project's central phenomenon does not generalise
  as stated. If TOP still collapses, the phenomenon is real for ESM-class confidence and only the
  EXPLANATION (rollout-trained heads) was wrong.""")


if __name__ == "__main__":
    main()
