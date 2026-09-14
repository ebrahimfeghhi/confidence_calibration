#!/usr/bin/env python3
"""residual_gap.py — with the MSA controlled, WHAT is left between ESMFold2 and Protenix?

The MSA ablation (msa_ablation_result.py) showed:

    base ESMFold2, no MSA   selector 25%
    base ESMFold2, +MSA     selector 62%     <- the MSA is worth +38% (P=0.998)
    Protenix,      +MSA     selector 86%     <- and +18-24% STILL separates them

Ebrahim notes that published results also have Protenix clearly ahead of ESMFold2-with-MSA, so that
residual is almost certainly real rather than our small-n noise. Which makes it the interesting
quantity: **with the MSA controlled, is Protenix's advantage its CONFIDENCE HEAD, or its POOL?**

Two things drive argmax accuracy:
  * DISCRIMINATION -- how well confidence orders good poses above bad ones (the head).
  * PREVALENCE     -- how many good poses there are to find at all (the generator/pool purity).
A model with an identical head will still win more often if its pool is purer, because there is
simply more truth to land on.

  T1  STRATIFY.       Compare selector accuracy within bins of pool purity. If the gap vanishes
                      inside a bin, purity explains it.
  T2  COUNTERFACTUAL. Resample each pool to the OTHER model's purity, keeping each model's own
                      confidence values attached to its own poses -- so the head's ability is
                      untouched and only prevalence changes. Then re-run argmax.
  T3  TOP-END.        Cliff's delta among the most-confident poses, which is the region argmax
                      actually reads and where the confident-decoy disease lives.

This is the same machinery as purity_vs_head.py, but between the two MSA-HAVING arms -- so the MSA
is no longer a confound, and whatever is left is a genuine model difference.
"""
import argparse, collections, csv
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
B_CSV = REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"        # base ESMFold2 + MSA
C_CSV = REPO / "results/csv_files/dockq_ost_protenix_full.csv"       # Protenix + MSA
SCORE = "pair_iptm"
CX = "7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3".split()


def load(path):
    by = collections.defaultdict(list)
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
    g, b = sc[good], sc[~good]
    if len(g) == 0 or len(b) == 0:
        return None
    return float(((g[:, None] > b[None, :]).sum() - (g[:, None] < b[None, :]).sum())
                 / (len(g) * len(b)))


def stats(poses, frac=0.25):
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    good = dq > ACC
    if not good.any():
        return None
    order = np.argsort(-sc)
    k = max(3, int(round(frac * len(dq))))
    top = order[:k]
    return dict(dq=dq, sc=sc, n=len(dq), base=float(good.mean()),
                hit=bool(dq[int(np.argmax(sc))] > ACC),
                cliff_all=cliffs(sc, good), cliff_top=cliffs(sc[top], good[top]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    B, C = load(B_CSV), load(C_CSV)
    keys = sorted(k for k in B if k[0] in CX and k in C)

    rows = []
    for k in keys:
        b = stats([p for p in B[k] if int(p[0]) < a.nseeds])
        cs = sorted({p[0] for p in C[k]})
        keep = set(rng.choice(cs, min(a.nseeds, len(cs)), replace=False))   # depth-match
        c = stats([p for p in C[k] if p[0] in keep])
        if b and c:
            rows.append((k, b, c))
    print(f"{len(rows)} interfaces where BOTH base-ESMFold2+MSA and Protenix have an acceptable pose")
    print(f"  (depth-matched to {a.nseeds} seeds x 5 samples = {a.nseeds*5} poses; raw pair_iptm)\n")

    bb = np.array([r[1]["base"] for r in rows]); bh = np.array([r[1]["hit"] for r in rows])
    cb = np.array([r[2]["base"] for r in rows]); ch = np.array([r[2]["hit"] for r in rows])
    print(f"{'':<20}{'selector':>10}{'pool purity':>13}{'Cliff d ALL':>13}{'Cliff d TOP':>13}")
    print("-" * 69)
    for nm, hit, base, key in (("base ESMFold2+MSA", bh, bb, 1), ("Protenix+MSA", ch, cb, 2)):
        ca = np.nanmean([r[key]["cliff_all"] for r in rows if r[key]["cliff_all"] is not None])
        ct = np.nanmean([r[key]["cliff_top"] for r in rows if r[key]["cliff_top"] is not None])
        print(f"{nm:<20}{hit.mean():>10.0%}{base.mean():>13.0%}{ca:>+13.3f}{ct:>+13.3f}")

    print("\n" + "=" * 69)
    print("T1  SELECTOR WITHIN BINS OF POOL PURITY")
    print("=" * 69)
    print(f"{'pool purity':<16}{'ESM+MSA':>10}{'(n)':>5}{'Protenix':>11}{'(n)':>5}{'gap':>8}")
    print("-" * 55)
    for lo, hi in ((0.0, 0.25), (0.25, 0.50), (0.50, 1.01)):
        m1, m2 = (bb >= lo) & (bb < hi), (cb >= lo) & (cb < hi)
        if m1.sum() < 2 or m2.sum() < 2:
            continue
        print(f"{f'{lo:.0%}-{hi:.0%}':<16}{bh[m1].mean():>10.0%}{m1.sum():>5}"
              f"{ch[m2].mean():>11.0%}{m2.sum():>5}{ch[m2].mean()-bh[m1].mean():>+8.0%}")

    print("\n" + "=" * 69)
    print("T2  COUNTERFACTUAL — swap the POOL PURITY, keep each model's own HEAD")
    print("=" * 69)

    def resample_to(s, target, reps=400):
        dq, sc, n = s["dq"], s["sc"], s["n"]
        gi = np.where(dq > ACC)[0]; bi = np.where(dq <= ACC)[0]
        if len(gi) == 0 or len(bi) == 0:
            return None
        ng = max(1, int(round(target * n))); nb = max(1, n - ng)
        hits = 0
        for _ in range(reps):
            idx = np.concatenate([rng.choice(gi, ng, replace=True), rng.choice(bi, nb, replace=True)])
            hits += dq[idx[int(np.argmax(sc[idx]))]] > ACC
        return hits / reps

    b_at_c = [x for x in (resample_to(r[1], cb.mean()) for r in rows) if x is not None]
    c_at_b = [x for x in (resample_to(r[2], bb.mean()) for r in rows) if x is not None]
    print(f"  ESM+MSA head, at its own purity  ({bb.mean():.0%}) : {bh.mean():>4.0%}   [observed]")
    print(f"  ESM+MSA head, at Protenix purity ({cb.mean():.0%}) : {np.mean(b_at_c):>4.0%}")
    print(f"  Protenix head, at its own purity ({cb.mean():.0%}) : {ch.mean():>4.0%}   [observed]")
    print(f"  Protenix head, at ESM+MSA purity ({bb.mean():.0%}) : {np.mean(c_at_b):>4.0%}")
    obs = ch.mean() - bh.mean()
    fair = np.mean(c_at_b) - bh.mean()            # both heads at ESM's purity
    print(f"\n  observed gap                                : {obs:+.0%}")
    print(f"  gap with purity EQUALISED (both at {bb.mean():.0%})     : {fair:+.0%}")
    if abs(obs) > 1e-9:
        print(f"  -> {100*(1-fair/obs):.0f}% of the residual gap is POOL PURITY;"
              f" {100*fair/obs:.0f}% is the CONFIDENCE HEAD")

    print("\n" + "=" * 69)
    print("T3  THE TOP END — where argmax actually reads")
    print("=" * 69)
    bt = [r[1]["cliff_top"] for r in rows if r[1]["cliff_top"] is not None]
    ct = [r[2]["cliff_top"] for r in rows if r[2]["cliff_top"] is not None]
    d = np.nanmean(ct) - np.nanmean(bt)
    bs = np.array([np.nanmean(rng.choice(ct, len(ct))) - np.nanmean(rng.choice(bt, len(bt)))
                   for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  Cliff's delta in the top quartile:  ESM+MSA {np.nanmean(bt):+.3f}   "
          f"Protenix {np.nanmean(ct):+.3f}")
    print(f"  difference {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"P(Protenix's head is better at the top) = {(bs > 0).mean():.3f}")


if __name__ == "__main__":
    main()
