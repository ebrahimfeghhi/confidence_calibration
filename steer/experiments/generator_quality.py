#!/usr/bin/env python3
"""generator_quality.py — is Protenix's TRUNK better, or only its confidence head?

Ebrahim asked: "is Protenix just trained on more / better data?"

I first answered that the generator is "nearly tied" because COVERAGE (does the pool contain any
acceptable pose?) is 85% vs 89%. That was a sloppy reading. Coverage is a saturating, all-or-nothing
statistic -- at 50 seeds almost any model eventually stumbles on a good pose. It says very little
about how GOOD the model is at building structures.

The measures that actually probe the trunk:

    PURITY        fraction of poses that are acceptable. A better generator wastes fewer samples.
    BEST DockQ    the single best pose in the pool -- the ceiling the model can reach at all.
    MEDIAN DockQ  the typical pose. This is the least forgiving generator statistic.
    HIGH-QUALITY  fraction of poses above DockQ 0.80 (CAPRI "high"), not just past the 0.23 line.

If Protenix's trunk is genuinely better (more/better training data), these should ALL favour it.
If instead they are close and only the SELECTOR differs, the advantage is in the confidence head.

Depth-matched, OST-scored, raw pair_iptm.
"""
import argparse, collections, csv
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ARMS = [
    ("ESMFold2-Fast, no MSA", REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"),
    ("Protenix,      +MSA  ", REPO / "results/csv_files/dockq_ost_protenix_full.csv"),
]
SCORE = "pair_iptm"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=50)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    pools = {n: load(p) for n, p in ARMS}
    keys = sorted(set(pools[ARMS[0][0]]) & set(pools[ARMS[1][0]]))
    print(f"{len(keys)} interfaces, depth-matched to {a.nseeds} seeds x 5 samples\n")

    stat = {}
    for name, _ in ARMS:
        purity, best, med, high, cov, sel = [], [], [], [], [], []
        for k in keys:
            seeds = sorted({p[0] for p in pools[name][k]})
            keep = set(rng.choice(seeds, min(a.nseeds, len(seeds)), replace=False))
            sub = [p for p in pools[name][k] if p[0] in keep]
            dq = np.array([p[1] for p in sub]); sc = np.array([p[2] for p in sub])
            purity.append((dq > ACC).mean())
            best.append(dq.max())
            med.append(np.median(dq))
            high.append((dq > 0.80).mean())
            cov.append(bool((dq > ACC).any()))
            if (dq > ACC).any():
                sel.append(bool(dq[int(np.argmax(sc))] > ACC))
        stat[name] = dict(purity=np.mean(purity), best=np.mean(best), med=np.mean(med),
                          high=np.mean(high), cov=np.mean(cov), sel=np.mean(sel))

    print("GENERATOR (the trunk) — how good are the POSES, regardless of which one gets picked?\n")
    print(f"{'':<24}{'coverage':>10}{'purity':>9}{'best DockQ':>12}{'median DockQ':>14}{'% > 0.80':>10}")
    print("-" * 79)
    for name, _ in ARMS:
        s = stat[name]
        print(f"{name:<24}{s['cov']:>10.0%}{s['purity']:>9.0%}{s['best']:>12.3f}"
              f"{s['med']:>14.3f}{s['high']:>10.0%}")

    print("\nSELECTOR (the confidence head) — given a good pose exists, is it picked?\n")
    print(f"{'':<24}{'selector':>10}")
    print("-" * 34)
    for name, _ in ARMS:
        print(f"{name:<24}{stat[name]['sel']:>10.0%}")

    e, p = stat[ARMS[0][0]], stat[ARMS[1][0]]
    print("\n" + "=" * 79)
    print("VERDICT ON 'IS IT JUST BETTER DATA?'")
    print("=" * 79)
    print(f"""
  generator gaps (Protenix − ESMFold2-Fast):
      coverage      {p['cov']-e['cov']:+.0%}      saturating, says little
      purity        {p['purity']-e['purity']:+.0%}      Protenix wastes fewer samples
      best DockQ    {p['best']-e['best']:+.3f}     the ceiling each model can reach
      median DockQ  {p['med']-e['med']:+.3f}     the TYPICAL pose -- the least forgiving statistic
      % > 0.80      {p['high']-e['high']:+.0%}      high-accuracy poses

  selector gap    {p['sel']-e['sel']:+.0%}

  A "Protenix just has more/better data" story predicts the TRUNK statistics move together with the
  selector. Read the two blocks above and see whether they do. If the generator gaps are small while
  the selector gap is large, the advantage is concentrated in the confidence head -- which is the one
  component we have CONFIRMED is trained differently (on the measured error of its own rollout
  samples: protenix.py:720-755, loss.py:1757-1795).""")


if __name__ == "__main__":
    main()
