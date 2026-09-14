#!/usr/bin/env python3
"""top_decile_structure.py — WHAT IS IN each model's top decile?

THE CHAIN OF FINDINGS THIS CLOSES:

  critical_band.py:  ESMFold2's confidence discriminates across the pool (Cliff d +0.404) but
                     INVERTS inside its own top decile (-0.011). Protenix's survives (+0.477).
  dont_take_argmax:  backing off from the argmax HELPS ESM and DESTROYS Protenix
                     (interaction +40%, 95% CI [+10%,+71%], P=0.996).

So ESM's most-confident poses are contaminated. WHY?

THE PROJECT'S STANDING LAW: "the confidence head measures self-consistency, not correctness,
and a rigid wrong-register dock is maximally self-consistent."

If that is what ESM's ipTM is doing, its top decile should COLLAPSE ONTO ONE TIGHT STRUCTURAL
BASIN -- the most self-consistent thing the model can produce, which is often a decoy. Protenix's
head is trained to predict the actual ERROR of its own rollout samples (protenix.py:720-755,
loss.py:1757-1795), so its top decile has no reason to be structurally homogeneous; it should be
DIVERSE, containing whichever poses the head believes are accurate.

PREDICTION
    ESM top decile      : high mutual DockQ, few basins, one dominant basin  (self-consistent)
    Protenix top decile : lower mutual DockQ, more basins                    (error-driven)

DATA. Depth-matched 250-pose pose-vs-pose DockQ matrices for BOTH models, built with the SAME
gemmi (0.7.5, .venv-gpu) and the SAME certified dockq_torch. Using two different parsers for the
two models would be exactly the sort of thing that manufactures a fake result.
    results/dockq_protenix/   (dockq_protenix.py)
    results/dockq_esm250/     (dockq_esm_matched.py, same rng seed 20260713 as every other
                               depth-matched analysis tonight, so the pools are the same poses)
"""
import csv, glob, os
from pathlib import Path

import numpy as np

ACC = 0.23
TAU = 0.35                     # same merge threshold as the clustering work
REPO = Path(__file__).resolve().parents[2]
PRO_DIR = REPO / "results/dockq_protenix"
ESM_DIR = REPO / "results/dockq_esm250"


def load(d, tag):
    npy = d / f"dockq_{tag}.npy"
    idx = d / f"dockq_{tag}_index.csv"
    if not npy.exists() or not idx.exists():
        return None
    D = np.load(npy).astype(np.float64)
    D = (D + D.T) / 2.0                       # DockQ is asymmetric: the reference defines contacts
    rows = list(csv.DictReader(open(idx)))
    ip = np.array([float(r["iptm"]) for r in rows])
    dq = np.array([float(r["dockq_gt"]) for r in rows])
    return D, ip, dq


def leader(D, ip, tau=TAU):
    """Leader clustering, ipTM-descending (Hartigan 1975). Assigns EVERY pose."""
    order = np.argsort(-ip)
    leaders, a = [], np.full(len(ip), -1)
    for i in order:
        if leaders:
            s = D[i, leaders]
            j = int(np.argmax(s))
            if s[j] >= tau:
                a[i] = j
                continue
        a[i] = len(leaders)
        leaders.append(i)
    return a, len(leaders)


def describe(D, ip, dq, frac=0.10):
    """Structural character of the top `frac` of poses by confidence."""
    n = len(ip)
    k = max(3, int(round(frac * n)))
    top = np.argsort(-ip)[:k]
    sub = D[np.ix_(top, top)]
    iu = np.triu_indices(k, 1)
    a, ncl = leader(D[np.ix_(top, top)], ip[top])
    sizes = np.bincount(a)
    return dict(
        n=n, k=k,
        cohesion=float(np.median(sub[iu])),                 # mutual DockQ inside the top decile
        n_basins=int(ncl),
        dominant=float(sizes.max() / k),                    # fraction in the biggest basin
        purity=float((dq[top] > ACC).mean()),
        pool_purity=float((dq > ACC).mean()),
        argmax_ok=bool(dq[int(np.argmax(ip))] > ACC),
    )


def main():
    tags_p = {Path(f).name[6:-4] for f in glob.glob(str(PRO_DIR / "dockq_*.npy"))}
    tags_e = {Path(f).name[6:-4] for f in glob.glob(str(ESM_DIR / "dockq_*.npy"))}
    tags = sorted(tags_p & tags_e)
    print(f"{len(tags)} interfaces with depth-matched 250-pose matrices for BOTH models\n")

    rows = {"Protenix": [], "ESMFold2": []}
    for t in tags:
        p, e = load(PRO_DIR, t), load(ESM_DIR, t)
        if p is None or e is None:
            continue
        for name, (D, ip, dq) in (("Protenix", p), ("ESMFold2", e)):
            if (dq > ACC).sum() == 0:
                continue                                    # NO-POSE: selector undefined
            rows[name].append((t, describe(D, ip, dq)))

    print("STRUCTURAL CHARACTER OF THE TOP 10% BY CONFIDENCE  (all interfaces with a good pose)\n")
    print(f"{'model':<11}{'n':>5}{'cohesion':>10}{'basins':>8}{'dominant':>10}{'top-10% purity':>16}"
          f"{'argmax OK':>11}")
    print("-" * 71)
    for m, rs in rows.items():
        v = [r[1] for r in rs]
        if not v:
            continue
        print(f"{m:<11}{len(v):>5}"
              f"{np.mean([x['cohesion'] for x in v]):>10.3f}"
              f"{np.mean([x['n_basins'] for x in v]):>8.1f}"
              f"{np.mean([x['dominant'] for x in v]):>10.0%}"
              f"{np.mean([x['purity'] for x in v]):>16.0%}"
              f"{np.mean([x['argmax_ok'] for x in v]):>11.0%}")

    # restrict to the selection-critical band, where the whole gap lives
    print("\n\nSELECTION-CRITICAL BAND ONLY (pools 10-50% acceptable — where argmax actually matters)\n")
    print(f"{'model':<11}{'n':>5}{'cohesion':>10}{'basins':>8}{'dominant':>10}{'top-10% purity':>16}"
          f"{'argmax OK':>11}")
    print("-" * 71)
    band = {}
    for m, rs in rows.items():
        v = [x for _, x in rs if 0.10 <= x["pool_purity"] < 0.50]
        band[m] = v
        if not v:
            continue
        print(f"{m:<11}{len(v):>5}"
              f"{np.mean([x['cohesion'] for x in v]):>10.3f}"
              f"{np.mean([x['n_basins'] for x in v]):>8.1f}"
              f"{np.mean([x['dominant'] for x in v]):>10.0%}"
              f"{np.mean([x['purity'] for x in v]):>16.0%}"
              f"{np.mean([x['argmax_ok'] for x in v]):>11.0%}")

    print("\n" + "=" * 71)
    print("THE PREDICTION: ESM's top decile collapses onto ONE tight basin; Protenix's does not.")
    print("=" * 71)
    if band["ESMFold2"] and band["Protenix"]:
        ce = np.mean([x["cohesion"] for x in band["ESMFold2"]])
        cp = np.mean([x["cohesion"] for x in band["Protenix"]])
        be = np.mean([x["n_basins"] for x in band["ESMFold2"]])
        bp = np.mean([x["n_basins"] for x in band["Protenix"]])
        de = np.mean([x["dominant"] for x in band["ESMFold2"]])
        dp = np.mean([x["dominant"] for x in band["Protenix"]])
        print(f"  mutual DockQ inside the top decile : ESM {ce:.3f}  vs  Protenix {cp:.3f}"
              f"   ({'ESM TIGHTER' if ce > cp else 'ESM LOOSER — prediction FAILS'})")
        print(f"  distinct basins in the top decile  : ESM {be:.1f}    vs  Protenix {bp:.1f}"
              f"     ({'ESM FEWER' if be < bp else 'ESM MORE — prediction FAILS'})")
        print(f"  share in the single biggest basin  : ESM {de:.0%}   vs  Protenix {dp:.0%}"
              f"    ({'ESM MORE CONCENTRATED' if de > dp else 'ESM LESS CONCENTRATED — prediction FAILS'})")

    print("\n\nPER-INTERFACE (critical band, ESMFold2, sorted by top-decile cohesion)")
    print(f"{'interface':<12}{'cohesion':>10}{'basins':>8}{'dominant':>10}{'top purity':>12}{'argmax':>8}")
    print("-" * 60)
    for t, x in sorted([r for r in rows["ESMFold2"] if 0.10 <= r[1]["pool_purity"] < 0.50],
                       key=lambda r: -r[1]["cohesion"]):
        print(f"{t:<12}{x['cohesion']:>10.3f}{x['n_basins']:>8}{x['dominant']:>10.0%}"
              f"{x['purity']:>12.0%}{('OK' if x['argmax_ok'] else 'FAIL'):>8}")


if __name__ == "__main__":
    main()
