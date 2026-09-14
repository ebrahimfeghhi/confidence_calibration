#!/usr/bin/env python3
"""selection_headroom.py — how much accuracy is imperfect SELECTION throwing away, and does it grow?

scaling_by_arm.py (148 interfaces, seeds 1 -> 50):

                            n=1   n=10   n=50
    coverage  ESMFold2-Fast  51%    75%    85%     it FINDS the answer more and more often
    selector  ESMFold2-Fast  86%    67%    63%     and RECOGNISES it less and less often
    TOP-1     ESMFold2-Fast  44%    50%    54%     so top-1 barely moves

This is the project's founding premise, and it is correct: **selection is the bottleneck, and
test-time scaling makes it worse.** Every extra seed adds a pose the model might not recognise.

So: how much is on the table? A PERFECT selector would convert coverage straight into top-1 --

    oracle top-1  =  coverage           (if there is a good pose, you pick it)
    actual top-1  =  coverage x selector
    HEADROOM      =  coverage x (1 - selector)

Headroom is exactly what a label-free selector could win, and its SLOPE says whether the problem is
getting better or worse as we spend more compute.

CONTEXT FOR THE ANSWER. top_decile_structure.py showed a confident decoy and a confident truth are
STRUCTURALLY INDISTINGUISHABLE, so this headroom cannot be captured from the pose ensemble alone.
It has to come from a better confidence signal -- which is what makes the size of it worth knowing.
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
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    pools = {n: load(p) for n, p in ARMS}
    keys = sorted(set(pools[ARMS[0][0]]) & set(pools[ARMS[1][0]]))
    depths = [1, 2, 5, 10, 20, 50]
    print(f"{len(keys)} interfaces, OST-scored, raw pair_iptm\n")

    for name, _ in ARMS:
        print("=" * 78)
        print(f"  {name.strip()}")
        print("=" * 78)
        print(f"{'seeds':>7}{'coverage':>11}{'selector':>10}{'top-1':>8}"
              f"{'ORACLE top-1':>14}{'HEADROOM':>11}")
        print(f"{'':>7}{'(finds it)':>11}{'(picks it)':>10}{'actual':>8}"
              f"{'= coverage':>14}{'left on table':>11}")
        print("-" * 61)
        rows = []
        for d in depths:
            cov, sel = [], []
            for _ in range(a.reps):
                has = hit = tot = 0
                for k in keys:
                    seeds = sorted({p[0] for p in pools[name][k]})
                    keep = set(rng.choice(seeds, min(d, len(seeds)), replace=False))
                    sub = [p for p in pools[name][k] if p[0] in keep]
                    dq = np.array([p[1] for p in sub]); sc = np.array([p[2] for p in sub])
                    if not (dq > ACC).any():
                        continue
                    has += 1; tot += 1
                    hit += dq[int(np.argmax(sc))] > ACC
                cov.append(has / len(keys))
                sel.append(hit / tot if tot else 0.0)
            c, s = np.mean(cov), np.mean(sel)
            rows.append((d, c, s, c * s, c, c * (1 - s)))
            print(f"{d:>7}{c:>11.0%}{s:>10.0%}{c*s:>8.0%}{c:>14.0%}{c*(1-s):>11.0%}")
        h0, h1 = rows[0][5], rows[-1][5]
        print(f"\n  headroom at 1 seed: {h0:.0%}   at {depths[-1]} seeds: {h1:.0%}"
              f"   ({'GROWING' if h1 > h0 else 'shrinking'} {h1-h0:+.0%})")
        print()

    print("=" * 78)
    print("WHAT IT MEANS")
    print("=" * 78)
    print("""
  HEADROOM = coverage x (1 - selector) = the top-1 accuracy a PERFECT label-free selector would add,
  given the poses the model has ALREADY generated. No extra compute; purely better recognition.

  If headroom GROWS with depth, then the more you scale, the more selection costs you -- and the
  case for a label-free selector gets STRONGER, not weaker, as compute grows. That is the project's
  founding premise, and this is the measurement of it.

  The catch, from top_decile_structure.py: a confident decoy and a confident truth are structurally
  INDISTINGUISHABLE, so none of this headroom is reachable from the pose ensemble. It has to come
  from a better confidence signal -- an MSA (worth ~+38% selector on its own), a discriminator
  trained on measured error, or a second model.""")


if __name__ == "__main__":
    main()
