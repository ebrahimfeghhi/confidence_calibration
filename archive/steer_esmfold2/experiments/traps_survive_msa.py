#!/usr/bin/env python3
"""traps_survive_msa.py — how much of the CONFIDENT-DECOY TRAP CLASS survives an MSA?

This project is built on a single class of failure:

    TRAP = the model's own confidence picks a pose below DockQ 0.23, even though an ACCEPTABLE
           pose exists somewhere in its pool. The answer is there; the model cannot see it.

Everything -- the clustering, the tau sweeps, the 6/22 -- exists to rescue that class.

inversion_across_arms.py then showed that the mechanism behind it (confidence INVERTING inside its
own top decile) is present in BOTH checkpoints WITHOUT an MSA and absent in BOTH with one. So the
question that decides whether this project has a subject at all:

    HOW MANY TRAPS ARE LEFT ONCE THE MODEL HAS AN MSA?

  Most of them evaporate  -> the trap class was largely an artefact of running single-sequence, and
                             the method work was aimed at a phantom.
  Most of them survive    -> the phenomenon is real; only our EXPLANATION of it was wrong, and the
                             rescue problem still stands.

Classification, per interface, per arm (all on identical seeds, OST-scored, raw pair_iptm):

    WIN     argmax lands on a pose with DockQ > 0.23
    TRAP    it does not, but such a pose EXISTS in the pool     <- the class this project targets
    NO-POSE no acceptable pose anywhere; a coverage failure, not a selection one

A trap converting to WIN under MSA means the MSA fixed the SELECTION (the pose was already there).
A trap converting to NO-POSE would mean the MSA made the POOL worse -- worth knowing separately.
"""
import argparse, collections, csv
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


def classify(poses):
    if not poses:
        return None
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    if not (dq > ACC).any():
        return "NO-POSE"
    return "WIN" if dq[int(np.argmax(sc))] > ACC else "TRAP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=14)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    pools = {n: load(p) for n, p in ARMS}
    keys = sorted(k for k in pools[ARMS[1][0]]
                  if k[0] in CX and all(k in pools[n] for n, _ in ARMS))

    cls = {}
    for name, _ in ARMS:
        cls[name] = {}
        for k in keys:
            seeds = sorted({p[0] for p in pools[name][k]})
            if "base ESMFold2" in name:
                keep = {str(s) for s in range(a.nseeds)}
            else:
                keep = set(rng.choice(seeds, min(a.nseeds, len(seeds)), replace=False))
            cls[name][k] = classify([p for p in pools[name][k] if p[0] in keep])

    print(f"{len(keys)} interfaces, {a.nseeds} seeds x 5 samples, identical across arms\n")
    print(f"{'arm':<28}{'WIN':>6}{'TRAP':>6}{'NO-POSE':>9}   {'trap rate':>10}")
    print("-" * 62)
    for name, _ in ARMS:
        c = collections.Counter(v for v in cls[name].values() if v)
        n = sum(c.values())
        print(f"{name:<28}{c['WIN']:>6}{c['TRAP']:>6}{c['NO-POSE']:>9}   {c['TRAP']/n:>10.0%}")

    # what happens to each arm-A' trap when the SAME checkpoint is given an MSA?
    a_, b_ = ARMS[1][0], ARMS[2][0]
    traps = [k for k in keys if cls[a_][k] == "TRAP"]
    print(f"\n{'=' * 62}")
    print(f"THE TRAPS OF THE NO-MSA ARM ({len(traps)}), AND WHAT THE MSA DOES TO THEM")
    print("=" * 62)
    if traps:
        fate = collections.Counter(cls[b_][k] for k in traps)
        print(f"  -> WIN     {fate['WIN']:>2}   the MSA FIXED the selection (the pose was already there)")
        print(f"  -> TRAP    {fate['TRAP']:>2}   still trapped")
        print(f"  -> NO-POSE {fate['NO-POSE']:>2}   the MSA made the POOL worse")
        print(f"\n  {fate['WIN']}/{len(traps)} = {fate['WIN']/len(traps):.0%} of no-MSA traps are CURED by the MSA")
    print(f"\n\nPER-INTERFACE")
    print(f"{'interface':<12}" + "".join(f"{n.split()[0]:>10}" for n, _ in ARMS))
    print("-" * 54)
    for k in keys:
        t = f"{k[0]}_{k[1]}{k[2]}"
        print(f"{t:<12}" + "".join(f"{str(cls[n][k]):>10}" for n, _ in ARMS))


if __name__ == "__main__":
    main()
