#!/usr/bin/env python
"""Within one seed, how similar are the 5 diffusion samples to each other -- good vs bad?

QUESTION. On most confident-decoy interfaces the acceptable pose appears in only a minority of a
seed's 5 samples. Either the 5 samples are one cloud and 0.23 is just a threshold crossing, or the
good pose sits somewhere genuinely different and the diffusion head occasionally jumps basins.

DATA IS ALREADY COMPUTED -- no new inference, no OST. scripts/pose_pose_dockq.py built 250x250
pose-pose DockQ matrices (GPU, exact DockQ 2.1.3) for seeds 101-150 in artifacts/pose_pose_dockq/,
with an index CSV giving seed/sample/dockq_gt per row. This just reads them.

Per seed with at least one acceptable pose, using the seed's 5x5 sub-block:
    GG  good-good pairs   (only when the seed has >=2 acceptable poses)
    GB  good-bad pairs
    BB  bad-bad pairs
All values are pose-vs-pose DockQ: 1.0 = identical pose, 0 = unrelated.

Usage: .../python scripts/within_seed_distinctness.py [--acc 0.23]
"""
import argparse, csv, json, os, sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
MDIR = f"{EXP}/artifacts/pose_pose_dockq"

ap = argparse.ArgumentParser()
ap.add_argument("--acc", type=float, default=0.23)
ap.add_argument("--interfaces", default="8che_AE,8che_BE,8ivx_AC,8oxw_AB,8oxw_AC,8av2_AC,9jbq_AC")
A = ap.parse_args()

out = {}
for lab in A.interfaces.split(","):
    npy = f"{MDIR}/dockq_{lab}_seeds101-150.npy"
    idx = f"{MDIR}/dockq_{lab}_seeds101-150_index.csv"
    if not (os.path.exists(npy) and os.path.exists(idx)):
        print(f"{lab}: MATRIX MISSING -- skipped", flush=True)
        continue
    M = np.load(npy)
    rows = list(csv.DictReader(open(idx)))
    assert len(rows) == M.shape[0] == M.shape[1], f"{lab}: index/matrix size mismatch"
    byseed = {}
    for r in rows:
        byseed.setdefault(int(r["seed"]), []).append((int(r["row"]), int(r["sample"]),
                                                      float(r["dockq_gt"])))
    print(f"\n=== {lab} (pose-pose DockQ within a seed; acceptable = GT DockQ >= {A.acc}) ===")
    print(f"{'seed':>6}{'n_good':>7}   {'good-good':<26}{'good-bad':<30}{'bad-bad':<26}")
    rec = []
    for s in sorted(byseed):
        v = sorted(byseed[s], key=lambda x: x[1])
        if len(v) != 5:
            continue
        good = [x for x in v if x[2] >= A.acc]
        bad = [x for x in v if x[2] < A.acc]
        if not good:
            continue
        # DockQ is NOT symmetric -- it depends on which structure is the reference, so M[i,j] and
        # M[j,i] differ. Average the two directions rather than arbitrarily taking one.
        sym = lambda i, j: 0.5 * (M[i, j] + M[j, i])
        gg = [sym(a[0], b[0]) for a, b in combinations(good, 2)]
        gb = [sym(a[0], b[0]) for a in good for b in bad]
        bb = [sym(a[0], b[0]) for a, b in combinations(bad, 2)]
        f = lambda arr: " ".join(f"{x:.2f}" for x in arr) if arr else "-"
        print(f"{s:>6}{len(good):>7}   {f(gg):<26}{f(gb):<30}{f(bb):<26}")
        rec.append({"seed": s, "n_good": len(good), "GG": [float(x) for x in gg],
                    "GB": [float(x) for x in gb], "BB": [float(x) for x in bb]})
    allgg = [x for r in rec for x in r["GG"]]
    allgb = [x for r in rec for x in r["GB"]]
    allbb = [x for r in rec for x in r["BB"]]
    m = lambda a: f"{np.mean(a):.3f}+-{np.std(a):.3f} (n={len(a)})" if a else "none"
    print(f"  POOLED  good-good {m(allgg)}   good-bad {m(allgb)}   bad-bad {m(allbb)}")
    out[lab] = rec

json.dump(out, open(f"{EXP}/artifacts/within_seed_distinctness.json", "w"), indent=1)
print(f"\nWROTE artifacts/within_seed_distinctness.json")
