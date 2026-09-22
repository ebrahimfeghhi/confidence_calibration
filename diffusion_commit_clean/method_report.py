#!/usr/bin/env python
"""Does the label-free selection method rescue confident decoys?

WHAT IS BEING SCORED. For each interface the method proposed 20 donor pairs with no access to ground
truth: anchor on the top-ipTM pose, order the rest by DEGREE (how many poses lie within pose-pose
DockQ 0.23), walk that order keeping only poses dissimilar to everything kept, then pair each leader
with its nearest unused pose. Every pair was self-templated. Here we pool everything the 20 folds
produced, pick the single most confident pose, and compare its DockQ to what the free pool would have
shipped. Ground truth enters ONLY at this scoring step.

POOLED, NEVER WITHIN-FOLD. In use you cannot tell which of the 20 folds came from a good donor, so
selection has to range over all of them at once. Scoring the best fold instead would be a ceiling, and
reporting one earlier in this project inflated 5 of 32 into 8 of 24.

PARTIAL INTERFACES ARE EXCLUDED. An interface with 3 of its 20 folds done has had only a fraction of
its shortlist explored, and counting it would understate the method early and drift upward as folds
land -- which looks like a trend and is an artefact.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                    # noqa: E402
from fmt import table                           # noqa: E402
from separation import pool_classes             # noqa: E402

ACC, K = 0.23, 20
cls = pool_classes()
R = f"{paths.DATA_DIR}/method/results.csv"
made, folds = collections.defaultdict(list), collections.defaultdict(set)
if os.path.exists(R):
    for r in csv.DictReader(open(R)):
        made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
        folds[r["iface"]].add(r["arm"])

rows = collections.defaultdict(lambda: [[], []])
partial = 0
for f in sorted(glob.glob("matrices/*.npz")):
    i = os.path.basename(f)[:-4]
    if i not in made:
        continue
    if len(folds[i]) < K:
        partial += 1
        continue
    z = np.load(f)
    before = float(max(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))[1])
    P = made[i]
    after = float(P[int(np.argmax([x for x, _ in P]))][1])
    rows[cls.get(i, "?")][0].append(before)
    rows[cls.get(i, "?")][1].append(after)

out = []
for k in ("decoy", "solved", "nocov"):
    b, a = (np.array(x) for x in rows[k])
    if not len(b):
        continue
    out.append([k, len(b), round(b.mean(), 3), round(a.mean(), 3),
                f"{int((b >= ACC).sum())}/{len(b)}", f"{int((a >= ACC).sum())}/{len(a)}",
                int(((b < ACC) & (a >= ACC)).sum()), int(((b >= ACC) & (a < ACC)).sum())])
print(table(["class", "interfaces complete", "mean before", "mean after",
             "acceptable before", "acceptable after", "rescued", "broken"], out, align="l r r r r r r r"))
print(f"\n  complete interfaces scored: {sum(len(v[0]) for v in rows.values())}   "
      f"partial (excluded): {partial}   folds done: {sum(len(v) for v in folds.values())}/3935")
