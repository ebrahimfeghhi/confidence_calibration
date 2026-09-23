#!/usr/bin/env python
"""Does the label-free selection method rescue confident decoys?  [top_50_complex]

Donors are selected PER COMPLEX (round-robin over the per-interface leader lists, capped at 50) and
each fold is scored on every interface of its complex. Supersedes k50_per_interface, where tags
collided between sibling interfaces and whichever interface was claimed first supplied the donors.

SELECTION IS POOLED OVER THE FREE POSES AND THE TEMPLATED ONES TOGETHER, which is what a user has
in hand. Templated-only selection scores slightly better on decoys (4/39 vs 3/40 when measured) but
broke a solved interface; pooling with free costs one rescue and preserved 24/24.

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
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                    # noqa: E402
from fmt import table                           # noqa: E402
from separation import pool_classes             # noqa: E402

MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
ACC = 0.23
cls = pool_classes()
R = f"{paths.DATA_DIR}/top_50_complex/results.csv"
expected = collections.Counter()
for _l in open(f"{paths.DATA_DIR}/top_50_complex/queue.tsv"):
    _f = _l.rstrip("\n").split("\t")
    if len(_f) >= 6:
        expected[_f[2]] += 1
made, folds = collections.defaultdict(list), collections.defaultdict(set)
if os.path.exists(R):
    for r in csv.DictReader(open(R)):
        made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
        folds[r["iface"]].add(r["arm"])

rows = collections.defaultdict(lambda: [[], [], []])
partial = 0
for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
    i = os.path.basename(f)[:-4]
    if i not in made:
        continue
    # K VARIES PER INTERFACE: the diversity filter exhausts the pool before 50 on 228 of 292
    # interfaces, so completeness is measured against what the QUEUE asked for, not a constant.
    if len(folds[i]) < expected.get(i, 10**9):
        partial += 1
        continue
    z = np.load(f)
    free = list(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))
    before = float(max(free)[1])
    # POOL THE FREE POSES BACK IN. Selecting over templated poses alone discards 250 poses of
    # evidence and, worse, discards the PREVALENCE signal: leader selection gives a basin holding 150
    # poses the same single representative as one holding 5. On 8rwb_AC that let a pose templated from
    # the pool's best decoy (ipTM 0.9376 -> 0.9387) outrank 30 genuinely good free poses and ship
    # DockQ 0.076 in place of 0.718. Pooling with the free poses restores the baseline as a floor.
    P = free + made[i]
    after = float(P[int(np.argmax([x for x, _ in P]))][1])
    tonly = float(P2[int(np.argmax([x for x, _ in P2]))][1]) if (P2 := made[i]) else before
    k = cls.get(i, "?")
    rows[k][0].append(before)
    rows[k][1].append(after)
    rows[k][2].append(tonly)

out = []
for k in ("decoy", "solved", "nocov"):
    b, a, t = (np.array(x) for x in rows[k])
    if not len(b):
        continue
    # BOTH pooling rules. templated-only discards the free pool's 250 poses and its prevalence
    # signal; free+templated keeps the baseline as a floor. Which wins depends on donor quality, so
    # neither is reported alone.
    for nm, v in (("templated only", t), ("free + templated", a)):
        out.append([k, nm, len(b), round(b.mean(), 3), round(v.mean(), 3),
                    f"{int((b >= ACC).sum())}/{len(b)}", f"{int((v >= ACC).sum())}/{len(v)}",
                    int(((b < ACC) & (v >= ACC)).sum()), int(((b >= ACC) & (v < ACC)).sum())])
print(table(["class", "pooling", "interfaces", "mean before", "mean after",
             "acceptable before", "acceptable after", "rescued", "broken"],
            out, align="l l r r r r r r r"))
print(f"\n  complete interfaces scored: {sum(len(v[0]) for v in rows.values())}   "
      f"partial (excluded): {partial}   "
      f"folds run: {len(os.listdir(f'{paths.DATA_DIR}/top_50_complex/claims'))} claimed "
      f"+ 428 carried over, of {sum(1 for _ in open(f'{paths.DATA_DIR}/top_50_complex/queue.tsv'))} queued")
