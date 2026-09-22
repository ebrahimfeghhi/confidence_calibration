#!/usr/bin/env python
"""Does ONE template do what five did?

Pooled ipTM selection over the single-template arms (best1 + conf1), against two baselines: the free
pool, and the five-template sweep on the SAME interfaces. Only interfaces complete in both are
compared, so the contrast is never between different interface sets.

Why this matters for the method: rescue with five templates required the donor set to be essentially
pure -- 5-of-5 good rescued 32% of decoys, 4-of-5 rescued 0% -- and no clustering we measured can
deliver a pure set of five. If ONE template works as well, the requirement drops to "surface any good
pose", which clustering can plausibly do.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                              # noqa: E402
from fmt import table                                     # noqa: E402
from separation import pool_classes                       # noqa: E402

ACC = 0.23
cls = pool_classes()


def load(path):
    made, seen = collections.defaultdict(list), collections.defaultdict(set)
    if not os.path.exists(path):
        return made, seen
    for r in csv.DictReader(open(path)):
        made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
        seen[r["iface"]].add((r["arm"], r["seed"]))
    return made, seen


one, one_seen = load(f"{paths.DATA_DIR}/single/results.csv")
five, five_seen = load(f"{paths.DATA_DIR}/top5/results.csv")
snap = {l.strip() for l in open(f"{paths.DATA_DIR}/top5/single_snapshot.txt")} \
    if os.path.exists(f"{paths.DATA_DIR}/top5/single_snapshot.txt") else set()

rows = collections.defaultdict(lambda: [[], [], []])
for f in sorted(glob.glob("matrices/*.npz")):
    i = os.path.basename(f)[:-4]
    k = cls.get(i)
    if i not in snap or len(one_seen.get(i, ())) < 1 or len(five_seen.get(i, ())) < 10:
        continue
    z = np.load(f)
    before = float(max(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))[1])
    pick = lambda P: float(P[int(np.argmax([x for x, _ in P]))][1])   # noqa: E731
    rows[k][0].append(before)
    rows[k][1].append(pick(five[i]))
    rows[k][2].append(pick(one[i]))

out = []
for k in ("decoy", "solved", "nocov"):
    b, f5, f1 = (np.array(x) for x in rows[k])
    if not len(b):
        continue
    out.append([k, len(b), round(b.mean(), 3), round(f5.mean(), 3), round(f1.mean(), 3),
                f"{int((b >= ACC).sum())}/{len(b)}",
                f"{int((f5 >= ACC).sum())}/{len(f5)}",
                f"{int((f1 >= ACC).sum())}/{len(f1)}"])
print(table(["class", "interfaces (n)", "free pool", "5-template", "1-template",
             "acceptable free", "acceptable 5-tmpl", "acceptable 1-tmpl"],
            out, align="l r r r r r r r"))
print(f"\n  progress: {len(one_seen)}/188 interfaces have a 1-template result")
