#!/usr/bin/env python
"""Define the interface set from the PREDICTIONS, in the anchor frame. No native structure.

WHY. Selection sorts poses per interface, so it needs to know which antibody-antigen chain pairs are
interfaces. We read that from the reference. Stoichiometry cannot replace it -- on 9k94, 5 antibodies
and 5 antigens give 25 possible pairs and only 5 are real, and across the benchmark enumerating all
pairs over-generates by 113 on 292. So the reference was supplying which pairs touch.

THE DEFINITION USED HERE: a chain pair is an interface if the two chains come within 5.0 A in AT LEAST
ONE pose. The cutoff is not a tuned parameter -- it is the same heavy-atom contact distance DockQ uses,
already load-bearing everywhere else in this pipeline. There is deliberately no "must appear in k
poses" rule: that would be a knob, and k=1 measured 2 missed and 37 spurious out of 292.

NAMED IN THE ANCHOR FRAME, not the crystal's. Slots are the anchor pose's chain pairs, so a pair is
labelled by the anchor's chain names even when the ANCHOR ITSELF has no contact there -- the mapping is
a chain bijection from superposition and never consults contacts. The anchor supplies names; the poses
supply which names are interfaces.

WHAT CANNOT BE SCORED. A pair with no counterpart in the reference has no ground-truth DockQ, so it is
selectable but not evaluable. Such pairs are written with have_gt=0. They still consume donor budget --
round-robin gives every interface an equal share -- which is the cost this definition pays, and the
reason to measure coverage on the evaluable interfaces before trusting it.

    pose_interfaces.py --out DIR [--jobs 8] [--poses 0]     (0 = every pose)
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402
from fmt import table  # noqa: E402

ANCHOR_DIR = "/home/ebrahim/tts_abag/diffusion_commit_clean/anchor_merged"


def one(task):
    """Contacting (ab, ag) pairs over the poses of one complex, in MODEL chain labels per pose."""
    pdb, items, abs_ag = task
    seen = collections.Counter()
    for pose, amap in items:
        if not os.path.exists(pose):
            continue
        for a_ab, a_ag in abs_ag:
            # translate the anchor-frame slot into THIS pose's chains
            if a_ab not in amap or a_ag not in amap:
                continue
            try:
                n = len(metrics.epitope_set(pose, amap[a_ab], amap[a_ag],
                                           cutoff=metrics.CONTACT_CUTOFF))
            except Exception:
                continue
            if n > 0:
                seen[(a_ab, a_ag)] += 1
    return pdb, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--poses", type=int, default=0)
    A = ap.parse_args()

    amaps = metrics.load_chain_maps(f"{ANCHOR_DIR}/chain_maps_anchor.csv")
    aroles = metrics.load_roles(f"{ANCHOR_DIR}/chain_roles_anchor.csv")
    xroles = metrics.load_roles()
    xmaps = metrics.load_chain_maps()

    # anchor-frame ab/ag chain sets per complex, and the reference interfaces translated into the
    # anchor frame so "which of these has ground truth" is answerable
    ab, ag, ref = (collections.defaultdict(set) for _ in range(3))
    for iface, (a, g) in aroles.items():
        pdb = iface.split("_")[0]
        ab[pdb].add(a)
        ag[pdb].add(g)
        ref[pdb].add((a, g))
    # single-interface complexes have no anchor-frame roles; they are trivially one pair
    for iface, (a, g) in xroles.items():
        pdb = iface.split("_")[0]
        if pdb not in ref:
            ab[pdb].add(a)
            ag[pdb].add(g)
            ref[pdb].add((a, g))

    poses = collections.defaultdict(list)
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            poses[r["pdb"]].append((r["seed"], int(r["sample"])))

    tasks = []
    for pdb in sorted(ref):
        keys = sorted(set(poses.get(pdb, ())))
        if A.poses:
            keys = keys[:A.poses]
        items = []
        for s, x in keys:
            m = amaps.get((pdb, s, x)) or xmaps.get((pdb, s, x))
            if m:
                items.append((paths.pool_pose(pdb, s, x), m))
        if not items:
            continue
        pairs = sorted({(a, g) for a in ab[pdb] for g in ag[pdb]})
        tasks.append((pdb, items, pairs))

    print(f"  complexes: {len(tasks)}   candidate pairs: {sum(len(t[2]) for t in tasks)}", flush=True)
    import multiprocessing as mp
    got = {}
    with mp.Pool(A.jobs) as p:
        for i, (pdb, seen) in enumerate(p.imap_unordered(one, tasks), 1):
            got[pdb] = seen
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    os.makedirs(A.out, exist_ok=True)
    rows, stat = [], collections.Counter()
    for pdb, seen in sorted(got.items()):
        for (a, g), n in sorted(seen.items()):
            have = (a, g) in ref[pdb]
            stat["evaluable (in the reference)" if have else "SPURIOUS (no ground truth)"] += 1
            rows.append([pdb, a, g, n, len(seen), int(have)])
        for a, g in sorted(ref[pdb] - set(seen)):
            stat["MISSED (in reference, no pose forms it)"] += 1
            rows.append([pdb, a, g, 0, len(seen), 1])
    with open(f"{A.out}/pose_interfaces.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "ab_chain", "ag_chain", "poses_in_contact", "pairs_found", "have_gt"])
        w.writerows(rows)
    print()
    print(table(["", "chain pairs"], [[k, v] for k, v in stat.most_common()], align="l r"))
    print(f"\n  wrote {A.out}/pose_interfaces.csv   ({len(rows)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
