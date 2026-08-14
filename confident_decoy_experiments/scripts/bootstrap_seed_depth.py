#!/usr/bin/env python
"""Would half the seeds have been enough? Bootstrap pooled-ipTM selection at reduced seed depth.

Each donor is run at 10 seeds, but every donor gets its OWN seed list (own seed + 9 drawn at random
per donor -- build_allpose_generic.py:12-15), so 250 donors have 250 distinct lists and there is no
shared set of seed numbers to subsample. "Half the seeds" therefore means, per donor independently,
keep k of its own 10.

Each draw: for every donor keep a random k of its seeds, pool the surviving poses across all donors,
take the single highest interface pair_iptm pose, report ITS DockQ -- exactly the label-free rule
score_allpose.py applies at full depth.

Reads <run>_posetable.json (build_pose_table.py), NOT <run>_scores.json: the latter rounds ipTM to 3
decimals, which manufactures ties and can flip the argmax to a different pose (allpose8che10 donor
137_3: seeds 108 and 137 both store 0.741 but score DockQ 0.873 vs 0.938).

Usage: .../python scripts/bootstrap_seed_depth.py --run allpose9jbq10 --ifaces 9jbq_AC --k 5 --n-boot 30
"""
import argparse, collections, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ARTIFACTS_DIR

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ifaces", required=True)
ap.add_argument("--k", type=int, default=5, help="seeds kept per donor")
ap.add_argument("--n-boot", type=int, default=30)
ap.add_argument("--rng", type=int, default=0)
ap.add_argument("--thresh", type=float, default=0.23)
A = ap.parse_args()

tab = json.load(open(f"{ARTIFACTS_DIR}/{A.run}_posetable.json"))

for lab in A.ifaces.split(","):
    # donor -> seed -> [(iptm, dockq), ...]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in tab["poses"]:
        if lab in p:
            by[p["donor"]][p["seed"]].append(tuple(p[lab]))
    if not by:
        sys.exit(f"no poses for {lab} in {A.run}")

    def pooled(donor_seed_sets):
        best_iptm, sel_dq, oracle = -1.0, float("nan"), 0.0
        for seeds in donor_seed_sets:
            for poses in seeds:
                for iptm, dq in poses:
                    if iptm > best_iptm:
                        best_iptm, sel_dq = iptm, dq
                    oracle = max(oracle, dq)
        return sel_dq, best_iptm, oracle

    full_sel, full_iptm, full_oracle = pooled([list(s.values()) for s in by.values()])

    rng = np.random.default_rng(A.rng)
    sels, oracles = [], []
    for _ in range(A.n_boot):
        sub = []
        for donor, seeds in by.items():
            keys = list(seeds)
            keep = rng.choice(len(keys), size=min(A.k, len(keys)), replace=False)
            sub.append([seeds[keys[i]] for i in keep])
        s_dq, _, orc = pooled(sub)
        sels.append(s_dq); oracles.append(orc)

    hit = sum(1 for x in sels if x >= A.thresh)
    lost = sum(1 for x, o in zip(sels, oracles) if o >= A.thresh and x < A.thresh)
    nseeds = st.median([len(s) for s in by.values()])
    print(f"\n=== {A.run} / {lab} ===")
    print(f"  donors {len(by)}, seeds/donor {nseeds:.0f}")
    print(f"  full depth: selected {full_sel:.3f} (ipTM {full_iptm:.4f})  oracle {full_oracle:.3f}")
    print(f"  bootstrap k={A.k}, {A.n_boot} draws:")
    print(f"    selected DockQ: mean {st.mean(sels):.3f}  median {st.median(sels):.3f}  "
          f"min {min(sels):.3f}  max {max(sels):.3f}  sd {st.pstdev(sels):.3f}")
    print(f"    >= {A.thresh}: {hit}/{A.n_boot} ({100*hit/A.n_boot:.0f}%)")
    print(f"    oracle within draw: mean {st.mean(oracles):.3f}  min {min(oracles):.3f}")
    print(f"    good pose existed but selection missed it: {lost}/{A.n_boot}")
