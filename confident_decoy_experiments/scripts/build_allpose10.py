#!/usr/bin/env python
"""allpose8che10: all 250 donor templates x 10 seeds -- BUDGET-MATCHED against the 2500-seed free run.

THE QUESTION. freedeep8che spends 2500 untemplated seeds (12500 poses) on 8che. This spends the SAME
2500 seeds, but split across 250 donor templates at 10 seeds each (also 12500 poses). Same compute,
same pose count, same interfaces -- the only difference is whether the sampling is templated. That
isolates "does templating help" from "does more sampling help", which every previous comparison
confounded (dockqdonor's 250 poses vs leader20's 5000 vs free's 250).

SEEDS PER DONOR = the donor's OWN seed + 9 drawn at random (rng seeded, so it reproduces). Rationale:
  - Ten seeds hedges the failure that wrecked allpose8che, which ran every donor at seed 101 alone --
    rank 44/50 for the one donor known to work -- and returned 0/250 donors producing a >0.6 pose,
    while the same donors at their own seeds returned 2/250 and a 0.948 pooled answer. The known-good
    donor clears 0.23 on 38/50 seeds, so P(all 10 fail) < 1e-6.
  - Including the OWN seed guarantees every donor gets the one draw that reproduces the condition
    under which its pose was originally generated, so a donor cannot be scored purely on foreign
    seeds. (Own-seed is not measurably special -- donor_own_seed.py puts its mean rank at 22.8/50 vs
    25.5 expected across 25 interfaces -- but it costs nothing to include and removes the objection.)
  - The other 9 are RANDOM per donor rather than a fixed set, so one unlucky seed cannot depress every
    donor simultaneously the way seed 101 did.
  Cost: donors no longer share a seed set, so a donor difference is not perfectly isolated from a seed
  difference. With 10 draws each the residual is small, and allpose8che (fixed seed) remains on disk
  as the perfectly-isolated-but-unlucky comparison.

Reuses allpose8che's 250 template JSONs, a3m files and mmCIF ids (p000-p249) unchanged -- identical
templates, only the seed list differs. Separate output root so names need not change.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_allpose10.py [--seeds 10]
Writes: artifacts/allpose8che10_jobs.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--pool-lo", type=int, default=101)
ap.add_argument("--pool-hi", type=int, default=1100, help="range random seeds are drawn from")
ap.add_argument("--rng", type=int, default=0)
A = ap.parse_args()

import numpy as np
rng = np.random.default_rng(A.rng)
src = json.load(open(f"{EXP}/artifacts/allpose8che_jobs.json"))
pool = np.arange(A.pool_lo, A.pool_hi + 1)
jobs = []
for j in src:
    own = j["donor_seed"]
    cand = pool[pool != own]
    extra = rng.choice(cand, size=A.seeds - 1, replace=False)
    k = dict(j)
    k["run"] = "allpose8che10"
    k["seeds"] = [int(own)] + sorted(int(x) for x in extra)   # own seed ALWAYS included, first
    k["own_seed_included"] = True
    jobs.append(k)

assert len({j["name"] for j in jobs}) == len(jobs), "duplicate job name"
assert all(os.path.exists(j["json"]) for j in jobs), "a template json is missing"
assert all(len(j["seeds"]) == A.seeds for j in jobs), "seed count mismatch"
assert all(len(set(j["seeds"])) == A.seeds for j in jobs), "duplicate seed within a job"
assert all(j["donor_seed"] in j["seeds"] for j in jobs), "a donor's own seed is missing"
json.dump(jobs, open(f"{EXP}/artifacts/allpose8che10_jobs.json", "w"), indent=1)
tot = len(jobs) * A.seeds
print(f"BUILT {len(jobs)} donor jobs x {A.seeds} seeds = {tot} seeds ({tot*5} poses)")
print(f"   budget match: freedeep8che = 2349 new + 151 existing = 2500 seeds (12500 poses)")
print(f"   this arm     = {tot} seeds ({tot*5} poses)")
allseeds = sorted({s for j in jobs for s in j["seeds"]})
print(f"   seeds per donor = own + {A.seeds-1} random from {A.pool_lo}-{A.pool_hi} (rng={A.rng})")
print(f"   distinct seeds used across all donors: {len(allseeds)} ({allseeds[0]}-{allseeds[-1]})")
print(f"   example: {jobs[0]['name']} donor_seed {jobs[0]['donor_seed']} -> {jobs[0]['seeds']}")
