#!/usr/bin/env python
"""Derive `allpose8cheseed` from `allpose8che`: same 250 templates, run under EACH DONOR'S OWN SEED.

allpose8che runs all 250 donor templates under a single fixed sampling seed (101), which isolates the
donor effect but leaves no matched untemplated control: seed 137's donor is sampled under seed 101, so
"what would seed 137 have produced without the template" is not the comparison being made.

This arm sets the run seed to the donor's originating seed, giving a PAIRED design -- each templated
job has an exact untemplated counterpart in the free run (same seed), so templating can be scored as a
within-seed change. The cost is that donor identity and sampling seed now vary together, so within
THIS arm a donor difference cannot be separated from a seed difference. The two arms are complements:
allpose8che separates donors, allpose8cheseed provides the paired control. Both are needed.

Reuses allpose8che's template JSONs, a3m files and mmCIF ids unchanged -- the templates are identical
and only the sampling seed differs. Outputs go to a SEPARATE root (artifacts/sweep_out/allpose8cheseed)
so job names can stay the same without colliding.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_allpose_ownseed.py
Writes: artifacts/allpose8cheseed_jobs.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

src = json.load(open(f"{EXP}/artifacts/allpose8che_jobs.json"))
jobs = []
for j in src:
    k = dict(j)
    k["run"] = "allpose8cheseed"
    k["seeds"] = [j["donor_seed"]]          # <- the only change: run under the donor's own seed
    jobs.append(k)

assert all(j["seeds"] == [j["donor_seed"]] for j in jobs), "seed not matched to donor"
assert len({j["name"] for j in jobs}) == len(jobs), "duplicate job name"
assert all(os.path.exists(j["json"]) for j in jobs), "a template json is missing"
json.dump(jobs, open(f"{EXP}/artifacts/allpose8cheseed_jobs.json", "w"), indent=1)
seeds = sorted({j["seeds"][0] for j in jobs})
print(f"BUILT {len(jobs)} jobs, run seeds span {seeds[0]}-{seeds[-1]} ({len(seeds)} distinct), "
      f"all matched to donor seed -> artifacts/allpose8cheseed_jobs.json")
