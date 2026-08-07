#!/usr/bin/env python
"""freedeep8che: 2500 additional UNTEMPLATED seeds of 8che, to test whether sampling alone gets there.

WHY. On 8che_BE, one donor template (dockqdonor's seed137/s3) reaches DockQ 0.958 and puts 58% of its
poses above 0.85, while 20 leader templates at 50 seeds each (5000 poses) top out at 0.671 and the
free run's own 250 poses top out at 0.529. The open question is whether the free model can reach the
0.85+ band AT ALL given enough draws, or whether the template is doing something sampling cannot.
2500 seeds = 12500 poses, 50x the free run's current depth.

"NO TEMPLATE" HERE MEANS: the base FoldBench json unchanged -- which DOES include the natural
hmmsearch template search (templatesPath -> prep/8che/msa/*/hmmsearch.a3m), exactly as the free run
used. It is NOT templates-disabled. This makes the 12500 new poses a direct extension of the existing
250-pose free pool rather than a different condition.

SEEDS 1000-3499, chosen to not collide with any existing range (101-150 canonical, 0-100 added for
some pdbs).

Split into 4 jobs so the scheduler puts one on each GPU and the model loads once per job rather than
once per seed. Each job gets a UNIQUE name -- four jobs sharing a name would share one log file and
silently overwrite each other's template-loading record.

Covers 8che_AE and 8che_BE both.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_free_deep.py [--seeds 2500]
Writes: artifacts/sweep_tmpl/freedeep8che/*.json, artifacts/freedeep8che_jobs.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2500)
ap.add_argument("--start", type=int, default=1000)
ap.add_argument("--splits", type=int, default=4)
A = ap.parse_args()

PDB = "8che"
TMPL = f"{EXP}/artifacts/sweep_tmpl/freedeep8che"
os.makedirs(TMPL, exist_ok=True)
base = json.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{PDB}-final-updated.json"))

allseeds = list(range(A.start, A.start + A.seeds))
per = -(-len(allseeds) // A.splits)
jobs = []
for i in range(A.splits):
    chunk = allseeds[i * per:(i + 1) * per]
    if not chunk:
        continue
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    name = f"{PDB}_freedeep_{chr(97+i)}"
    obj["name"] = name                                  # unique per job -- see docstring
    # templatesPath left EXACTLY as the base json has it (natural hmmsearch search) -- untouched
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": "freedeep8che", "pdb": PDB, "interface": f"{PDB}_free_{chr(97+i)}",
                 "json": jp, "name": name, "seeds": chunk, "set": "free"})

json.dump(jobs, open(f"{EXP}/artifacts/freedeep8che_jobs.json", "w"), indent=1)
assert len({j["name"] for j in jobs}) == len(jobs), "duplicate job name"
assert sum(len(j["seeds"]) for j in jobs) == A.seeds, "seed count mismatch"
assert len(set(s for j in jobs for s in j["seeds"])) == A.seeds, "duplicate seed across jobs"
for j in jobs:
    o = json.load(open(j["json"]))[0]
    assert all("templatesPath" in e["proteinChain"] for e in o["sequences"] if "proteinChain" in e), \
        "a chain lost its natural templatesPath -- this would NOT be the free condition"
print(f"BUILT {len(jobs)} jobs x ~{per} seeds = {A.seeds} seeds "
      f"({A.seeds*5} poses), seeds {allseeds[0]}-{allseeds[-1]}")
for j in jobs:
    print(f"   {j['name']}: seeds {j['seeds'][0]}-{j['seeds'][-1]} ({len(j['seeds'])})")
