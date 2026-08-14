#!/usr/bin/env python
"""Full-precision (donor, seed, sample) -> (interface pair_iptm, DockQ) table for an allpose arm.

WHY THIS EXISTS. <run>_scores.json rounds per_seed ipTM to 3 decimals. That is fine for reading, but
it creates ties that did not exist: in allpose8che10 seeds 108 and 137 of donor 8che_allpose_137_3
both store sel_iptm 0.741 while their poses score DockQ 0.873 and 0.938. Any analysis that re-derives
the argmax from the rounded field can pick the wrong pose. score_allpose.py itself is unaffected --
it selects on unrounded values -- but downstream resampling (bootstrap_seed_depth.py) is not.

Reads only cached artifacts: the arm's OST cache (already computed) and the per-pose confidence
JSONs. No OST calls, no GPU.

Usage: .../python scripts/build_pose_table.py --run allpose9jbq10 --ifaces 9jbq_AC:A:C
Writes: artifacts/<run>_posetable.json
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ARTIFACTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ifaces", required=True, help="comma-separated label:chain1:chain2")
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

ART = str(ARTIFACTS_DIR)
OUT = f"{ART}/sweep_out/{A.run}/{A.run}"
CACHE = f"{ART}/{A.run}_ost"
IFACES = {}
for spec in A.ifaces.split(","):
    lab, c1, c2 = spec.split(":")
    IFACES[lab] = (c1, c2)

jobs = json.load(open(f"{ART}/{A.run}_jobs.json"))


def one(args):
    """(name, seed, sample) -> {iface: [iptm, dockq]} at full precision, or None."""
    j, s, k = args
    name = j["name"]
    o = f"{CACHE}/{name}_{s}_{k}.json"
    conf = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
    if not (os.path.exists(o) and os.path.exists(conf)):
        return None
    try:
        d = json.load(open(o))
        M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):      # BY NAME, never dockq[0]
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = [M[x][y], qs[i]]                # UNROUNDED
                break
    if not got:
        return None
    return {"donor": name, "donor_seed": j["donor_seed"], "donor_sample": j["donor_sample"],
            "seed": s, "sample": k, **{lab: v for lab, v in got.items()}}


tasks = [(j, s, k) for j in jobs for s in j["seeds"] for k in range(5)]
with ThreadPoolExecutor(A.workers) as ex:
    res = [r for r in ex.map(one, tasks) if r]

out = f"{ART}/{A.run}_posetable.json"
json.dump({"run": A.run, "ifaces": list(IFACES), "n_poses": len(res), "poses": res},
          open(out, "w"))
print(f"{A.run}: {len(res)} poses (of {len(tasks)} attempted) -> {out}")
for lab in IFACES:
    v = [p[lab] for p in res if lab in p]
    if v:
        best = max(v, key=lambda x: x[0])
        print(f"  {lab}: {len(v)} poses | pooled-ipTM selection {best[1]:.3f} (ipTM {best[0]:.4f}) "
              f"| oracle {max(x[1] for x in v):.3f}")
