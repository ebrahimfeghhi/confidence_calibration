#!/usr/bin/env python
"""Score freedeep8che: does UNTEMPLATED sampling reach the band ipTM can see?

12500 free poses (2500 seeds) on 8che, against the measured conversion thresholds: if a pool's best
pose is 0.23-0.40, ipTM selects an acceptable one 2% of the time; 0.40-0.60 -> 14%; above 0.60 -> ~80%.
So the question is not whether free sampling makes an acceptable pose (it already does, 92/755) but
whether it makes one above ~0.6 -- which it has NEVER done on 8che_BE in 755 poses.

Pools the new seeds with the 151 pre-existing ones (0-150, scored by score_free_extra.py) for a 2500-
seed total, and reports the new block and the pooled total separately.

Protocol identical to the rest of the project (one OST call per pose, full native vs full model,
--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq, interface matched BY NAME).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_freedeep.py [--workers 20]
Writes: artifacts/freedeep8che_scores.json
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, OST, ENV, GT, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=20)
A = ap.parse_args()
PDB = "8che"
IFACES = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
OUT = f"{EXP}/artifacts/sweep_out/freedeep8che/freedeep8che"
CACHE = f"{EXP}/artifacts/freedeep8che_ost"; os.makedirs(CACHE, exist_ok=True)
jobs = json.load(open(f"{EXP}/artifacts/freedeep8che_jobs.json"))


def pose(a):
    name, s, k = a
    cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{PDB}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(f"{OUT}/{name}/seed_{s}/predictions/"
                           f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = (M[x][y], qs[i], s, k)
                break
    return got or None


tasks = [(j["name"], s, k) for j in jobs for s in j["seeds"] for k in SAMPLES]
with ThreadPoolExecutor(A.workers) as ex:
    new = [r for r in ex.map(pose, tasks) if r]
print(f"NEW seeds scored: {len(new)} poses (of {len(tasks)} slots; run may still be generating)\n")

prior = json.load(open(f"{EXP}/artifacts/free_extra_8che.json")) if \
    os.path.exists(f"{EXP}/artifacts/free_extra_8che.json") else {}
out = {}
for lab in IFACES:
    v = [r[lab] for r in new if lab in r]
    if not v:
        continue
    d = np.array([x[1] for x in v]); ip = np.array([x[0] for x in v])
    sel = max(v, key=lambda x: x[0])
    p = prior.get(lab, {}).get("POOLED 0-150", {})
    rec = {
        "new_poses": len(v), "new_best": round(float(d.max()), 3),
        "new_n_acc": int((d >= THRESH).sum()), "new_n_06": int((d >= 0.6).sum()),
        "new_n_085": int((d >= 0.85).sum()),
        "new_sel_dockq": round(sel[1], 3), "new_sel_iptm": round(sel[0], 3),
        "new_sel_seed": sel[2], "new_sel_sample": sel[3],
        "prior_poses": p.get("n"), "prior_best": p.get("best"),
        "prior_n_06": p.get("n_06"), "prior_sel_dockq": p.get("sel_dockq"),
    }
    if p:
        rec["total_poses"] = len(v) + p["n"]
        rec["total_best"] = round(max(float(d.max()), p["best"]), 3)
        rec["total_n_06"] = int((d >= 0.6).sum()) + p["n_06"]
        rec["total_n_085"] = int((d >= 0.85).sum()) + p["n_085"]
        # pooled selection: whichever block holds the single highest-ipTM pose
        rec["total_sel_dockq"] = round(sel[1] if sel[0] >= p["sel_iptm"] else p["sel_dockq"], 3)
    out[lab] = rec
    print(f"=== {lab} ===")
    print(f"  NEW   {rec['new_poses']:6d} poses | best {rec['new_best']:.3f} | "
          f">=0.23 {rec['new_n_acc']:5d} | >=0.6 {rec['new_n_06']:4d} | >=0.85 {rec['new_n_085']:4d} "
          f"| top-ipTM selects {rec['new_sel_dockq']:.3f}")
    if p:
        print(f"  PRIOR {p['n']:6d} poses | best {p['best']:.3f} | "
              f">=0.23 {p['n_acc']:5d} | >=0.6 {p['n_06']:4d} | >=0.85 {p['n_085']:4d} "
              f"| top-ipTM selects {p['sel_dockq']:.3f}")
        print(f"  TOTAL {rec['total_poses']:6d} poses | best {rec['total_best']:.3f} | "
              f">=0.6 {rec['total_n_06']:4d} | >=0.85 {rec['total_n_085']:4d} "
              f"| top-ipTM selects {rec['total_sel_dockq']:.3f}")
    print()
json.dump(out, open(f"{EXP}/artifacts/freedeep8che_scores.json", "w"), indent=1)
print(f"WROTE {EXP}/artifacts/freedeep8che_scores.json")
