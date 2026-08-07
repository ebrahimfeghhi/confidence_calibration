#!/usr/bin/env python
"""8che_BE, leader20 at 50 seeds: is a GOOD donor's ipTM distribution higher than a BAD donor's?

leader20 failed on 8che_BE even at 50 seeds/template (5000 poses, selected DockQ 0.162). This asks
whether the failure is visible in ipTM at the TEMPLATE level: if templating on an acceptable leader
yields a systematically higher median/mean ipTM than templating on an unacceptable one, then the
per-template ipTM summary is itself a donor-selection signal. If not, ipTM cannot tell the good
templates from the bad ones here.

Per leader rank: n poses, median/mean/max interface pair_iptm, median/max DockQ, and the donor's own
DockQ vs native (the oracle label, used only to group -- never inside the signal).

Reads cached OST where present and calls OST only for uncached poses (taskset-pinned).
Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/che_leader_iptm.py
Writes: artifacts/che_leader_iptm.json
"""
import json, os, sys, subprocess, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SAMPLES, THRESH, OST, ENV, GT

LAB, PDB = "8che_BE", "8che"
OUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"
CACHE = f"{EXP}/artifacts/leader20_ost"
jobs = [j for j in json.load(open(f"{EXP}/artifacts/chedeep_jobs.json")) if j["interface"] == LAB]
L = {x["rank"]: x for x in json.load(open(f"{EXP}/artifacts/leader20_index.json"))[LAB]["leaders"]}
c1, c2 = LAB.split("_")[1][0], LAB.split("_")[1][1]


def one(a):
    j, s, k = a
    name = j["name"]
    cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{LAB}_L{j['leader_rank']:02d}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", cif,
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
    for i, t in enumerate(ifs):                       # interface matched BY NAME
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                return (M[x][y], qs[i])
    return None


rows = []
for j in sorted(jobs, key=lambda x: x["leader_rank"]):
    tasks = [(j, s, k) for s in j["seeds"] for k in SAMPLES]
    with ThreadPoolExecutor(12) as ex:
        got = [g for g in ex.map(one, tasks) if g]
    if not got:
        continue
    ip = [g[0] for g in got]; dq = [g[1] for g in got]
    r = j["leader_rank"]; lead = L.get(r, {})
    rows.append({"leader_rank": r, "n_poses": len(got),
                 "donor_dockq": lead.get("dockq_gt"), "donor_acceptable": bool(lead.get("acceptable")),
                 "iptm_median": round(st.median(ip), 3), "iptm_mean": round(st.mean(ip), 3),
                 "iptm_max": round(max(ip), 3),
                 "dockq_median": round(st.median(dq), 3), "dockq_max": round(max(dq), 3),
                 "frac_acceptable": round(sum(1 for d in dq if d >= THRESH) / len(dq), 3)})
    v = rows[-1]
    print(f"L{r:02d} {'GOOD' if v['donor_acceptable'] else 'bad '} donorDockQ "
          f"{str(v['donor_dockq']):>5} | n {v['n_poses']:4d} | ipTM med {v['iptm_median']:.3f} "
          f"mean {v['iptm_mean']:.3f} max {v['iptm_max']:.3f} | DockQ med {v['dockq_median']:.3f} "
          f"max {v['dockq_max']:.3f} | frac_acc {v['frac_acceptable']:.3f}", flush=True)

g = [r for r in rows if r["donor_acceptable"]]
b = [r for r in rows if not r["donor_acceptable"]]
summ = {}
for m in ("iptm_median", "iptm_mean", "iptm_max"):
    summ[m] = {"good_donors": round(st.median([r[m] for r in g]), 3) if g else None,
               "bad_donors": round(st.median([r[m] for r in b]), 3) if b else None}
print(f"\n  {len(g)} GOOD donor templates, {len(b)} bad ones")
print(f"  {'metric':14} {'good (median over tmpl)':>24} {'bad':>8}")
for m, v in summ.items():
    print(f"  {m:14} {str(v['good_donors']):>24} {str(v['bad_donors']):>8}")
if g and b:
    hi = sum(1 for x in g for y in b if x["iptm_median"] > y["iptm_median"])
    print(f"\n  pairwise: a GOOD donor has higher median ipTM than a bad one in "
          f"{hi}/{len(g)*len(b)} pairs (AUC {hi/(len(g)*len(b)):.2f})")
json.dump({"interface": LAB, "seeds_per_template": len(jobs[0]["seeds"]),
           "summary": summ, "rows": rows},
          open(f"{EXP}/artifacts/che_leader_iptm.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/che_leader_iptm.json")
