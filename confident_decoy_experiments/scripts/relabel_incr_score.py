#!/usr/bin/env python
"""Incremental scorer for the label-bug re-runs (agonly/abonly/pddag x 8yxi/8fbw, corrected labels).
Scores each interface as soon as its job's 50 seeds finish. Same OST flags + selection rule as every
other run (top interface pair_iptm pose's DockQ >= 0.23), so results are directly comparable.

KEY QUESTION: agonly previously "rescued" 8yxi_AC, 8yxi_BC, 8fbw_AC — but templated the HEAVY chain by
mistake. With the antigen correctly templated, do they still rescue (=> genuinely antigen-driven) or
not (=> they were antibody-driven, revising the antigen-vs-antibody disjoint decomposition)?"""
import json, os, glob, subprocess, time
from concurrent.futures import ThreadPoolExecutor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
OUT  = f"{EXP}/artifacts/relabel_out"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
OST  = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV  = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE = f"{EXP}/artifacts/relabel_ost"; os.makedirs(CACHE, exist_ok=True)
SCORES  = f"{EXP}/artifacts/relabel_scores.json"
SUMMARY = f"{EXP}/artifacts/relabel_scores_summary.json"
meta = json.load(open(f"{EXP}/artifacts/meta.json"))
RUNS = ["agonly", "abonly", "pddag"]
PDBS = ["8yxi", "8fbw"]
# prior (buggy-label) results for direct comparison
PRIOR = {("agonly","8yxi_AC"):"RESCUED*", ("agonly","8yxi_BC"):"RESCUED*", ("agonly","8fbw_AC"):"RESCUED*",
         ("abonly","8yxi_AC"):"no", ("abonly","8yxi_BC"):"no", ("abonly","8fbw_AC"):"no",
         ("pddag","8yxi_AC"):"no", ("pddag","8yxi_BC"):"no", ("pddag","8fbw_AC"):"no"}

def out(m): print(m, flush=True)
def run_ost(m, n, o):
    # delegates to eval/ost_dockq.py -- the single OST entry point. This used to be a
    # local copy; the copies drifted and five of them still passed --rigid-scores, which
    # crashes on small interfaces and loses the whole score.
    return _ost_run(m, n, o)

def ex(o, cf, c1, c2):
    try: d = json.load(open(o))
    except Exception: return None
    dqv = mc = None
    for k, t in enumerate(d.get("dockq_interfaces") or []):
        if {t[0], t[1]} == {c1, c2}:
            q = d.get("dockq") or []; dqv = q[k] if k < len(q) else None; mc = (t[2], t[3]); break
    pi = None
    try:
        c = json.load(open(cf)); M = c.get("chain_pair_iptm")
        if mc and M:
            i = ord(mc[0]) - ord('A'); j = ord(mc[1]) - ord('A')
            if 0 <= i < len(M) and 0 <= j < len(M): pi = M[i][j]
    except Exception: pass
    return dqv, pi

def score(name, ifc, pdb, c1, c2):
    nat = f"{GT}/{pdb}-assembly1.cif"; jobs = []
    for s in sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{name}/seed_*")):
        for k in range(5):
            cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
            cf  = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if os.path.exists(cif): jobs.append((cif, cf, f"{CACHE}/{name}_{ifc}_{s}_{k}.json"))
    with ThreadPoolExecutor(max_workers=10) as e:
        list(e.map(lambda j: run_ost(j[0], nat, j[2]), jobs))
    poses = []
    for cif, cf, o in jobs:
        r = ex(o, cf, c1, c2)
        if r and r[0] is not None: poses.append({"pair_iptm": r[1], "dockq": r[0]})
    if not poses: return None
    sel = max(poses, key=lambda x: (x["pair_iptm"] if x["pair_iptm"] is not None else -1))
    return poses, sel

targets = []
for run in RUNS:
    for pdb in PDBS:
        for ifc, m in meta.items():
            if m["pdb"] == pdb: targets.append((f"{pdb}_{run}", run, ifc, pdb, m["c1"], m["c2"]))
allscores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
done = {}
out(f"INCR[relabel] scorer started: {len(targets)} (run,interface) targets")
t0 = time.time()
while len(done) < len(targets) and time.time() - t0 < 60000:
    for name, run, ifc, pdb, c1, c2 in targets:
        key = f"{run}:{ifc}"
        if key in done: continue
        if len(glob.glob(f"{OUT}/{name}/seed_*")) < 50: continue
        r = score(name, ifc, pdb, c1, c2)
        if r is None: continue
        poses, sel = r
        resc = sel["dockq"] >= 0.23
        done[key] = {"run": run, "interface": ifc, "sel_dockq": sel["dockq"],
                     "sel_pair_iptm": sel["pair_iptm"], "n_poses": len(poses), "rescued": resc,
                     "prior_buggy_label": PRIOR.get((run, ifc), "?")}
        allscores[key] = poses
        json.dump(allscores, open(SCORES, "w"))
        json.dump({"scored": len(done), "total": len(targets), "per_target": done}, open(SUMMARY, "w"), indent=1)
        pi = sel["pair_iptm"] if sel["pair_iptm"] is not None else 0
        flag = ""
        prior = PRIOR.get((run, ifc))
        if prior and prior.startswith("RESCUED") and not resc: flag = "   <<< FLIPPED: was rescued with the WRONG (heavy) chain => antibody-driven"
        if prior == "no" and resc: flag = "   <<< NEW rescue after label fix"
        out(f"RESULT[relabel] {run:6} {ifc:9}: sel_DockQ={sel['dockq']:.3f} pair_iptm={pi:.3f} "
            f"-> {'RESCUED' if resc else 'not-rescued':12} (prior[buggy]={prior}) [{len(done)}/{len(targets)}]{flag}")
    time.sleep(45)
out(f"ALL_DONE[relabel]: {len(done)}/{len(targets)}")
for k, v in sorted(done.items()):
    out(f"  {k:18} DockQ={v['sel_dockq']:.3f} {'RESCUED' if v['rescued'] else '-':8} prior={v['prior_buggy_label']}")
