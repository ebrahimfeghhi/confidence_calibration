#!/usr/bin/env python
"""Incremental scorer for the pddag run: scores each interface as soon as its PDB's 50 seeds
finish (does not wait for the whole run). Reproduces the canonical experiment scoring EXACTLY
(same OST flags as incr_score.py that produced agonly/oracle numbers) so results are comparable.

Per pose: OST compare-structures --dockq (model vs native) -> DockQ for interface (c1,c2);
pair_iptm read from the summary_confidence json's chain_pair_iptm matrix indexed by the MODEL
chains OST mapped to the interface. Selection rule: max-pair_iptm pose's DockQ >= 0.23 => rescued.
Writes per-interface scores + a running summary. Run with base conda python (only needs json/glob)."""
import json, os, glob, subprocess, time, sys
from concurrent.futures import ThreadPoolExecutor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
OUT  = f"{EXP}/artifacts/pddag_out"
SUF  = "pddag"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
OST  = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV  = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE   = f"{EXP}/artifacts/pddag_ost_incr"; os.makedirs(CACHE, exist_ok=True)
SCORES  = f"{EXP}/artifacts/pddag_scores.json"
SUMMARY = f"{EXP}/artifacts/pddag_scores_summary.json"
meta = json.load(open(f"{EXP}/artifacts/meta.json"))

def out(m): print(m, flush=True)
def run_ost(m, n, o):
    if os.path.exists(o) and os.path.getsize(o) > 0: return
    subprocess.run(["taskset","-c","0-49",OST,"compare-structures","-m",m,"-r",n,"-o",o,
                    "--fault-tolerant","--min-pep-length","4","--dockq"],
                   env=ENV, capture_output=True, timeout=300)
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
def score_iface(ifc, pdb, c1, c2):
    name = f"{pdb}_{SUF}"; nat = f"{GT}/{pdb}-assembly1.cif"; jobs = []
    for s in sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{name}/seed_*")):
        for k in range(5):
            cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
            cf  = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if os.path.exists(cif): jobs.append((cif, cf, f"{CACHE}/{ifc}_{s}_{k}.json"))
    with ThreadPoolExecutor(max_workers=12) as e:
        list(e.map(lambda j: run_ost(j[0], nat, j[2]), jobs))
    poses = []
    for cif, cf, o in jobs:
        r = ex(o, cf, c1, c2)
        if r and r[0] is not None: poses.append({"pair_iptm": r[1], "dockq": r[0]})
    if not poses: return None
    sel = max(poses, key=lambda x: (x["pair_iptm"] if x["pair_iptm"] is not None else -1))
    return poses, sel

bypdb = {}
for ifc, m in meta.items(): bypdb.setdefault(m["pdb"], []).append((ifc, m["c1"], m["c2"]))
total = len(meta); done = {}
allscores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
out(f"INCR[pddag] scorer started: {len(bypdb)} pdbs / {total} interfaces")
t0 = time.time()
while len(done) < total and time.time() - t0 < 60000:
    for pdb, ifs in bypdb.items():
        if all(ifc in done for ifc, _, _ in ifs): continue
        if len(glob.glob(f"{OUT}/{pdb}_{SUF}/seed_*")) < 50: continue
        for ifc, c1, c2 in ifs:
            if ifc in done: continue
            r = score_iface(ifc, pdb, c1, c2)
            if r is None: continue
            poses, sel = r
            done[ifc] = {"sel_dockq": sel["dockq"], "sel_pair_iptm": sel["pair_iptm"],
                         "n_poses": len(poses), "rescued": sel["dockq"] >= 0.23}
            allscores[ifc] = poses
            json.dump(allscores, open(SCORES, "w"))
            resc = sorted([i for i, v in done.items() if v["rescued"]])
            json.dump({"scored": len(done), "total": total, "rescued": f"{len(resc)}/{total}",
                       "rescued_ifaces": resc, "per_iface": done},
                      open(SUMMARY, "w"), indent=1)
            pi = sel["pair_iptm"] if sel["pair_iptm"] is not None else 0
            out(f"RESULT[pddag] {ifc}: sel_DockQ={sel['dockq']:.3f} sel_pair_iptm={pi:.3f} "
                f"-> {'RESCUED' if sel['dockq']>=0.23 else 'not-rescued'}  "
                f"[{len(done)}/{total}, running rescued={len(resc)}]")
    time.sleep(60)
resc = sorted([i for i, v in done.items() if v["rescued"]])
out(f"ALL_DONE[pddag]: {len(done)}/{total} scored; RESCUED {len(resc)}/{total} -> {resc}")
