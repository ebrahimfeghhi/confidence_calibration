#!/usr/bin/env python
"""Score the `leader20` method per interface, as each interface's 20 template runs complete.

THE METHOD'S ANSWER IS POOLED, not per-run: for one interface, all 20 template conditions x 5 seeds x
5 samples = up to 500 poses go into one pool, and the method's output is the single highest
interface-`pair_iptm` pose in that pool. So an interface becomes scorable only when its templates have
finished -- unlike previous arms, where each job was independently scorable.

Reports per interface:
  method_dockq        DockQ of the pooled top-ipTM pose  <- the method's actual answer
  method_rescued      method_dockq >= 0.23
  winning_leader      which template condition produced the winner (1 = the decoy's own fold)
  best_available      best DockQ anywhere in the pool (the ceiling pooling could reach)
  n_acceptable        how many of the pooled poses are acceptable
  free_*              the free run's numbers, for comparison

Also records, for the interfaces still generating, how many templates are done -- so partial progress
is visible without pretending a partial pool is the answer.

SCORING PROTOCOL unchanged (tts_abag/eval/eval_ost_foldbench.py): one OST call per pose, full native
assembly vs full model, interface extracted BY NAME.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_leader20.py [--min-templates 20]
Writes: artifacts/leader20_scores.json
"""
import argparse, json, os, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT, EXP, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--min-templates", type=int, default=20,
                help="only score an interface once this many template runs are complete")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--jobs", default=None, help="jobs file; default the 5-seed leader20 set")
ap.add_argument("--out-suffix", default="", help="write to artifacts/leader20_scores<suffix>.json")
A = ap.parse_args()

OUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"
CACHE = f"{EXP}/artifacts/leader20_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/leader20_scores{A.out_suffix}.json"
jobs = json.load(open(A.jobs if A.jobs else f"{EXP}/artifacts/leader20_jobs.json"))
index = json.load(open(f"{EXP}/artifacts/leader20_index.json"))
free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))

by_iface = collections.defaultdict(list)
for j in jobs:
    by_iface[j["interface"]].append(j)


def pose(a):
    """(pair_iptm, dockq, leader_rank, seed, sample) for one pose, or None."""
    job, s, k = a
    name = job["name"]
    cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    lab = job["interface"]
    o = f"{CACHE}/{lab}_L{job['leader_rank']:02d}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{job['pdb']}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(f"{OUT}/{name}/seed_{s}/predictions/"
                          f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):        # BY NAME
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                return (M[x][y], qs[i], job["leader_rank"], s, k)
    return None


def n_done(job):
    return sum(1 for s in job["seeds"] if os.path.exists(
        f"{OUT}/{job['name']}/seed_{s}/predictions/{job['name']}_summary_confidence_sample_4.json"))


res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
pending = []
for lab, js in sorted(by_iface.items()):
    if lab in res:
        continue
    complete = [j for j in js if n_done(j) >= len(j["seeds"])]
    if len(complete) < min(A.min_templates, len(js)):
        pending.append(f"{lab}: {len(complete)}/{len(js)} templates done"); continue

    tasks = [(j, s, k) for j in complete for s in j["seeds"] for k in SAMPLES]
    with ThreadPoolExecutor(A.workers) as ex:
        got = [g for g in ex.map(pose, tasks) if g]
    if not got:
        print(f"{lab:11} NO POSES SCORED", flush=True); continue

    win = max(got, key=lambda x: x[0])            # the method's answer: pooled top ipTM
    ok = [p for p in got if p[1] >= THRESH]
    f = free.get(lab, {})
    shortlist_has_good = any(x["acceptable"] for x in index[lab]["leaders"])
    res[lab] = {
        "pdb": index[lab]["pdb"], "n_templates": len(complete), "n_poses": len(got),
        "method_iptm": round(win[0], 3), "method_dockq": round(win[1], 3),
        "method_rescued": win[1] >= THRESH,
        "winning_leader": win[2], "winning_seed": win[3], "winning_sample": win[4],
        "best_available": round(max(p[1] for p in got), 3),
        "n_acceptable": len(ok), "frac_acceptable": round(len(ok) / len(got), 4),
        "shortlist_contained_good_pose": shortlist_has_good,
        "free_n_acceptable": f.get("free_n_acceptable"), "free_sel_dockq": f.get("free_sel_dockq"),
        "free_best_dockq": f.get("free_best_dockq"),
    }
    v = res[lab]
    print(f"{lab:11} {v['n_templates']:2d} tmpl {v['n_poses']:4d} poses | method DockQ "
          f"{f.get('free_sel_dockq')} -> {v['method_dockq']:.3f} (ipTM {v['method_iptm']:.3f}, "
          f"leader {v['winning_leader']:2d}) | pool best {v['best_available']:.3f}"
          f"{'   RESCUED' if v['method_rescued'] else ''}", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)

n = sum(1 for v in res.values() if v["method_rescued"])
print(f"\nleader20 METHOD: {n}/{len(res)} interfaces rescued")
if res:
    sl = sum(1 for v in res.values() if v["shortlist_contained_good_pose"])
    print(f"  shortlist contained a good pose: {sl}/{len(res)}")
    print(f"  conversion (shortlist had one -> method found one): {n}/{sl}" if sl else "")
    from collections import Counter
    print(f"  winning leader rank: {dict(Counter(v['winning_leader'] for v in res.values()))}")
if pending:
    print(f"\nstill generating ({len(pending)}):")
    for p in pending[:10]:
        print("  " + p)
print(f"\nWROTE {OUTJ}")
