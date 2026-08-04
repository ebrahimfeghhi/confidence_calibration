#!/usr/bin/env python
"""Score any PER-INTERFACE template arm (dockqdonor, ifacelddt26, globallddt26).

The older `score_sweep.py` assumes one run per PDB and the 15-decoy set. The arms built for §5's
canonical 26 run one job per INTERFACE, so each job's prediction directory belongs to exactly one
labelled interface and must be scored for that interface only.

SCORING PROTOCOL (tts_abag/eval/eval_ost_foldbench.py -- not improvised): ONE OST call per pose, full
native assembly vs full model, `--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq`, OST's
own global chain mapping, then the labelled interface extracted BY NAME. If the pair is absent from
OST's mapping the pose is skipped -- never substituted with another interface.

Rescue rule, identical to every other arm: the pose with the highest interface `pair_iptm` has
DockQ >= 0.23.

Usage:
  .../python scripts/score_arm.py --arm dockqdonor \
      --jobs artifacts/dockq_arm_jobs.json --out artifacts/sweep_out/dockqdonor
Writes: artifacts/<arm>_scores.json  (+ OST cache under artifacts/<arm>_ost/)
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT, EXP, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=12)
A = ap.parse_args()

CACHE = f"{EXP}/artifacts/{A.arm}_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/{A.arm}_scores.json"
jobs = json.load(open(A.jobs if os.path.isabs(A.jobs) else f"{EXP}/{A.jobs}"))
OUT = A.out if os.path.isabs(A.out) else f"{EXP}/{A.out}"

free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))


def run_pose(a):
    """(pair_iptm, dockq) for this job's interface at one pose, or None."""
    job, s, k = a
    name = job["name"]
    cif = f"{OUT}/{job['run']}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    lab = job["interface"]
    o = f"{CACHE}/{lab}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{job['pdb']}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(f"{OUT}/{job['run']}/{name}/seed_{s}/predictions/"
                          f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):            # BY NAME, never dockq[0]
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                return (M[x][y], qs[i])
    return None


res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for job in jobs:
    lab = job["interface"]
    tasks = [(job, s, k) for s in SEEDS for k in SAMPLES]
    with ThreadPoolExecutor(A.workers) as ex:
        got = [g for g in ex.map(run_pose, tasks) if g]
    if not got:
        print(f"{lab:12} NO POSES SCORED", flush=True); continue
    sel = max(got, key=lambda x: x[0])
    ok = [p for p in got if p[1] >= THRESH]
    f = free.get(lab, {})
    res[lab] = {"pdb": job["pdb"], "arm": A.arm, "n": len(got), "n_acceptable": len(ok),
                "top_pair_iptm": round(sel[0], 3), "sel_dockq": round(sel[1], 3),
                "best_dockq": round(max(p[1] for p in got), 3), "rescued": sel[1] >= THRESH,
                "free_n_acceptable": f.get("free_n_acceptable"),
                "free_sel_dockq": f.get("free_sel_dockq"), "free_best_dockq": f.get("free_best_dockq")}
    v = res[lab]
    print(f"{lab:12} poses {v['n']:4d}  acceptable {v['free_n_acceptable']}->{v['n_acceptable']:3d}  "
          f"sel {v['free_sel_dockq']}->{v['sel_dockq']:.3f}  best {v['best_dockq']:.3f}"
          f"{'   RESCUED' if v['rescued'] else ''}", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)

n = sum(1 for v in res.values() if v["rescued"])
print(f"\n{A.arm}: RESCUED {n}/{len(res)} scored")
print(f"WROTE {OUTJ}")
