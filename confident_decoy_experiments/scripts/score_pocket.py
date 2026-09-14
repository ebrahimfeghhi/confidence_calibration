#!/usr/bin/env python
"""Score the pocket-constrained runs.

Layout differs from the other arms: artifacts/sweep_out/pocket/pocket_<pdb>_<ab><ag>/<name>/seed_N/
predictions/..., because each INTERFACE gets its own constrained input json (the pocket is specific to
one antibody-antigen pair), whereas other arms are keyed by complex alone.

Protocol copied from score_msaoff5.py: one OST compare-structures per pose, --dockq only, labelled
pair extracted BY NAME with the multi-copy "-2" suffix stripped.
"""
import csv, glob, json, os, re, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor

OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
CACHE = "artifacts/pocket_ost"
os.makedirs(CACHE, exist_ok=True)
NW = int(os.environ.get("PK_NWORK", "8"))


def base(c):
    return c.split("-")[0]


jobs = []
for d in sorted(glob.glob("artifacts/sweep_out/pocket/pocket_*")):
    tag = os.path.basename(d).replace("pocket_", "")
    for cif in glob.glob(f"{d}/*/seed_*/predictions/*_sample_*.cif"):
        m = re.search(r"seed_(\d+)/predictions/.*_sample_(\d+)\.cif$", cif)
        if m:
            jobs.append((tag, int(m.group(1)), int(m.group(2)), cif))


def run(a):
    tag, s, k, cif = a
    pdb = tag.split("_")[0]
    ab, ag = tag.split("_")[1][0], tag.split("_")[1][1]
    o = f"{CACHE}/{tag}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        try:
            subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", cif,
                            "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=900)
        except Exception:
            return None
    try:
        d = json.load(open(o))
    except Exception:
        return None
    conf = cif.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json")
    try:
        M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        M = None
    q, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if i < len(q) and {base(t[0]), base(t[1])} == {ab, ag}:
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            ip = M[x][y] if (M and 0 <= x < len(M) and 0 <= y < len(M)) else -1
            return (tag, ip, q[i])
    return None


print(f"scoring {len(jobs)} poses, {NW} workers", flush=True)
res = collections.defaultdict(list)
n = 0
with ThreadPoolExecutor(NW) as ex:
    for r in ex.map(run, jobs):
        n += 1
        if r:
            res[r[0]].append((r[1], r[2]))
        if n % 1000 == 0:
            print(f"  {n}", flush=True)
with open("artifacts/pocket_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["interface", "n", "sel_dockq", "best_dockq", "top_iptm"])
    for tag, v in sorted(res.items()):
        sel = max(v)
        w.writerow([tag, len(v), round(sel[1], 3), round(max(x[1] for x in v), 3), round(sel[0], 3)])
print(f"wrote {len(res)} interfaces -> artifacts/pocket_scores.csv", flush=True)
