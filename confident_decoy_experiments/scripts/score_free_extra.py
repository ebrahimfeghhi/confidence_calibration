#!/usr/bin/env python
"""Score 8che's ALREADY-EXISTING free seeds 0-100 (505 poses) that no analysis has used.

Every free-run number in this project for 8che came from seeds 101-150 (250 poses) -- but seeds 0-150
exist and are complete, so 505 poses have been sitting unscored. Before spending an overnight run
asking "can free sampling reach DockQ 0.6", it is worth checking what 755 poses already show.

Scores both interfaces, protocol identical to the rest of the project (one OST call per pose, full
native vs full model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq, interface
matched BY NAME), and reports the 0-100 block and the canonical 101-150 block SEPARATELY as well as
pooled -- they were generated in different batches, so pooling them silently would hide any batch
difference rather than expose it.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_free_extra.py [--workers 16]
Writes: artifacts/free_extra_8che.json
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, OST, ENV, GT, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()
PDB = "8che"
IFACES = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
CACHE = f"{EXP}/artifacts/free_extra_ost"; os.makedirs(CACHE, exist_ok=True)


def pose(a):
    s, k = a
    cif = f"{PRED}/{PDB}/seed_{s}/predictions/{PDB}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{PDB}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{PDB}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(f"{PRED}/{PDB}/seed_{s}/predictions/"
                           f"{PDB}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
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


tasks = [(s, k) for s in range(0, 151) for k in SAMPLES]
with ThreadPoolExecutor(A.workers) as ex:
    res = [r for r in ex.map(pose, tasks) if r]
print(f"scored {len(res)} poses of {len(tasks)} attempted\n")

out = {}
for lab in IFACES:
    v = [r[lab] for r in res if lab in r]
    blocks = {"seeds 0-100 (never analysed)": [x for x in v if x[2] <= 100],
              "seeds 101-150 (canonical)": [x for x in v if x[2] > 100],
              "POOLED 0-150": v}
    print(f"=== {lab} ===")
    print(f"  {'block':30} {'poses':>6} {'best DockQ':>11} {'>=0.23':>8} {'>=0.6':>7} {'>=0.85':>7} "
          f"{'top-ipTM DockQ':>15}")
    rec = {}
    for nm, b in blocks.items():
        if not b:
            continue
        d = np.array([x[1] for x in b])
        sel = max(b, key=lambda x: x[0])
        rec[nm] = {"n": len(b), "best": round(float(d.max()), 3),
                   "n_acc": int((d >= THRESH).sum()), "n_06": int((d >= 0.6).sum()),
                   "n_085": int((d >= 0.85).sum()),
                   "sel_dockq": round(sel[1], 3), "sel_iptm": round(sel[0], 3),
                   "sel_seed": sel[2], "sel_sample": sel[3]}
        r = rec[nm]
        print(f"  {nm:30} {r['n']:6d} {r['best']:11.3f} {r['n_acc']:8d} {r['n_06']:7d} "
              f"{r['n_085']:7d} {r['sel_dockq']:15.3f}")
    out[lab] = rec
    print()
json.dump(out, open(f"{EXP}/artifacts/free_extra_8che.json", "w"), indent=1)
print(f"WROTE {EXP}/artifacts/free_extra_8che.json")
