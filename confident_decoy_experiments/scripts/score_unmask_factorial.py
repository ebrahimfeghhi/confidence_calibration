#!/usr/bin/env python
"""Score the 6-cell unmasked-template factorial on 8che, against the masked baseline.

All six cells share donor 137_3 (seed 137 sample 3, own DockQ 0.529) and the SAME 10 seeds as
allpose8che10's 137_3 arm, so every number here is seed- and template-matched to that masked run.
Cell C (subsampled MSA, dropout on) is the ONLY cell whose condition matches the masked baseline, so
C-vs-baseline is the clean mask contrast; the other cells additionally vary MSA depth or dropout.

SCORING PROTOCOL is the project standard (tts_abag/eval/eval_ost_foldbench.py): one OST call per
pose, full native assembly vs full model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4
--dockq, interface extracted BY NAME (never dockq[0]).

Reports per cell, per interface: best pose, what pooled ipTM selects, and how many of the 10 seeds
produced a near-native pose -- the statistic that distinguished 137_3 (6/10) in the masked run.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_unmask_factorial.py [--workers 16]
Writes: artifacts/unmask_factorial_scores.json
"""
import argparse, json, os, glob, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, OST, ENV, GT

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

ART = f"{EXP}/artifacts"
CACHE = f"{ART}/unmaskfac_ost"; os.makedirs(CACHE, exist_ok=True)
IFACES = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
SEEDS = [137, 108, 130, 250, 332, 832, 1036, 1040, 1074, 1076]
CELLS = {"A": ("none", "on", "unmasked"), "B": ("none", "off", "unmasked"),
         "C": ("subsampled", "on", "unmasked"), "D": ("subsampled", "off", "unmasked"),
         "E": ("full", "on", "unmasked"), "F": ("full", "off", "unmasked"),
         "G": ("none", "on", "MASKED"), "H": ("none", "off", "MASKED")}


def pose(a):
    cell, s, k, cif, conf = a
    o = f"{CACHE}/{cell}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/8che-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o)); M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):        # BY NAME
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = (M[x][y], qs[i])
                break
    return (cell, s, got) if got else None


tasks = []
for cell in CELLS:
    for s in SEEDS:
        for k in range(5):
            g = glob.glob(f"{ART}/sweep_out/unmaskfac_{cell}/*/seed_{s}/predictions/*_sample_{k}.cif")
            if not g:
                continue
            cif = g[0]
            conf = cif.replace(f"_sample_{k}.cif", f"_summary_confidence_sample_{k}.json")
            if os.path.exists(conf):
                tasks.append((cell, s, k, cif, conf))
print(f"{len(tasks)} poses to score across {len(CELLS)} cells")

with ThreadPoolExecutor(A.workers) as ex:
    res = [r for r in ex.map(pose, tasks) if r]

by = {}
for cell, s, g in res:
    by.setdefault(cell, {}).setdefault(s, []).append(g)

out = {}
for cell, (msa, drop, mask) in CELLS.items():
    sd = by.get(cell, {})
    rec = {"msa": msa, "dropout": drop, "mask": mask, "n_seeds": len(sd),
           "n_poses": sum(len(v) for v in sd.values())}
    for lab in IFACES:
        allp = [p[lab] for v in sd.values() for p in v if lab in p]
        if not allp:
            continue
        seedbest = {s: max(p[lab][1] for p in v if lab in p) for s, v in sd.items()}
        rec[lab] = {
            "best": round(max(x[1] for x in allp), 3),
            "sel_dockq": round(max(allp, key=lambda x: x[0])[1], 3),
            "sel_iptm": round(max(x[0] for x in allp), 3),
            "seeds_above_085": sum(1 for b in seedbest.values() if b > 0.85),
            "seeds_above_060": sum(1 for b in seedbest.values() if b > 0.6),
            "median_seed_best": round(sorted(seedbest.values())[len(seedbest) // 2], 3),
        }
    out[cell] = rec

hdr = f"{'cell':>4} {'mask':>9} {'MSA':>11} {'drop':>5} {'poses':>6} | "
for lab in IFACES:
    hdr += f"{lab + ' best':>13} {'ipTM pick':>10} {'>0.85':>6} {'>0.6':>5} | "
print("\n" + hdr)
for cell in "ABCDEFGH":
    r = out[cell]
    line = f"{cell:>4} {r['mask']:>9} {r['msa']:>11} {r['dropout']:>5} {r['n_poses']:>6} | "
    for lab in IFACES:
        v = r.get(lab)
        line += (f"{v['best']:>13.3f} {v['sel_dockq']:>10.3f} {v['seeds_above_085']:>4}/10 "
                 f"{v['seeds_above_060']:>3}/10 | ") if v else f"{'--':>13} {'--':>10} {'--':>6} {'--':>5} | "
    print(line)

print("\nMASKED baseline (allpose8che10, donor 137_3, same 10 seeds):")
b = json.load(open(f"{ART}/allpose8che10_scores.json"))
br = [r for r in b["rows"] if r["name"] == "8che_allpose_137_3"][0]
for lab in IFACES:
    v = br[lab]
    n85 = sum(1 for x in v["per_seed"].values() if x["max"] > 0.85)
    n60 = sum(1 for x in v["per_seed"].values() if x["max"] > 0.6)
    print(f"  {lab}: best {v['out_max']:.3f}  ipTM pick {v['out_sel']:.3f}  "
          f">0.85 {n85}/10  >0.6 {n60}/10")
    out.setdefault("masked_baseline", {})[lab] = {
        "best": v["out_max"], "sel_dockq": v["out_sel"],
        "seeds_above_085": n85, "seeds_above_060": n60}

json.dump(out, open(f"{ART}/unmask_factorial_scores.json", "w"), indent=2)
print(f"\nWROTE {ART}/unmask_factorial_scores.json")
