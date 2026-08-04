#!/usr/bin/env python
"""Matched free-vs-native comparison at 755 poses for 8ivx and 8oxw.

Both were native-template FAILURES at 250 poses, but with very different behaviour:
    8ivx_AC   4 -> 2 acceptable   (templating made it no better)
    8oxw_AB   2 -> 12 acceptable  (~6x more correct poses, yet selection still picked DockQ 0.004)
    8oxw_AC   2 -> 13 acceptable  (same)
Those counts are small enough that the direction is not established. Extending BOTH arms to seeds
0-150 (755 poses) asks two things:
  1. is the 8oxw coverage gain real, or noise at n=2?
  2. if it is real, does selection ever start picking a correct pose as more appear -- or does it stay
     pinned near 0.004 regardless, which would make 8oxw a genuine selection failure like 8fas rather
     than a coverage failure?

IMPORTANT: both arms are scored over the SAME seed range here. The earlier deep-sampling run deepened
only the free arm, so those numbers must not be compared against 250-pose native numbers.

Protocol unchanged (tts_abag/eval/eval_ost_foldbench.py): one OST call per pose, full native assembly
vs full model, labelled interface extracted by name.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/deep_native_compare.py
Writes: artifacts/deep_native_compare.json
"""
import csv, json, os, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT, EXP, PRED, SAMPLES, THRESH

PDBS = ["8ivx", "8oxw"]
SEEDS = range(0, 151)
NAT = f"{EXP}/artifacts/native_run_out/native50"
CACHE_FREE = f"{EXP}/artifacts/deep_sample_ost"          # reuse: free poses already scored there
CACHE_NAT = f"{EXP}/artifacts/deep_native_ost"
os.makedirs(CACHE_NAT, exist_ok=True)
OUTJ = f"{EXP}/artifacts/deep_native_compare.json"
NWORK = 16

pairs = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    pdb = r["pdb_id"].split("-")[0]
    if pdb in PDBS:
        pairs[pdb].append((r["interface_chain_id_1"], r["interface_chain_id_2"]))


def score(arm, pdb, prs, s, k):
    if arm == "free":
        cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
        conf = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
        o = f"{CACHE_FREE}/{pdb}_{s}_{k}.json"
    else:
        nm = f"{pdb}_native50"
        cif = f"{NAT}/{nm}/seed_{s}/predictions/{nm}_sample_{k}.cif"
        conf = f"{NAT}/{nm}/seed_{s}/predictions/{nm}_summary_confidence_sample_{k}.json"
        o = f"{CACHE_NAT}/{pdb}_{s}_{k}.json"
    if not os.path.exists(cif):
        return {}
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o)); M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        return {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    out = {}
    for c1, c2 in prs:
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                out[(c1, c2)] = (M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1, qs[i])
                break
    return out


res = {}
for pdb in PDBS:
    prs = pairs[pdb]
    for arm in ("free", "native"):
        jobs = [(arm, pdb, prs, s, k) for s in SEEDS for k in SAMPLES]
        with ThreadPoolExecutor(NWORK) as ex:
            got = list(ex.map(lambda a: score(*a), jobs))
        for c1, c2 in prs:
            ps = [g[(c1, c2)] for g in got if (c1, c2) in g]
            if not ps:
                continue
            sel = max(ps, key=lambda x: x[0])
            ok = [p for p in ps if p[1] >= THRESH]
            res.setdefault(f"{pdb}_{c1}{c2}", {})[arm] = {
                "n": len(ps), "n_acceptable": len(ok), "frac": round(len(ok) / len(ps), 4),
                "top_pair_iptm": round(sel[0], 3), "sel_dockq": round(sel[1], 3),
                "best_dockq": round(max(p[1] for p in ps), 3), "rescued": sel[1] >= THRESH}
        json.dump(res, open(OUTJ, "w"), indent=1)
        print(f"{pdb} {arm}: done", flush=True)

print(f"\n{'interface':11} {'arm':7} {'poses':>6} {'acceptable':>11} {'rate':>7} {'sel DockQ':>10} {'best':>7}")
for lab in sorted(res):
    for arm in ("free", "native"):
        v = res[lab].get(arm)
        if not v: continue
        print(f"{lab:11} {arm:7} {v['n']:6d} {v['n_acceptable']:11d} {v['frac']:7.1%} "
              f"{v['sel_dockq']:10.3f} {v['best_dockq']:7.3f}"
              f"{'  RESCUED' if v['rescued'] else ''}")
print(f"\nWROTE {OUTJ}")
