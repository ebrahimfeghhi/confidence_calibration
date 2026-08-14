#!/usr/bin/env python
"""Pooled free-run selection: top interface-pair_iptm over the WHOLE free pool (seeds 0-2499).

WHY. score_freedeep.py reports only the NEW deep block (e.g. 11745 poses from seeds 151-2499) and
pools with earlier seeds only when given a --prior, which only 8che had. That makes the free arm ~6%
smaller than the 12500-pose templated arm it is compared against, and it can report the wrong number
outright: on 8aok_AB the deep block selects 0.103 while the older 0-150 block selects 0.112, so the
pooled answer is 0.112.

This reads every cached OST score for a complex's free poses -- the pre-existing 0-150 block (whatever
cache holds it) plus the deep block -- pairs each with its interface pair_iptm from the pose's own
confidence JSON, and applies the project's selection rule to the union.

Writes artifacts/free_pooled_<pdb>.json with the per-pose table so this is never recomputed.

Usage: .../python scripts/pool_free_selection.py
"""
import glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paths import ARTIFACTS_DIR

ART = str(ARTIFACTS_DIR)
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"

# pdb -> (interfaces as label:refchain1:refchain2, caches holding the 0-150 block)
CFG = {
    "8che": (["8che_AE:A:E", "8che_BE:B:E"], ["free_extra_ost"]),
    "9jbq": (["9jbq_AC:A:C"], ["free9jbq_0_150_ost", "extra_set_ost"]),
    "8av2": (["8av2_AC:A:C"], ["free8av2_0_150_ost", "decoy_set_ost"]),
    "8slb": (["8slb_AB:A:B", "8slb_AC:A:C"], ["free8slb_0_150_ost", "decoy_set_ost"]),
    "8aok": (["8aok_AB:A:B"], ["free8aok_0_150_ost", "decoy_set_ost"]),
}


def dq_from(path, ifaces):
    """{label: dockq} for one cached OST json, matched BY NAME (never dockq[0])."""
    try:
        d = json.load(open(path))
    except Exception:
        return {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    out = {}
    for lab, c1, c2 in ifaces:
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                out[lab] = (qs[i], t[2], t[3])       # dockq + the MODEL chain letters
                break
    return out


def seed_sample_from(name):
    """<...>_<seed>_<sample>.json -> (seed, sample)"""
    b = os.path.basename(name)[:-5].split("_")
    try:
        return int(b[-2]), int(b[-1])
    except ValueError:
        return None


for pdb, (ispecs, caches) in CFG.items():
    ifaces = [tuple(s.split(":")) for s in ispecs]
    files = []
    for c in caches:
        files += glob.glob(f"{ART}/{c}/*.json")
    # keep only files for THIS pdb/interface, and de-duplicate by (seed, sample)
    seen, keep = set(), []
    for f in files:
        b = os.path.basename(f)
        if not (b.startswith(pdb) or any(b.startswith(l) for l, _, _ in ifaces)):
            continue
        ss = seed_sample_from(f)
        if ss is None or ss in seen:
            continue
        seen.add(ss); keep.append((ss, f))
    deep = glob.glob(f"{ART}/freedeep{pdb}_ost/*.json")
    for f in deep:
        ss = seed_sample_from(f)
        if ss and ss not in seen:
            seen.add(ss); keep.append((ss, f))

    def one(item):
        (s, k), f = item
        got = dq_from(f, ifaces)
        if not got:
            return None
        cj = None
        for cand in (f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json",):
            if os.path.exists(cand):
                cj = cand
        if cj is None:   # deep-block poses live under the freedeep run's own output tree
            g = glob.glob(f"{ART}/sweep_out/freedeep{pdb}/freedeep{pdb}/*/seed_{s}/predictions/"
                          f"*_summary_confidence_sample_{k}.json")
            cj = g[0] if g else None
        if cj is None:
            return None
        try:
            M = json.load(open(cj))["chain_pair_iptm"]
        except Exception:
            return None
        rec = {"seed": s, "sample": k}
        for lab, (dq, m1, m2) in got.items():
            x, y = ord(m1) - 65, ord(m2) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                rec[lab] = [M[x][y], dq]
        return rec if len(rec) > 2 else None

    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(one, keep) if r]
    json.dump({"pdb": pdb, "n_poses": len(rows), "poses": rows},
              open(f"{ART}/free_pooled_{pdb}.json", "w"))
    print(f"=== {pdb} === pooled {len(rows)} free poses")
    for lab, _, _ in ifaces:
        v = [(r[lab][0], r[lab][1]) for r in rows if lab in r]
        if not v:
            print(f"   {lab}: no poses"); continue
        top = max(v, key=lambda x: x[0])
        a = np.array([x[1] for x in v])
        print(f"   {lab}: n={len(v)} | pooled top-ipTM {top[0]:.3f} -> DockQ {top[1]:.3f} | "
              f"oracle {a.max():.3f} | n>=0.6 {(a >= 0.6).sum()}")
