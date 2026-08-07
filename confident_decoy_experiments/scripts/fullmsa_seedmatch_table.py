#!/usr/bin/env python
"""Seed-matched (101-150, 250 poses each) comparison: ORIGINAL (MSA-subsampled) vs FULL-MSA.

ORIGINAL per-pose scores come from the canonical CSV, EXCEPT the 4 'regen' pdbs
(8fbw/8q3j/8ux6/8yxi) whose canonical rows are at seeds 0-49 -- for those we use
artifacts/free_101_150_scores.json (produced by score_free_101_150.py).
FULL-MSA per-pose scores are re-read from the cached OST outputs in artifacts/fullmsa_n50/.

Columns (see README 2e for prose definitions):
  n_ok    = # of the 250 poses with DockQ >= 0.23 (coverage)
  top_ipt = pair_iptm of the #1-ranked pose (what the model would hand you)
  sel_dq  = DockQ of that #1 pose  -> rescued iff >= 0.23  (THE outcome)
  best_dq = best DockQ anywhere in the pool (ceiling under perfect selection)
  gap     = top_ipt - pair_iptm(best acceptable pose); 0.0 => rescued by construction
Writes artifacts/fullmsa_seedmatch_table.json
"""
import csv, json, os

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
CSV = "/home/ebrahim/tts_abag/results/csv_files/dockq_ost_protenix_full.csv"
CACHE = f"{EXP}/artifacts/fullmsa_n50"
OUT = f"{EXP}/artifacts/fullmsa_out"
meta = json.load(open(f"{EXP}/artifacts/meta.json"))
regen = json.load(open(f"{EXP}/artifacts/free_101_150_scores.json"))
SEEDS = range(101, 151)


def summarize(poses):
    sel = max(poses, key=lambda x: x[0])
    ok = [p for p in poses if p[1] >= 0.23]
    return {"n": len(poses), "n_ok": len(ok), "top_iptm": round(sel[0], 3),
            "sel_dq": round(sel[1], 3), "best_dq": round(max(p[1] for p in poses), 3),
            "gap": round(sel[0] - max(p[0] for p in ok), 3) if ok else None,
            "rescued": sel[1] >= 0.23}


# ---- ORIGINAL ----
canon = {}
for r in csv.DictReader(open(CSV)):
    pdb = r["assembly_id"].split("-")[0]
    if not (101 <= int(r["seed"]) <= 150):
        continue
    canon.setdefault((pdb, frozenset((r["chain1"], r["chain2"]))), []).append(
        (float(r["pair_iptm"]), float(r["dockq"])))

# ---- FULL-MSA (re-read cached OST) ----
def fullmsa_poses(ifc, m):
    p, c1, c2 = m["pdb"], m["c1"], m["c2"]
    poses = []
    for s in SEEDS:
        for k in range(5):
            o = f"{CACHE}/dq_{ifc}_{s}_{k}.json"
            if not os.path.exists(o):
                continue
            try:
                d = json.load(open(o))
            except Exception:
                continue
            dq = mc = None
            for i, t in enumerate(d.get("dockq_interfaces") or []):
                if {t[0], t[1]} == {c1, c2}:
                    q = d.get("dockq") or []
                    dq = q[i] if i < len(q) else None
                    mc = (t[2], t[3])
                    break
            if dq is None:
                continue
            pi = -1
            try:
                M = json.load(open(f"{OUT}/{p}/seed_{s}/predictions/"
                                  f"{p}_summary_confidence_sample_{k}.json")).get("chain_pair_iptm")
                a, b = ord(mc[0]) - 65, ord(mc[1]) - 65
                if M and 0 <= a < len(M) and 0 <= b < len(M):
                    pi = M[a][b]
            except Exception:
                pass
            poses.append((pi, dq))
    return poses


rows = []
for ifc, m in sorted(meta.items()):
    key = (m["pdb"], frozenset((m["c1"], m["c2"])))
    orig = [tuple(x) for x in regen[ifc]] if ifc in regen else canon.get(key)
    full = fullmsa_poses(ifc, m)
    if not orig or not full:
        print(f"SKIP {ifc}: orig={len(orig or [])} full={len(full)}")
        continue
    o, f = summarize(orig), summarize(full)
    dgap = round(f["gap"] - o["gap"], 3) if (o["gap"] is not None and f["gap"] is not None) else None
    rows.append({"interface": ifc, "orig": o, "full": f, "dgap": dgap})

json.dump(rows, open(f"{EXP}/artifacts/fullmsa_seedmatch_table.json", "w"), indent=1)

hdr = (f"{'interface':10} | {'n_ok':>4} {'top_ipt':>7} {'sel_dq':>6} {'best_dq':>7} {'gap':>6} "
       f"| {'n_ok':>4} {'top_ipt':>7} {'sel_dq':>6} {'best_dq':>7} {'gap':>6} | {'dgap':>6}")
print(f"{'':10} | {'ORIGINAL (MSA-subsampled)':^39} | {'FULL MSA':^39} |")
print(hdr)
for r in rows:
    o, f = r["orig"], r["full"]
    g = lambda v: f"{v:+.3f}" if v is not None else "  n/a"
    print(f"{r['interface']:10} | {o['n_ok']:4d} {o['top_iptm']:7.3f} {o['sel_dq']:6.3f} "
          f"{o['best_dq']:7.3f} {g(o['gap']):>6} | {f['n_ok']:4d} {f['top_iptm']:7.3f} "
          f"{f['sel_dq']:6.3f} {f['best_dq']:7.3f} {g(f['gap']):>6} | {g(r['dgap']):>6}"
          + ("  <-- rescue changes" if o["rescued"] != f["rescued"] else ""))

ro = [r["interface"] for r in rows if r["orig"]["rescued"]]
rf = [r["interface"] for r in rows if r["full"]["rescued"]]
print(f"\nRESCUED  ORIGINAL {len(ro)}/{len(rows)} {ro}")
print(f"RESCUED  FULL-MSA {len(rf)}/{len(rows)} {rf}")
d = [r["dgap"] for r in rows if r["dgap"] is not None]
d.sort()
print(f"dgap: median {d[len(d)//2]:+.3f}; gap SMALLER (better) in {sum(1 for x in d if x < 0)}/{len(d)}")
