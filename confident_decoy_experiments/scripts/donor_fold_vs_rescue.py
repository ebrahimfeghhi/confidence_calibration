#!/usr/bin/env python
"""Does the DONOR POSE'S FOLD QUALITY predict whether dockqdonor rescues?

Native templating rescues 17/25; dockqdonor 9/25, and almost a strict subset (8 both, 9 native-only,
1 dockqdonor-only). The obvious explanation: native templates are EXACT, while a dockqdonor template is
a model fold with residual error -- and docking is hypersensitive to sub-Angstrom template accuracy
(the self-templating work found two folds both at lDDT ~0.96 templating to DockQ 0.63 vs 0.11).

Prediction if that is right: the donor poses on interfaces dockqdonor FAILS should be measurably worse
folded -- especially at the binding site -- than the donors on interfaces it rescues. The donor was
chosen by DockQ, which says nothing about fold accuracy, so nothing forces its folds to be good.

Per interface this looks up the donor pose (seed, sample) actually used, and reports that pose's
per-chain fold quality against the native:
    global_lddt  whole-chain lDDT
    iface_lddt   lDDT over the binding-site residues
then groups interfaces by outcome and tests the separation.

CAVEAT: `iface_lddt` in fold_selection_table.json is UNION-based for a chain that sits in two labelled
interfaces (8che, 8fbw, 8oxw, 8q3j, 8slb, 8yxi), i.e. residues contacting EITHER partner. Exact
per-interface values would need a per-residue recompute. Flagged per row.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/donor_fold_vs_rescue.py
Writes: artifacts/donor_fold_vs_rescue.json
"""
import json, os, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

don = json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
dq = json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
nat = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
TBL = json.load(open(f"{EXP}/artifacts/fold_selection_table.json"))

MULTI = {"8che", "8fbw", "8oxw", "8q3j", "8slb", "8yxi"}      # union-based iface_lddt applies here
rows, missing = [], []

for lab in sorted(set(dq) & set(don)):
    d = don[lab]; pdb = d["pdb"]; s, k = d["seed"], d["sample"]
    if pdb not in TBL:
        missing.append(f"{lab}: {pdb} not in fold_selection_table"); continue
    # the donor's fold quality, per templated chain
    gl, il, chains = [], [], []
    for qi, nc in d["chains"].items():
        rec = TBL[pdb].get(qi.replace("q", ""))
        if rec is None:
            continue
        hit = next((p for p in rec["poses"] if p[0] == s and p[1] == k), None)
        if hit is None:
            continue
        if hit[2] is not None: gl.append(hit[2])
        if hit[3] is not None: il.append(hit[3])
        chains.append(nc)
    if not gl or not il:
        missing.append(f"{lab}: donor pose seed{s}/s{k} absent from fold table"); continue

    rows.append({
        "interface": lab, "pdb": pdb, "donor_seed": s, "donor_sample": k,
        "donor_dockq": d["donor_dockq"],
        "min_global_lddt": round(min(gl), 3), "mean_global_lddt": round(st.mean(gl), 3),
        "min_iface_lddt": round(min(il), 3), "mean_iface_lddt": round(st.mean(il), 3),
        "n_chains": len(gl), "union_iface_lddt": pdb in MULTI,
        "dockqdonor_rescued": dq[lab]["rescued"],
        "native_rescued": nat[lab]["rescued"],
        "dockqdonor_n_acceptable": dq[lab]["n_acceptable"],
        "native_n_acceptable": nat[lab]["n_acceptable"],
    })

for r in rows:
    r["group"] = ("both" if r["dockqdonor_rescued"] and r["native_rescued"] else
                  "native_only" if r["native_rescued"] else
                  "dockqdonor_only" if r["dockqdonor_rescued"] else "neither")

print(f"{'interface':11} {'group':16} {'donor DockQ':>11} {'min glob lDDT':>13} {'min iface lDDT':>14} {'':3}")
for r in sorted(rows, key=lambda x: (x["group"], -x["min_iface_lddt"])):
    print(f"{r['interface']:11} {r['group']:16} {r['donor_dockq']:11.3f} {r['min_global_lddt']:13.3f} "
          f"{r['min_iface_lddt']:14.3f} {'†' if r['union_iface_lddt'] else '':3}")

print("\ngroup medians (min across the templated chains):")
print(f"{'group':16} {'n':>3} {'donor DockQ':>11} {'global lDDT':>11} {'iface lDDT':>10}")
for g in ("both", "native_only", "dockqdonor_only", "neither"):
    v = [r for r in rows if r["group"] == g]
    if not v: continue
    print(f"{g:16} {len(v):3d} {st.median(r['donor_dockq'] for r in v):11.3f} "
          f"{st.median(r['min_global_lddt'] for r in v):11.3f} "
          f"{st.median(r['min_iface_lddt'] for r in v):10.3f}")

resc = [r for r in rows if r["dockqdonor_rescued"]]
fail = [r for r in rows if not r["dockqdonor_rescued"]]
print(f"\ndockqdonor RESCUED (n={len(resc)}) vs FAILED (n={len(fail)}):")
for metric in ("min_global_lddt", "min_iface_lddt", "mean_iface_lddt", "donor_dockq"):
    a = st.median(r[metric] for r in resc); b = st.median(r[metric] for r in fail)
    print(f"  {metric:18} rescued {a:.3f}   failed {b:.3f}   diff {a-b:+.3f}")

# rank correlation between donor fold quality and rescue, without scipy
def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order): r[i] = pos + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0

y = [1.0 if r["dockqdonor_rescued"] else 0.0 for r in rows]
for metric in ("min_global_lddt", "min_iface_lddt", "donor_dockq"):
    print(f"  spearman({metric}, rescued) = {spearman([r[metric] for r in rows], y):+.3f}")

json.dump(rows, open(f"{EXP}/artifacts/donor_fold_vs_rescue.json", "w"), indent=1)
if missing:
    print(f"\nMISSING {len(missing)}:")
    for m in missing: print("  " + m)
print(f"\n† iface lDDT is union-based for multi-interface pdbs")
print(f"WROTE artifacts/donor_fold_vs_rescue.json")
