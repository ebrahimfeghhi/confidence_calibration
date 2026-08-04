#!/usr/bin/env python
"""Would a DockQ-selected template donor differ from the ifacelDDT-selected one? (no GPU)

§4's template sweep picks the donor pose by FOLD quality (whole/interface lDDT or pLDDT). It never
picks by DockQ. The proposed arm is: take the pose that DOCKED correctly but which ipTM ranked low,
and template from its folds. Because templates carry only intra-chain geometry (multichain_mask), this
leaks none of that pose's docking arrangement -- only the backbone conformations it happened to adopt.
The motivating mechanism is CDR-H3 induced fit: the correctly-docked pose may hold H3 near its bound
conformation, which fold-accuracy scoring would not specifically reward.

But if the DockQ-best pose is usually ALSO the ifacelDDT-best pose, the new arm is redundant with an
arm we already ran, and there is nothing to learn. This decides that before spending any GPU.

Per templatable chain it compares two donors:
    lddt donor   argmax interface lDDT   (what arm `ifacelddt` used)
    dockq donor  argmax DockQ over that chain's labelled interface(s), among acceptable poses
and reports how often they coincide, plus what each donor gives up on the other's metric.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/dockq_vs_lddt_donor.py
Writes: artifacts/dockq_vs_lddt_donor.json
"""
import csv, json, os, sys, collections, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH

FOLDS = json.load(open(f"{EXP}/artifacts/fold_selection_table.json"))
CACHES = [f"{EXP}/artifacts/abag_all_ost", f"{EXP}/artifacts/decoy_set_ost"]
DECOYS = {k for k, v in json.load(open(f"{EXP}/artifacts/native_arm_scores.json")).items()
          if v["free_status"] == "confident_decoy"}

# pdb -> [(c1, c2)] from the FoldBench targets csv (label_asym_id, as everywhere else)
pairs = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    pairs[r["pdb_id"].split("-")[0]].append((r["interface_chain_id_1"], r["interface_chain_id_2"]))


def dockq_by_pose(pdb, c1, c2):
    """{(seed, sample): dockq} for one labelled interface, from whichever OST cache holds it."""
    out = {}
    for s in SEEDS:
        for k in SAMPLES:
            for cache in CACHES:
                o = f"{cache}/{pdb}_{s}_{k}.json"
                if not (os.path.exists(o) and os.path.getsize(o)):
                    continue
                try:
                    d = json.load(open(o))
                except Exception:
                    continue
                qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
                for i, t in enumerate(ifs):
                    if {t[0], t[1]} == {c1, c2} and i < len(qs):   # by NAME, never dockq[0]
                        out[(s, k)] = qs[i]
                        break
                if (s, k) in out:
                    break
    return out


rows, skipped = [], []
print(f"{'pdb':6} {'chain':6} {'set':6} {'lddt donor':>14} {'dockq donor':>14} {'same?':>6} "
      f"{'il(lddt)':>9} {'il(dockq)':>10} {'dq(lddt)':>9} {'dq(dockq)':>10}")
for pdb in sorted(FOLDS):
    if pdb not in pairs:
        skipped.append(f"{pdb}: no interfaces in targets csv"); continue
    # DockQ per interface for this pdb, once
    dq = {(c1, c2): dockq_by_pose(pdb, c1, c2) for c1, c2 in pairs[pdb]}
    for qidx, info in sorted(FOLDS[pdb].items()):
        nc = info["native_chain"]
        poses = info["poses"]                       # [seed, sample, global_lddt, iface_lddt, gp, ip]
        if not poses:
            continue
        # this chain's interfaces -> union of DockQ across them (a chain can sit in two)
        mine = [p for p in pairs[pdb] if nc in p]
        if not mine:
            skipped.append(f"{pdb} chain {nc}: in no labelled interface"); continue
        combined = {}
        for pr in mine:
            for key, v in dq[pr].items():
                combined[key] = max(combined.get(key, -1), v)     # best of its interfaces
        if not combined:
            skipped.append(f"{pdb} chain {nc}: no cached DockQ"); continue

        il = {(p[0], p[1]): p[3] for p in poses}
        lddt_donor = max(il, key=lambda x: il[x])
        acc = {k: v for k, v in combined.items() if v >= THRESH and k in il}
        if not acc:
            skipped.append(f"{pdb} chain {nc}: no acceptable pose to donate from"); continue
        dockq_donor = max(acc, key=lambda x: acc[x])

        lab = [f"{pdb}_{a}{b}" for a, b in mine]
        rows.append(dict(pdb=pdb, chain=nc, is_decoy=any(l in DECOYS for l in lab),
                         lddt_donor=list(lddt_donor), dockq_donor=list(dockq_donor),
                         same=lddt_donor == dockq_donor,
                         il_at_lddt=round(il[lddt_donor], 3),
                         il_at_dockq=round(il[dockq_donor], 3),
                         dq_at_lddt=round(combined.get(lddt_donor, float("nan")), 3),
                         dq_at_dockq=round(acc[dockq_donor], 3),
                         n_acceptable=len(acc)))
        r = rows[-1]
        print(f"{pdb:6} {nc:6} {'decoy' if r['is_decoy'] else 'good':6} "
              f"{str(lddt_donor):>14} {str(dockq_donor):>14} {'SAME' if r['same'] else 'diff':>6} "
              f"{r['il_at_lddt']:9.3f} {r['il_at_dockq']:10.3f} {r['dq_at_lddt']:9.3f} "
              f"{r['dq_at_dockq']:10.3f}", flush=True)

json.dump(rows, open(f"{EXP}/artifacts/dockq_vs_lddt_donor.json", "w"), indent=1)
print()
for nm, sel in (("decoy interfaces", True), ("good interfaces", False)):
    v = [r for r in rows if r["is_decoy"] == sel]
    if not v:
        continue
    same = sum(1 for r in v if r["same"])
    print(f"  {nm:18} {len(v):3d} chains   same donor {same:3d} ({same/len(v):.0%})   "
          f"median iface-lDDT given up by the DockQ donor "
          f"{statistics.median(r['il_at_lddt'] - r['il_at_dockq'] for r in v):+.3f}   "
          f"median DockQ given up by the lDDT donor "
          f"{statistics.median(r['dq_at_dockq'] - r['dq_at_lddt'] for r in v):+.3f}")
if skipped:
    print(f"\nSKIPPED {len(skipped)}:")
    for s in skipped[:20]:
        print("  " + s)
print(f"\nWROTE artifacts/dockq_vs_lddt_donor.json")
