#!/usr/bin/env python
"""Re-tally every template run's rescue count from its per-pose score JSON.

WHY: the headline counts (agonly 8/15, abonly 5/15, native50 14/15) live in prose in
confident_decoy_README.md, not in code. Two things make them suspect:
  1. the antigen-label bug already moved agonly 8 -> 6 once;
  2. on 2026-08-01 a bug was found where reading dockq[0] instead of matching the interface by NAME
     silently scored the wrong interface for PDBs contributing two interfaces (7z2m, 8slb, 8yxi, 8che)
     -- several of which sit in the rescue sets.
Project rule: re-tally from the score JSON, never from memory.

Rescue rule (identical everywhere): the pose with the highest interface pair_iptm has DockQ >= 0.23.

BUG CHECK: for any PDB contributing 2+ interfaces, compare their pose lists. Identical lists means the
scorer read one interface twice -- exactly the failure found on 2026-08-01.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/retally_templates.py
Writes: artifacts/retally_templates.json
"""
import json, os, glob, collections

EXP = "/home/ebrahim/confident_decoy_experiments"
SC = f"{EXP}/artifacts/scores"
OUTJ = f"{EXP}/artifacts/retally_templates.json"
DOCUMENTED = {"agonly_scores": 6, "abonly_scores": 8, "native50_scores": 14,
              "oracle_scores": 8, "expb_scores": 2, "expc_scores": 2, "expd_scores": 0}
DECOYS = set(json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))["decoys"])   # confident decoys ONLY (per_interface also holds excluded 8ux6_DF)


def poses(entry):
    """Normalise the several on-disk shapes to [(pair_iptm, dockq)].
    Seen in artifacts/scores/: list of dicts (abonly/oracle/native50/topiptm/sposefv/topdockq),
    list of [iptm, dockq] pairs (agonly/expb/expc/expd/safety), and dict-keyed-by-seed (native_scores)."""
    if isinstance(entry, dict):                     # {seed: {...}} or {seed: dockq}
        vals = list(entry.values())
        if vals and isinstance(vals[0], dict):
            return [(v.get("pair_iptm", -1), v.get("dockq")) for v in vals if v.get("dockq") is not None]
        return []                                   # no ipTM available -> cannot apply the rescue rule
    if not isinstance(entry, list) or not entry: return []
    if isinstance(entry[0], dict):
        return [(e.get("pair_iptm", -1), e.get("dockq")) for e in entry if e.get("dockq") is not None]
    if isinstance(entry[0], (list, tuple)) and len(entry[0]) >= 2:
        return [(e[0], e[1]) for e in entry if e[1] is not None]
    return []


out = {}
for f in sorted(glob.glob(f"{SC}/*_scores.json")):
    name = os.path.basename(f)[:-5]
    try:
        d = json.load(open(f))
    except Exception as e:
        out[name] = {"error": str(e)}; continue
    if not isinstance(d, dict): out[name] = {"error": "unexpected format"}; continue
    per, hits = {}, []
    for ifc, entry in d.items():
        p = poses(entry)
        if not p: continue
        sel = max(p, key=lambda x: x[0])
        ok = [x for x in p if x[1] >= 0.23]
        r = sel[1] >= 0.23
        per[ifc] = {"n": len(p), "n_ok": len(ok), "top_iptm": round(sel[0], 3),
                    "sel_dockq": round(sel[1], 3), "rescued": r}
        if r: hits.append(ifc)
    # BUG CHECK: sibling interfaces from the same pdb with byte-identical pose lists
    bypdb = collections.defaultdict(list)
    for ifc in d:
        bypdb[ifc.split("_")[0]].append(ifc)
    dupes = []
    for pdb, ifcs in bypdb.items():
        if len(ifcs) < 2: continue
        for i in range(len(ifcs)):
            for j in range(i + 1, len(ifcs)):
                if poses(d[ifcs[i]]) == poses(d[ifcs[j]]) and poses(d[ifcs[i]]):
                    dupes.append(f"{ifcs[i]} == {ifcs[j]}")
    decoy_hits = sorted(h for h in hits if h in DECOYS)
    out[name] = {"n_interfaces": len(per), "rescued_total": len(hits),
                 "rescued_in_decoy_set": len(decoy_hits), "decoy_hits": decoy_hits,
                 "identical_sibling_interfaces": dupes, "per_interface": per}

json.dump(out, open(OUTJ, "w"), indent=1)

print(f"{'run':22} {'#ifaces':>8} {'rescued':>8} {'in decoy set':>13} {'documented':>11}  flag")
for name, v in out.items():
    if "error" in v: print(f"{name:22} ERROR {v['error']}"); continue
    doc = DOCUMENTED.get(name)
    flag = ""
    if doc is not None and v["rescued_in_decoy_set"] != doc:
        flag = f"  <== MISMATCH vs documented {doc}"
    if v["identical_sibling_interfaces"]:
        flag += f"  <== DUPLICATE INTERFACES: {v['identical_sibling_interfaces']}"
    print(f"{name:22} {v['n_interfaces']:8d} {v['rescued_total']:8d} {v['rescued_in_decoy_set']:13d} "
          f"{str(doc):>11}{flag}")

print("\n--- rescue sets within the 15-interface decoy set ---")
for name, v in out.items():
    if "error" in v: continue
    if name in ("agonly_scores", "abonly_scores", "native50_scores", "oracle_scores"):
        print(f"{name:22} {v['rescued_in_decoy_set']:2d}: {v['decoy_hits']}")
ag = set(out.get("agonly_scores", {}).get("decoy_hits", []))
ab = set(out.get("abonly_scores", {}).get("decoy_hits", []))
if ag and ab:
    print(f"\nabonly & agonly overlap: {sorted(ag & ab) or 'EMPTY (disjoint)'}")
    print(f"union {len(ag | ab)} / neither: {sorted(DECOYS - ag - ab)}")
print(f"\nWROTE {OUTJ}")
