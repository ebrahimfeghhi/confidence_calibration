#!/usr/bin/env python
"""Audit ALL FoldBench antibody-antigen target rows for the cross-copy label bug.

Found 2026-08-01 on 7z2m: the target list pairs antibody chain A with antigen J, but A belongs to one
symmetry copy of the complex and J to the other, so they touch at only 3 residues. Likely cause: a
representative chain is picked per entity independently (first alphabetically), without checking the
two representatives live in the same copy. That can only misfire when an assembly holds >1 copy.

For every row: count native contact residues across the labeled pair, and -- when the labeled pair
barely touches -- find the same-entity partner that DOES contact, which is the correct chain.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/foldbench_label_audit.py
Writes: artifacts/foldbench_label_audit.json
"""
import csv, json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import native_chains, contact_residues, GT, ARTIFACTS_DIR   # label_asym_id-correct helpers

CSV_IN = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
OUTJ = f"{ARTIFACTS_DIR}/foldbench_label_audit.json"
MIN_CONTACT = 10        # fewer residues than this is not a binding interface
import gemmi


def entity_of(pdb):
    """label_asym_id -> label_entity_id. Same entity = identical sequence = symmetry copy."""
    b = gemmi.cif.read(f"{GT}/{pdb}-assembly1.cif").sole_block()
    cols = [list(b.find_values("_atom_site." + c))
            for c in ("label_asym_id", "label_entity_id", "group_PDB")]
    m = {}
    for a, e, g in zip(*cols):
        if g == "ATOM":
            m.setdefault(a, e)
    return m


rows, bad, skipped = [], [], []
for r in csv.DictReader(open(CSV_IN)):
    pdb = r["pdb_id"].split("-")[0]
    c1, c2 = r["interface_chain_id_1"], r["interface_chain_id_2"]
    if not os.path.exists(f"{GT}/{pdb}-assembly1.cif"):
        skipped.append(f"{pdb}: no ground truth cif")
        continue
    try:
        n = contact_residues(pdb, c1, c2)
    except (KeyError, ValueError, FileNotFoundError) as e:
        skipped.append(f"{pdb} {c1}-{c2}: {type(e).__name__} {str(e)[:70]}")
        continue
    ent = entity_of(pdb)
    ncopies = collections.Counter(ent.values())
    rec = {"pdb": pdb, "chain1": c1, "chain2": c2, "contact_residues": n,
           "multi_copy": max(ncopies.values()) > 1 if ncopies else False}
    if n < MIN_CONTACT:
        # the labeled pair barely touches -- look for a same-entity partner of c2 that DOES contact c1
        alts = []
        for ch, e in ent.items():
            if ch in (c1, c2) or e != ent.get(c2):
                continue
            try:
                alts.append((contact_residues(pdb, c1, ch), ch))
            except Exception:
                pass
        rec["best_same_entity_alternative"] = (
            {"chain": max(alts)[1], "contact_residues": max(alts)[0]} if alts else None)
        bad.append(rec)
    rows.append(rec)

json.dump({"min_contact": MIN_CONTACT, "n_rows": len(rows), "n_suspect": len(bad),
           "suspect": bad, "skipped": skipped, "all_rows": rows}, open(OUTJ, "w"), indent=1)

print(f"audited {len(rows)} rows; skipped {len(skipped)}")
print(f"\nSUSPECT (labeled pair < {MIN_CONTACT} contact residues): {len(bad)}")
for b in bad:
    alt = b.get("best_same_entity_alternative")
    a = f"-> correct partner {b['chain1']}-{alt['chain']} ({alt['contact_residues']} res)" if alt else "-> no better partner found"
    print(f"  {b['pdb']} {b['chain1']}-{b['chain2']}: {b['contact_residues']} res "
          f"| multi-copy={b['multi_copy']} {a}")
if skipped:
    print(f"\nSKIPPED ({len(skipped)}):")
    for s in skipped[:20]:
        print("  " + s)
print(f"\nWROTE {OUTJ}")
