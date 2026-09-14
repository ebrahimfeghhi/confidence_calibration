#!/usr/bin/env python
"""Build a Protenix input json (sequences only) from a PDB biological-assembly mmCIF.

This is step 1 of the pipeline already used for the antibody-antigen set:
    <pdb>.json  --(runner/batch_inference.py --run msa_template)-->  <pdb>-update-msa.json
                                                               -->  <pdb>-final-updated.json
                --(runner/inference.py)--> predictions

The existing foldbench_abag/jsons/*.json were produced by an unknown route, so this script is
VALIDATED BY REPRODUCTION: run with --validate it rebuilds an antibody-antigen json from the same
assembly cif and diffs it against the known-good file. Only the schema those files exhibit is
emitted:
    proteinChain : {sequence, count}          (+ modifications where present in the originals)
    ligand       : {ligand: "CCD_<code>", count}
    name         : bare pdb id, no -assembly1 suffix
    one complex per file, wrapped in a list
Entities are grouped by identical sequence, count = number of copies in the assembly.
"""
import argparse, collections, json, os, sys

import gemmi

AA3 = gemmi.ResidueInfo


def entities_from_cif(path, min_len=4):
    """(protein sequences, ligand codes) for the assembly, one entry per CHAIN COPY.

    Uses the DEPOSITED ENTITY SEQUENCE (SEQRES / _entity_poly), not the modelled residues. The
    known-good antibody-antigen jsons contain the full construct including expression tags
    ("MHHHHHHHLGKLQY..."), which the coordinates do not resolve; and two copies of one entity have
    different resolved ranges, so grouping on observed residues split them into separate entities
    with count 1 instead of one entity with count 2. Both errors disappear when the entity sequence
    is used.
    """
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_hydrogens()
    ent_seq = {}
    for ent in st.entities:
        if ent.polymer_type in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            s = gemmi.one_letter_code(ent.full_sequence)
            if s:
                ent_seq[ent.name] = s.upper().replace("-", "")
    prot, lig = [], []
    for ch in st[0]:
        sub = ch.get_polymer()
        ename = st.get_entity_of(sub).name if sub and st.get_entity_of(sub) else None
        if ename in ent_seq and len(ent_seq[ename]) >= min_len:
            prot.append(ent_seq[ename])
            continue
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info and (info.is_water() or info.is_amino_acid()):
                continue
            lig.append(res.name)
    return prot, lig


def build(pdb, cif, min_len=4):
    prot, lig = entities_from_cif(cif, min_len)
    seqs = []
    for s, n in collections.Counter(prot).items():   # Counter keeps first-seen order (py3.7+)
        seqs.append({"proteinChain": {"sequence": s, "count": n}})
    for l, n in collections.Counter(lig).items():
        seqs.append({"ligand": {"ligand": f"CCD_{l}", "count": n}})
    return [{"sequences": seqs, "name": pdb}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--validate", default="", help="path to a known-good json to diff against")
    A = ap.parse_args()
    got = build(A.pdb, A.cif)
    if A.validate:
        want = json.load(open(A.validate))
        gs = [(list(e)[0], list(e.values())[0]) for e in got[0]["sequences"]]
        ws = [(list(e)[0], list(e.values())[0]) for e in want[0]["sequences"]]
        ok = True
        if len(gs) != len(ws):
            print(f"  MISMATCH entity count: got {len(gs)} want {len(ws)}"); ok = False
        for i, (g, w) in enumerate(zip(gs, ws)):
            if g[0] != w[0]:
                print(f"  entity {i}: kind {g[0]} vs {w[0]}"); ok = False; continue
            for k in ("sequence", "ligand", "count"):
                if w[1].get(k) != g[1].get(k):
                    print(f"  entity {i} [{g[0]}] {k}: got {str(g[1].get(k))[:60]} "
                          f"want {str(w[1].get(k))[:60]}"); ok = False
        print(("  MATCH" if ok else "  DIFFERS") + f"  ({A.pdb})")
        return 0 if ok else 1
    out = A.out or f"{A.pdb}.json"
    json.dump(got, open(out, "w"), indent=1)
    print(f"  wrote {out}: {len(got[0]['sequences'])} entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
