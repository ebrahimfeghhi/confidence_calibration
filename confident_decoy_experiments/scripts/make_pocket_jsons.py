#!/usr/bin/env python
"""Build Protenix input JSONs with a POCKET CONSTRAINT set to the TRUE epitope.

THE EXPERIMENT (oracle, not a method): our analysis says failures are wrong-surface -- 87% of failed
poses contact essentially none of the true epitope, and fold quality is constant across seeds. So the
question is whether the model CAN dock correctly when told where, or whether it cannot build the
complex at all. Constraining to the ground-truth epitope separates "can't find it" from "can't build
it". If a target still fails when handed the answer, no amount of steering or repulsion will save it.

Protenix supports this natively -- `pocket_embedder` and `contact_embedder` are ENABLED in the
protenix-v2 config, and the docs describe the pocket constraint as intended for exactly this case:
"guide the binding interface between a chain of interest (e.g. a ligand or an antibody) and specific
residues in another chain (e.g. epitopes)". Every run so far has logged `#pocket:0`, i.e. unused.

FORMAT (docs/infer_json_format.md):
  "constraint": {"pocket": {"binder_chain": {"entity": E, "copy": C},
                            "contact_residues": [{"entity": E, "copy": C, "position": P}, ...],
                            "max_distance": 6}}
`position` is 1-based within the entity's sequence.

CHAIN AND RESIDUE MAPPING COMES FROM OST, per the project rule: the ground-truth chain that
corresponds to each model chain is read from OST's `chain_mapping` (which is REFERENCE -> MODEL and
must be inverted), and epitope residues are mapped from GT numbering into sequence positions by
explicit alignment with a residue-identity assertion.

Usage:
  .../foldbench/bin/python scripts/make_pocket_jsons.py --interfaces 8a44_AC,8uky_AC [--out DIR]
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import GT_DIR, IFACE_CSV, _ost, align_map, chain_residues, poses_free
from iface_chains import first_pose_with
from epitope_rmsd import native_epitope

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SRC_JSON = "/home/ebrahim/Protenix/foldbench_abag/jsons"


def entity_copy_map(src):
    """model chain letter -> (entity_index_1based, copy_index_1based) for the source input json.

    Protenix input jsons list `sequences` as entities in order; chains are assigned A, B, C... across
    entities and their copies, which is how the rest of this project refers to them.
    """
    seqs = src["sequences"] if isinstance(src, dict) else src[0]["sequences"]
    out, ch = {}, 0
    for ei, ent in enumerate(seqs, start=1):
        body = list(ent.values())[0]
        n = int(body.get("count", 1))
        for ci in range(1, n + 1):
            out[chr(ord("A") + ch)] = (ei, ci, body.get("sequence", ""))
            ch += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfaces", required=True,
                    help="comma-separated, e.g. 8a44_AC (pdb_<ab><ag>)")
    ap.add_argument("--out", default=f"{EXP}/artifacts/pocket_jsons")
    ap.add_argument("--max-distance", type=float, default=6.0)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    _, io, _ = _ost()
    made, skipped = [], []
    for spec in A.interfaces.split(","):
      try:
          pdb, chains = spec.split("_")
          ab, ag = chains[0], chains[1]
          src_path = f"{SRC_JSON}/{pdb}-final-updated.json"
          if not os.path.exists(src_path):
              skipped.append((spec, "no source json")); continue
          src = json.load(open(src_path))
          base = src[0] if isinstance(src, list) else src
          emap = entity_copy_map(base)
          if ab not in emap or ag not in emap:
              skipped.append((spec, f"chain not in entity map {sorted(emap)}")); continue

          # `ab`/`ag` from the spec are REFERENCE chain labels (FoldBench convention), NOT model
          # chains -- see scripts/iface_chains.py. Mapping them as model chains built 8 of 31 pocket
          # constraints from the wrong interface.
          try:
              ch0, _, _ = first_pose_with(pdb, ab, ag)
          except KeyError as e:
              skipped.append((spec, f"interface not in any cached OST pose: {e}")); continue
          ab_gt, ag_gt = ch0["ref_ab"], ch0["ref_ag"]
          ref = io.LoadMMCIF(f"{GT_DIR}/{pdb}-assembly1.cif", fault_tolerant=True)
          epi_gt = native_epitope(ref, ab_gt, ag_gt)
          if len(epi_gt) < 3:
              skipped.append((spec, f"GT epitope only {len(epi_gt)} residues (GT {ab_gt},{ag_gt})"))
              continue
          if len(epi_gt) > 120:
              # a real epitope is ~10-50 residues; hundreds means the mapped chain pair is not the
              # antibody-antigen interface (e.g. two copies of the same entity in a homodimer)
              skipped.append((spec, f"implausible epitope of {len(epi_gt)} residues (GT {ab_gt},{ag_gt})"))
              continue

          # GT residue numbers -> 1-based positions in the ENTITY SEQUENCE used by the input json
          rres = chain_residues(ref, ag_gt)
          rseq = "".join(x[1] for x in rres)
          mseq = emap[ag][2]
          if not mseq or not rseq:
              # non-polymer entities (ligands) carry no "sequence"; nothing to map positions into
              skipped.append((spec, f"no sequence for chain {ag} (ligand entity?)")); continue
          amap = align_map(mseq, rseq)                        # json-seq index -> GT index
          inv = {j: i for i, j in amap.items()}
          gt_idx = {rres[j][0]: j for j in range(len(rres))}
          positions = []
          for num in sorted(epi_gt):
              j = gt_idx.get(num)
              if j is None or j not in inv:
                  continue
              i = inv[j]
              assert mseq[i] == rseq[j], f"{spec}: residue mismatch mapping GT {num}"
              positions.append(int(i) + 1)                    # 1-based; int() because
              # OST/numpy residue numbers come back as np.int64, which json cannot encode
          if len(positions) < 3:
              skipped.append((spec, f"only {len(positions)} epitope residues mapped into the sequence"))
              continue

          e_ab, c_ab, _ = emap[ab]
          e_ag, c_ag, _ = emap[ag]
          out = json.loads(json.dumps(src))                   # deep copy, preserve list/dict shape
          tgt = out[0] if isinstance(out, list) else out
          tgt["name"] = f"{pdb}_pocket_{ab}{ag}"
          tgt["constraint"] = {"pocket": {
              "binder_chain": {"entity": e_ab, "copy": c_ab},
              "contact_residues": [{"entity": e_ag, "copy": c_ag, "position": p} for p in positions],
              "max_distance": A.max_distance}}
          dst = f"{A.out}/{pdb}_{ab}{ag}_pocket.json"
          json.dump(out, open(dst, "w"), indent=1)
          made.append((spec, len(positions), f"{ab}(e{e_ab}c{c_ab}) <- {ag}(e{e_ag}c{c_ag})"))
          print(f"  {spec}: {len(positions)} epitope residues, binder {ab} e{e_ab}c{c_ab}, "
                f"pocket {ag} e{e_ag}c{c_ag} -> {os.path.basename(dst)}", flush=True)

      except AssertionError as e:
        skipped.append((spec, f"residue mapping failed: {e}"))
      except Exception as e:
        skipped.append((spec, f"{type(e).__name__}: {e}"))

    print(f"\n  wrote {len(made)} constrained jsons -> {A.out}")
    if skipped:
        print(f"  skipped {len(skipped)}:")
        for s, why in skipped:
            print(f"    {s}: {why}")


if __name__ == "__main__":
    main()
