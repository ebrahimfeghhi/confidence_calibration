"""
Compute paired and unpaired MSA depths for all 113 foldbench Ab-Ag structures
by running Protenix's FeatureAssemblyLine directly from the input JSONs.

Outputs: foldbench_abag/msa_depths.csv
Columns: pdb_id, prot_paired_num_alignments, prot_unpaired_num_alignments
"""
import csv
import json
import string
from pathlib import Path

import numpy as np

from protenix.data.msa.msa_featurizer import FeatureAssemblyLine
from protenix.data.constants import PROTEIN_CHAIN, LIGAND_CHAIN_TYPES

JSON_DIR = Path("foldbench_abag/jsons")
OUT_CSV  = Path("foldbench_abag/msa_depths.csv")


def build_meta(bioassembly):
    """
    Construct the meta dict that FeatureAssemblyLine.assemble() expects.
    Only includes protein chains — ligands/RNA/DNA don't affect prot depth counts.
    """
    chain_labels = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]
    meta = {}
    curr_aid = 0

    for eid, info in enumerate(bioassembly):
        if "proteinChain" not in info:
            continue
        c = info["proteinChain"]
        seq, count = c["sequence"], c["count"]

        u_a3m, p_a3m = "", ""
        if c.get("unpairedMsaPath"):
            u_a3m = Path(c["unpairedMsaPath"]).read_text()
        if c.get("pairedMsaPath"):
            p_a3m = Path(c["pairedMsaPath"]).read_text()

        for c_idx in range(count):
            aid = curr_aid + c_idx
            meta[aid] = {
                "entity_id": eid,
                "chain_id": chain_labels[aid],
                "sequence": seq,
                "paired_msa": p_a3m,
                "unpaired_msa": u_a3m,
                "chain_entity_type": PROTEIN_CHAIN,
            }
        curr_aid += count

    return meta


def make_std_idxs(meta):
    """
    Build a trivial std_idxs: one position per residue per chain, in order.
    This is only used for indexing the MSA feature matrix in step 6 of assemble(),
    which doesn't affect the depth counts we care about.
    """
    total = sum(len(info["sequence"]) for info in meta.values())
    return np.arange(total, dtype=np.int32)


def main():
    json_files = sorted(JSON_DIR.glob("*-final-updated.json"))
    print(f"Found {len(json_files)} structures")

    rows = []
    for jf in json_files:
        pdb_id = jf.stem.replace("-final-updated", "")
        try:
            data = json.loads(jf.read_text())
            bioassembly = data[0]["sequences"]
            meta = build_meta(bioassembly)
            std_idxs = make_std_idxs(meta)
            feat = FeatureAssemblyLine().assemble(meta, std_idxs)
            rows.append({
                "pdb_id": pdb_id,
                "prot_paired_num_alignments": int(feat.prot_paired_num_alignments),
                "prot_unpaired_num_alignments": int(feat.prot_unpaired_num_alignments),
            })
            print(f"  {pdb_id}: paired={feat.prot_paired_num_alignments}  unpaired={feat.prot_unpaired_num_alignments}")
        except Exception as e:
            print(f"  ERROR {pdb_id}: {e}")
            rows.append({"pdb_id": pdb_id, "prot_paired_num_alignments": -1, "prot_unpaired_num_alignments": -1})

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "prot_paired_num_alignments", "prot_unpaired_num_alignments"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
