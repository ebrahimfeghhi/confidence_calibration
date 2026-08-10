"""
Plot histogram of interface atom counts across all evaluated antibody-antigen interfaces.
Interface atoms = heavy atoms from either chain within 5 Å of any heavy atom on the partner chain.
"""
import csv
from collections import defaultdict
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import matplotlib.pyplot as plt
import numpy as np

GROUND_TRUTH = Path("/home/ebrahim/FoldBench/examples/ground_truths")
TARGETS_CSV  = Path("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")
OUT_PNG      = Path("results/interface_atom_counts.png")
CONTACT_DIST = 5.0  # Angstroms


def count_interface_atoms(native_cif: Path, chain1: str, chain2: str) -> int | None:
    """Count heavy atoms within CONTACT_DIST of the partner chain at the interface."""
    chain1 = chain1.split("-")[0]  # strip assembly suffix e.g. "B-2" → "B"
    chain2 = chain2.split("-")[0]

    try:
        cif_file = pdbx.CIFFile.read(native_cif)
        s = pdbx.get_structure(cif_file, model=1, use_author_fields=False)
        s = s[s.element != "H"]  # heavy atoms only

        c1 = s[s.chain_id == chain1]
        c2 = s[s.chain_id == chain2]
        if len(c1) == 0 or len(c2) == 0:
            return None

        dists = struc.distance(c1.coord[:, None, :], c2.coord[None, :, :])
        in_contact = dists < CONTACT_DIST
        n = int(in_contact.any(axis=1).sum()) + int(in_contact.any(axis=0).sum())
        return n
    except Exception as e:
        print(f"  ERROR {native_cif.name} {chain1}-{chain2}: {e}")
        return None


def main():
    interfaces = []
    with open(TARGETS_CSV) as f:
        for row in csv.DictReader(f):
            pdb_id = row["pdb_id"]
            chain1 = row["interface_chain_id_1"]
            chain2 = row["interface_chain_id_2"]
            native_cif = GROUND_TRUTH / f"{pdb_id}.cif"
            if native_cif.exists():
                interfaces.append((pdb_id, chain1, chain2, native_cif))

    print(f"Computing interface atom counts for {len(interfaces)} interfaces...")
    counts = []
    for pdb_id, chain1, chain2, native_cif in interfaces:
        n = count_interface_atoms(native_cif, chain1, chain2)
        if n is not None:
            counts.append(n)
            print(f"  {pdb_id} {chain1}|{chain2}: {n} atoms")
        else:
            print(f"  {pdb_id} {chain1}|{chain2}: skipped")

    counts = np.array(counts)
    print(f"\nTotal interfaces: {len(counts)}")
    print(f"Mean:   {counts.mean():.1f}")
    print(f"Median: {np.median(counts):.1f}")
    print(f"Min:    {counts.min()}")
    print(f"Max:    {counts.max()}")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(counts.mean(),   color="red",    linestyle="--", label=f"Mean = {counts.mean():.0f}")
    ax.axvline(np.median(counts), color="orange", linestyle="--", label=f"Median = {np.median(counts):.0f}")
    ax.set_xlabel(f"Interface atom count (heavy atoms within {CONTACT_DIST:.0f} Å)")
    ax.set_ylabel("Number of interfaces")
    ax.set_title(f"Interface size distribution ({len(counts)} Ab-Ag interfaces)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved to {OUT_PNG}")


if __name__ == "__main__":
    main()
