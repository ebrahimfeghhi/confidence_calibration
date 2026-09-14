#!/usr/bin/env python
"""How big are FoldBench's antigens?

Motivated by finding two SHORT PEPTIDE antigens among a handful of anomalies (8fas is an 11-residue
(NANP)n malaria repeat; 8che_AE is a 15-mer), which raised the question of how common they are and
whether DockQ on them is even a stable measurement.

Length = residues RESOLVED in the ground-truth assembly for the antigen chain, not the construct
sequence -- that is what DockQ and the epitope sets are actually computed on. Antigen chain names
come from FoldBench's `antigen_chain` column, which is a REFERENCE chain label (treating it as a
model chain is the bug that put 72 of 161 interfaces on the wrong chains).

Usage: .../foldbench/bin/python scripts/plot_antigen_lengths.py
"""
import csv
import os
import sys

import numpy as np

# The OST env has no matplotlib and the plotting env has no OST, so this runs in two passes:
#   --compute  (foldbench python, has OST)  -> artifacts/antigen_lengths.csv
#   default    (Protenix venv, has matplotlib) -> reads that csv and plots
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
TGT = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
COL, COL2 = "#b3452c", "#31688e"


CSV = f"/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/antigen_lengths.csv"


def compute():
    from cdr_h3_rmsd import GT_DIR, _ost, chain_residues
    _, io, _ = _ost()
    scored = {v["interface"] for v in
              csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv"))}
    cache, rows = {}, []
    for r in csv.DictReader(open(TGT)):
        pdb = r["pdb_id"].split("-")[0]
        ag = r["antigen_chain"]
        f = f"{GT_DIR}/{pdb}-assembly1.cif"
        if not os.path.exists(f):
            continue
        if pdb not in cache:
            cache[pdb] = io.LoadMMCIF(f, fault_tolerant=True)
        try:
            n = len(chain_residues(cache[pdb], ag))
        except Exception:
            continue
        if n == 0:
            continue
        rows.append((f'{pdb}_{r["antibody_chain"]}{ag}', n))
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["interface", "antigen_len", "scored"])
        for i, n in rows:
            w.writerow([i, n, int(i in scored)])
    print(f"wrote {CSV}: {len(rows)} interfaces")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, scored = [], set()
    for r in csv.DictReader(open(CSV)):
        rows.append((r["interface"], int(r["antigen_len"])))
        if r["scored"] == "1":
            scored.add(r["interface"])

    L = np.array([n for _, n in rows])
    Ls = np.array([n for i, n in rows if i in scored])
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    bins = np.arange(0, 740, 20)          # linear 20-residue bins
    ax.hist(L, bins=bins, color=COL, alpha=0.85, label=f"all FoldBench ab-ag interfaces (n={len(L)})")
    ax.hist(Ls, bins=bins, color=COL2, alpha=0.6,
            label=f"the subset we scored (n={len(Ls)})")
    ax.axvline(50, color="0.3", lw=1.6, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.text(58, ax.get_ylim()[1] * 0.62, "50 residues", fontsize=12, color="0.3", va="top")
    ax.set_xlim(0, 730)
    ax.set_xticks(range(0, 701, 100))
    ax.set_xlabel("antigen chain length (residues resolved in the ground truth)", fontsize=14)
    ax.set_ylabel("number of interfaces", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/antigen_lengths.png", dpi=150)

    print(f"n={len(L)} interfaces over {len({i.rsplit(chr(95),1)[0] for i,_ in rows})} complexes")
    print(f"  median {int(np.median(L))}   mean {L.mean():.0f}   "
          f"min {L.min()}   max {L.max()}")
    for t in (20, 30, 50, 100):
        print(f"  antigen shorter than {t:>3} residues: {int((L < t).sum()):>3}  "
              f"({100*(L<t).mean():>4.1f}%)")
    print("\nshortest 12:")
    for i, n in sorted(rows, key=lambda x: x[1])[:12]:
        print(f"   {i:<12} {n:>4} residues{'   [scored]' if i in scored else ''}")
    print("\nwrote artifacts/antigen_lengths.png")


if __name__ == "__main__":
    if "--compute" in sys.argv:
        compute()
    else:
        main()
