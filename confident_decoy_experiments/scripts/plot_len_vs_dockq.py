#!/usr/bin/env python
"""Does antigen size predict how well the model does?

x  antigen chain length, residues RESOLVED in the ground truth (artifacts/antigen_lengths.csv)
y  ipTM-selected DockQ pooled over seeds 101-150 (`sel50_dockq`) -- what the pipeline ships

Motivated by finding that 14% of FoldBench's ab-ag set has a peptide antigen (<50 residues), which
raised the question of whether the benchmark's headline numbers are carried by targets where
"finding the epitope" is close to trivial.

Usage: .../python scripts/plot_len_vs_dockq.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
PEP, PRO = "#b3452c", "#31688e"
CUT, T = 50, 0.23


def main():
    length = {r["interface"]: int(r["antigen_len"])
              for r in csv.DictReader(open(f"{EXP}/artifacts/antigen_lengths.csv"))}
    xs, ys = [], []
    for v in csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")):
        if v["interface"] in length:
            xs.append(length[v["interface"]])
            ys.append(float(v["sel50_dockq"]))
    x, y = np.array(xs), np.array(ys)
    pep = x < CUT

    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    ax.axhline(T, color="0.75", lw=1.2)
    ax.text(720, T + 0.02, "DockQ 0.23", fontsize=11, color="0.45", ha="right")
    ax.axvline(CUT, color="0.75", lw=1.2, ls="--")
    ax.scatter(x[pep], y[pep], s=42, c=PEP, alpha=0.8, lw=0,
               label=f"peptide antigen, <{CUT} res (n={int(pep.sum())}), "
                     f"median DockQ {np.median(y[pep]):.2f}")
    ax.scatter(x[~pep], y[~pep], s=42, c=PRO, alpha=0.75, lw=0,
               label=f"protein antigen (n={int((~pep).sum())}), "
                     f"median DockQ {np.median(y[~pep]):.2f}")
    # trend on the protein antigens only -- the peptides sit in a separate cluster and would
    # otherwise dominate the fit through leverage rather than through any real relationship
    m, b = np.polyfit(x[~pep], y[~pep], 1)
    gx = np.linspace(x[~pep].min(), x[~pep].max(), 50)
    r2 = np.corrcoef(x[~pep], y[~pep])[0, 1] ** 2
    ax.plot(gx, m * gx + b, color="0.3", lw=1.5,
            label=f"fit, protein antigens only: slope {m:+.5f}/residue, r$^2$ = {r2:.3f}")
    ax.set_xlabel("antigen chain length (residues resolved in the ground truth)", fontsize=14)
    ax.set_ylabel("DockQ of the ipTM-selected pose, seeds 101-150", fontsize=14)
    ax.set_xlim(0, 730)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/len_vs_dockq.png", dpi=150)

    print(f"n={len(x)}   overall r2 {np.corrcoef(x,y)[0,1]**2:.3f}   "
          f"protein-only r2 {r2:.3f}  slope {m:+.5f}/residue")
    print(f"  peptide  (<{CUT}): n={int(pep.sum())} median {np.median(y[pep]):.3f}  "
          f"mean {y[pep].mean():.3f}  acceptable {100*(y[pep]>=T).mean():.0f}%")
    print(f"  protein (>={CUT}): n={int((~pep).sum())} median {np.median(y[~pep]):.3f}  "
          f"mean {y[~pep].mean():.3f}  acceptable {100*(y[~pep]>=T).mean():.0f}%")
    print("wrote artifacts/len_vs_dockq.png")


if __name__ == "__main__":
    main()
