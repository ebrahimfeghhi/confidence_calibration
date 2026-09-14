#!/usr/bin/env python
"""When a pose moves to a different epitope, does the model also FOLD the chains differently?

Two possibilities that the epitope analysis cannot distinguish:
  (a) the monomer folds are essentially fixed and only the DOCKING SITE moves between poses, or
  (b) poses that dock badly are also folding the antibody or antigen badly, in which case epitope
      choice is downstream of a shared upstream failure rather than the thing to fix.

Per-chain lDDT answers it and costs nothing: PXMeter already writes one per chain per pose into
artifacts/pxm2025/pxm_json/<pdb>_<seed>_<sample>.json under "chain". lDDT is an INTRA-chain
distance-difference score, so a chain's value does not depend on where the other chain docked -- it is
a clean monomer-fold readout. (Sanity-checked here by reporting its correlation with DockQ: a strong
one would mean the assumption is wrong.)

The "chain" block is keyed by REFERENCE chain name (verified on 8vqf, whose keys include the
reference-only chain 'P' rather than its model counterpart 'E'), which is the same namespace as
artifacts/pxm2025/chain_roles.csv, so antibody and antigen are looked up directly.

TWO READOUTS, per interface, over its 250 poses:
  1. SD of ab-chain lDDT and ag-chain lDDT, against the SD of epitope Jaccard. If the folds sit near
     zero while the epitope swings, answer (a) follows immediately.
  2. correlation of each chain's lDDT with epitope Jaccard. ~0 means fold quality and docking site are
     independent; clearly positive would support (b).

FOLDBENCH CANNOT BE INCLUDED: its OST runs used --dockq only, so artifacts/abag_all_ost/*.json carry
no per-chain lDDT. PXM-2025 only.

CAVEAT: lDDT is fairly insensitive to loop-level rearrangement, so a CDR-H3 flip can leave it nearly
unchanged. A null here means "no large-scale refold", not "the antibody is literally identical";
scripts/cdr_h3_rmsd.py is the sharper follow-up.

EXACT SEED SET enforced (300000-300049 x 5 samples); interfaces missing any pose are dropped and
counted.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/fold_vs_epitope.py
Writes artifacts/pxm2025/fold_vs_epitope.png and prints the summary table.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
SEEDS = frozenset(range(300000, 300050))
N_SAMPLE = 5


FB_SEEDS = frozenset(range(101, 151))
# FoldBench-only: epitope/paratope RMSD are defined on the FIXED NATIVE residue set (not each pose's
# own contacts, which would score different patches per pose and be incomparable across seeds).
#   local  = superpose the patch Calpha onto their GT counterparts -> patch CONFORMATION only
#   global = superpose the whole chain, then measure the patch -> conformation PLUS placement
# `local` is the direct RMSD analogue of the per-chain lDDT test above.
# TWO epitope-RMSD tables exist and disagree on 13,057 of 50,575 shared poses with no script or commit
# explaining v2, so BOTH are carried and reported side by side rather than one being picked.
RMSD_SRC = [("epi_local", "epitope_rmsd.csv", "epi_local"),
            ("epi_global", "epitope_rmsd.csv", "epi_global"),
            ("epi_local_v2", "epitope_rmsd_v2.csv", "epi_local"),
            ("para_local", "paratope_rmsd.csv", "para_local"),
            ("para_global", "paratope_rmsd.csv", "para_global")]


def rmsd_mode():
    """FoldBench: does the epitope/paratope PATCH change shape between poses of one interface?"""
    key = lambda r: (r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))))
    epi, dq = {}, {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/epitope_overlap.csv")):
        if int(r["seed"]) in FB_SEEDS and r["jaccard"]:
            epi[(key(r), int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])
    for r in csv.DictReader(open(f"{EXP}/artifacts/dockq_per_pose_foldbench.csv")):
        if int(r["seed"]) in FB_SEEDS and r["dockq"]:
            dq[(key(r), int(r["seed"]), int(r["sample"]))] = float(r["dockq"])

    tabs = {}
    for name, fn, col in RMSD_SRC:
        d = {}
        for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
            if int(r["seed"]) in FB_SEEDS and r.get(col):
                try:
                    d[(key(r), int(r["seed"]), int(r["sample"]))] = float(r[col])
                except ValueError:
                    pass
        tabs[name] = d

    print(f"FoldBench, seeds 101-150 x 5 samples\n")
    print(f"{'quantity':<14}{'n iface':>9}{'mean A':>9}{'median SD A':>13}"
          f"{'r vs epi Jac':>14}{'r vs DockQ':>12}")
    out = {}
    for name, _, _ in RMSD_SRC:
        g = collections.defaultdict(list)
        for k, v in tabs[name].items():
            if k in epi and k in dq:
                g[k[0]].append((v, epi[k], dq[k]))
        mu, sd, re_, rd = [], [], [], []
        for iface, rows in g.items():
            if len(rows) != 250:
                continue
            a = np.array(rows)
            mu.append(a[:, 0].mean()); sd.append(a[:, 0].std())
            for dst, col in ((re_, 1), (rd, 2)):
                dst.append(np.corrcoef(a[:, 0], a[:, col])[0, 1]
                           if a[:, 0].std() > 1e-9 and a[:, col].std() > 1e-9 else np.nan)
        f = lambda x: np.array(x, float)
        re_a, rd_a = f(re_), f(rd)
        out[name] = (f(sd), rd_a)
        print(f"{name:<14}{len(mu):>9}{np.mean(mu):>9.2f}{np.median(f(sd)):>13.2f}"
              f"{np.nanmedian(re_a):>+14.3f}{np.nanmedian(rd_a):>+12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    cols = {"epi_local": "#b3452c", "epi_global": "#d08b6a", "epi_local_v2": "#7d2f1e",
            "para_local": "#31688e", "para_global": "#7aa7c7"}
    for name, (sd, rd) in out.items():
        axes[0].hist(np.clip(sd, 0, 12), bins=np.linspace(0, 12, 41), histtype="step",
                     lw=2.0, color=cols[name], label=f"{name}  (median {np.median(sd):.2f} A)")
        axes[1].hist(rd[~np.isnan(rd)], bins=np.linspace(-1, 1, 41), histtype="step",
                     lw=2.0, color=cols[name],
                     label=f"{name}  (median {np.nanmedian(rd):+.2f})")
    axes[0].set_xlabel("within-interface SD of the RMSD, across 250 poses (A)", fontsize=12)
    axes[0].set_title("How much does the patch geometry move?", fontsize=12)
    axes[1].set_xlabel("Pearson r vs DockQ, within interface", fontsize=12)
    axes[1].set_title("Does it track pose quality?", fontsize=12)
    axes[1].axvline(0, color="0.6", lw=1.0)
    for ax in axes:
        ax.set_ylabel("number of interfaces", fontsize=12)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/patch_rmsd_vs_quality.png", dpi=150)
    print("\nwrote artifacts/patch_rmsd_vs_quality.png")


def main():
    if "--rmsd" in __import__("sys").argv:
        return rmsd_mode()
    roles = {}
    for r in csv.DictReader(open(f"{P}/chain_roles.csv")):
        if r["status"].startswith("ok"):
            roles[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))))] = (
                r["ab_chain"], r["ag_chain"])

    epi, dq = {}, {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        if int(r["seed"]) in SEEDS:
            epi[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))),
                 int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if int(r["seed"]) in SEEDS and r["dockq"]:
            dq[(r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))),
                int(r["seed"]), int(r["sample"]))] = float(r["dockq"])

    # per-chain lDDT, from the pose jsons
    lddt = {}
    for f in glob.glob(f"{P}/pxm_json/*_*_*.json"):
        m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m or int(m.group(2)) not in SEEDS:
            continue
        try:
            ch = json.load(open(f)).get("chain") or {}
        except Exception:
            continue
        lddt[(m.group(1), int(m.group(2)), int(m.group(3)))] = {
            k: v.get("lddt") for k, v in ch.items()}

    g = collections.defaultdict(list)
    for k, j in epi.items():
        pdb, pair, s, smp = k
        if k not in dq or (pdb, pair) not in roles:
            continue
        ch = lddt.get((pdb, s, smp))
        if not ch:
            continue
        ab, ag = roles[(pdb, pair)]
        if ch.get(ab) is None or ch.get(ag) is None:
            continue
        g[(pdb, pair)].append((j, dq[k], ch[ab], ch[ag]))

    rows, dropped = [], 0
    for iface, v in g.items():
        if len(v) != len(SEEDS) * N_SAMPLE:
            dropped += 1
            continue
        a = np.array(v)
        j, d, lab, lag = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        cor = lambda x, y: (np.corrcoef(x, y)[0, 1]
                            if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
        rows.append(dict(iface=iface, sd_epi=j.std(), sd_dq=d.std(),
                         sd_ab=lab.std(), sd_ag=lag.std(),
                         r_ab_epi=cor(lab, j), r_ag_epi=cor(lag, j),
                         r_ab_dq=cor(lab, d), r_ag_dq=cor(lag, d),
                         m_ab=lab.mean(), m_ag=lag.mean()))
    F = lambda k: np.array([r[k] for r in rows], float)

    print(f"PXM-2025: {len(rows)} interfaces with the complete 250-pose set "
          f"({dropped} dropped)\n")
    print("1) HOW MUCH DOES EACH QUANTITY MOVE WITHIN AN INTERFACE?  (median SD over interfaces)")
    for k, lab in (("sd_epi", "epitope Jaccard"), ("sd_dq", "DockQ"),
                   ("sd_ab", "antibody-chain lDDT"), ("sd_ag", "antigen-chain lDDT")):
        v = F(k)
        print(f"   {lab:<24} median SD {np.median(v):.4f}   mean {v.mean():.4f}   "
              f"max {v.max():.4f}")
    print(f"\n   mean antibody lDDT {F('m_ab').mean():.3f}   "
          f"mean antigen lDDT {F('m_ag').mean():.3f}")

    print("\n2) WITHIN-INTERFACE CORRELATIONS (median over interfaces)")
    for k, lab in (("r_ab_epi", "antibody lDDT vs epitope Jaccard"),
                   ("r_ag_epi", "antigen  lDDT vs epitope Jaccard"),
                   ("r_ab_dq", "antibody lDDT vs DockQ  [assumption check]"),
                   ("r_ag_dq", "antigen  lDDT vs DockQ  [assumption check]")):
        v = F(k)
        v = v[~np.isnan(v)]
        print(f"   {lab:<44} median r {np.median(v):+.3f}   "
              f"|r|>0.3 in {100*(np.abs(v)>0.3).mean():.0f}%   n={len(v)}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    for k, c, lab in (("sd_epi", "#b3452c", "epitope Jaccard"),
                      ("sd_ab", "#31688e", "antibody-chain lDDT"),
                      ("sd_ag", "#3f7d4e", "antigen-chain lDDT")):
        ax.hist(F(k), bins=np.linspace(0, 0.35, 41), histtype="step", lw=2.0, color=c,
                label=f"{lab}  (median {np.median(F(k)):.3f})")
    ax.set_xlabel("SD within the interface, across its 250 poses", fontsize=12)
    ax.set_ylabel("number of interfaces", fontsize=12)
    ax.set_title("How much does each quantity move?", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(alpha=0.2)

    for ax, (k, c, lab) in zip(axes[1:], (
            ("r_ab_epi", "#31688e", "antibody-chain lDDT"),
            ("r_ag_epi", "#3f7d4e", "antigen-chain lDDT"))):
        v = F(k)
        v = v[~np.isnan(v)]
        ax.hist(v, bins=np.linspace(-1, 1, 41), color=c, alpha=0.55)
        ax.hist(v, bins=np.linspace(-1, 1, 41), histtype="step", lw=1.8, color=c)
        ax.axvline(0, color="0.6", lw=1.0)
        ax.axvline(np.median(v), color="0.2", lw=1.5, ls="--",
                   label=f"median {np.median(v):+.3f}")
        ax.set_title(f"{lab} vs epitope Jaccard\nwithin interface (n={len(v)})", fontsize=12)
        ax.set_xlabel("Pearson r across the interface's 250 poses", fontsize=12)
        ax.set_ylabel("number of interfaces", fontsize=12)
        ax.set_xlim(-1, 1)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        ax.grid(alpha=0.2)
    for ax in axes:
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(f"{P}/fold_vs_epitope.png", dpi=150)
    print(f"\nwrote {P}/fold_vs_epitope.png")


if __name__ == "__main__":
    main()
