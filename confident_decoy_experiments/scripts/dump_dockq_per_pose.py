#!/usr/bin/env python
"""Flatten FoldBench's per-pose ab-ag DockQ into one CSV.

WHY. PXM-2025 already has a flat per-pose table (artifacts/pxm2025/dockq_pxm2025.csv) but FoldBench's
DockQ only exists inside artifacts/abag_all_ost/<pdb>_<seed>_<sample>.json, one file per pose, read
through scripts/iface_chains.resolve(). Any analysis of the DISTRIBUTION of DockQ across poses (rather
than of a single selected pose) has to walk all of them, so it is worth doing once.

NOTE there is an artifacts/dockq_sd_data.npz carrying per-interface mean/SD for 151 ab-ag and 212 pp
interfaces, but no script in the repo produces it and git has no record of one, so its pose set and
filters are unknown. It is deliberately NOT used; this script rebuilds from the OST jsons instead.

CHAINS come from scripts/iface_chains.resolve(), which reads BOTH sides of the correspondence out of
the same `dockq_interfaces` row DockQ was computed with. FoldBench's antibody_chain/antigen_chain
columns are REFERENCE labels, and treating them as model chains is the bug that put 72 of 161
interfaces on the wrong chains. The (pdb, ab, ag, seed, sample) keys are taken from
artifacts/epitope_overlap.csv so this table covers exactly the poses the epitope analysis covers.

Seed window is SEED_LO..SEED_HI from scripts/selection_protocol.py (101-150), 5 samples each = 250
poses per interface, matching the selection protocol.

Usage: /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/dump_dockq_per_pose.py [--jobs 24]
Outputs artifacts/dockq_per_pose_foldbench.csv: pdb,ab_chain,ag_chain,seed,sample,dockq,fnat,irmsd,lrmsd
"""
import argparse
import collections
import csv
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iface_chains import resolve  # noqa: E402
from selection_protocol import chain_idx  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SEED_LO, SEED_HI = 101, 150
OUT = f"{EXP}/artifacts/dockq_per_pose_foldbench.csv"
# pair_iptm is added so the SELECTOR can be evaluated per pose, not just at the selected
# pose: the within-interface ipTM-vs-DockQ correlation is the direct measure of whether
# confidence can rank, independent of any selection protocol.
COLS = ["pdb", "ab_chain", "ag_chain", "seed", "sample", "dockq", "fnat", "irmsd",
        "lrmsd", "pair_iptm"]
PRED = "/home/ebrahim/tts_abag/poses/predictions_template"


def one(k):
    pdb, ab, ag, seed, smp = k
    try:
        ch = resolve(pdb, seed, smp, ab, ag)
    except Exception as e:
        return ["__ERR__", f"{pdb}_{ab}{ag}", f"{type(e).__name__}: {e}"]
    if ch["dockq"] is None:
        return ["__ERR__", f"{pdb}_{ab}{ag}", "dockq is None"]
    # interface ipTM for the MODEL chains OST matched, never the global iptm
    pi = None
    try:
        M = json.load(open(f"{PRED}/{pdb}/seed_{seed}/predictions/"
                           f"{pdb}_summary_confidence_sample_{smp}.json")).get("chain_pair_iptm")
        i, j = chain_idx(ch["mdl_ab"]), chain_idx(ch["mdl_ag"])
        if M and 0 <= i < len(M) and 0 <= j < len(M):
            pi = M[i][j]
    except Exception:
        pi = None
    return [pdb, ab, ag, seed, smp, ch["dockq"], ch["fnat"], ch["irmsd"], ch["lrmsd"], pi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    A = ap.parse_args()

    keys = set()
    for r in csv.DictReader(open(f"{EXP}/artifacts/epitope_overlap.csv")):
        if SEED_LO <= int(r["seed"]) <= SEED_HI:
            keys.add((r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"]))
    keys = sorted(keys)
    print(f"{len(keys)} poses over "
          f"{len({(p, a, g) for p, a, g, _, _ in keys})} interfaces", file=sys.stderr)

    errs, n = collections.Counter(), 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        with Pool(A.jobs) as pool:
            for row in pool.imap_unordered(one, keys, chunksize=64):
                if row[0] == "__ERR__":
                    errs[row[1]] += 1
                    continue
                w.writerow(row)
                n += 1
    print(f"wrote {n} poses to {OUT}")
    if errs:
        print(f"{sum(errs.values())} poses dropped over {len(errs)} interfaces; worst:")
        for k, v in errs.most_common(8):
            print(f"  {v:>5}  {k}")


if __name__ == "__main__":
    main()
