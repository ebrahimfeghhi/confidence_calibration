#!/usr/bin/env python
"""When does the epitope commit? Measured on CONTACT PROBABILITY, two ways.

WHY NOT EXPECTED DISTANCE. The first version of this used the distogram's expected bin. That is
broken here: every antigen-antibody pair puts most of its mass in the ">21.4 A" catch-all bin while a
contacting pair also holds ~20% below 8 A, so the expectation averages "4 A" and ">21 A" into ~17 A --
a value the model does not believe in. P(d < 8 A) reads only the part that matters: 25x contrast
between the top antigen residue and the median, and 11 of its top 15 residues fall inside 8slb's true
epitope.

TWO LOSSES, same data (the user asked for both):
  A  per antigen residue, the length-N_ab vector of per-pair P(contact); compare two runs with
     sorted-W1 (sort both, mean |difference|), summed over antigen residues. Sorting discards WHICH
     antibody residue supplies each contact, so the loss is blind to the paratope.
  B  per antigen residue, a single number = max P(contact) over antibody partners -> an "epitope
     profile" of length N_antigen; compare with L1. The max already discards the antibody identity,
     so paratope invariance is free and no sorting is needed.

Reads the per-cycle dumps written when DUMP_CYCLE_DIR + DUMP_CYCLE_BINS are set (see
protenix/model/protenix.py), which store the full 64-bin distribution for the antigen x antibody
block so any statistic can be recomputed offline.

Usage: .../python scripts/commit_contact.py --arms nodrop2,stock2
"""
import argparse
import glob
import os
import re

import numpy as np

MIN_BIN, MAX_BIN, N_BIN = 2.3125, 21.6875, 64
BW = (MAX_BIN - MIN_BIN) / N_BIN
CONTACT_BINS = int((8.0 - MIN_BIN) / BW) + 1        # bins fully below 8 A


def load(arm):
    """-> {pass: {cycle: P(contact) matrix [N_antigen, N_antibody]}}"""
    out = {}
    for f in sorted(glob.glob(f"artifacts/commit/{arm}/*_cycle*.npz")):
        m = re.search(r"_pass(\d+)_cycle(\d+)\.npz$", os.path.basename(f))
        z = np.load(f)
        if "probs" not in z:
            continue
        p = z["probs"].astype(np.float32)
        out.setdefault(int(m.group(1)), {})[int(m.group(2))] = p[..., :CONTACT_BINS].sum(-1)
    return {k: v for k, v in out.items() if len(v) >= 10}


def loss_A(X, Y):
    """sorted-W1 over each antigen residue's vector of per-pair contact probabilities."""
    return float(np.abs(np.sort(X, axis=1) - np.sort(Y, axis=1)).mean(axis=1).sum())


def loss_B(X, Y):
    """L1 between epitope profiles (max contact prob per antigen residue)."""
    return float(np.abs(X.max(axis=1) - Y.max(axis=1)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="nodrop2,stock2")
    ap.add_argument("--topk", type=int, default=40)
    A = ap.parse_args()
    arms = A.arms.split(",")
    data = {a: load(a) for a in arms}

    for name, fn in (("A  sorted-W1 over per-pair contact probs", loss_A),
                     ("B  L1 between epitope profiles", loss_B)):
        print(f"\n{'='*78}\n{name}\n{'='*78}")
        print("movement between consecutive recycles, normalised to each seed's own largest move")
        print(f"{'arm':<10}{'n':>4}  " + "".join(f"{k}->{k+1:<5}" for k in range(9)))
        for a in arms:
            M = data[a]
            ch = np.array([[fn(M[p][k], M[p][k + 1]) for k in range(9)] for p in M])
            nrm = ch / ch.max(axis=1, keepdims=True)
            print(f"{a:<10}{len(M):>4}  " + "".join(f"{v:<7.2f}" for v in nrm.mean(0)))
        print("\nraw values (same units within a row, comparable across arms)")
        for a in arms:
            M = data[a]
            ch = np.array([[fn(M[p][k], M[p][k + 1]) for k in range(9)] for p in M])
            print(f"{a:<10}{len(M):>4}  " + "".join(f"{v:<7.3f}" for v in ch.mean(0)))
        print("\nhow far apart two DIFFERENT seeds are, at each recycle")
        for a in arms:
            M = data[a]
            ps = list(M)
            bt = [np.mean([fn(M[x][k], M[y][k]) for i, x in enumerate(ps) for y in ps[i + 1:]])
                  for k in range(10)]
            print(f"{a:<10}{len(M):>4}  " + "".join(f"{v:<7.3f}" for v in bt))

    # set-overlap view: does the contacted set settle?
    print(f"\n{'='*78}\nsimilarity of the top-{A.topk} contacted residues to that seed's FINAL set\n{'='*78}")
    print(f"{'arm':<10}{'n':>4}  " + "".join(f"{k:<7}" for k in range(10)))
    for a in arms:
        M = data[a]
        J = []
        for p in M:
            fin = set(np.argsort(M[p][9].max(axis=1))[::-1][:A.topk].tolist())
            J.append([len(set(np.argsort(M[p][k].max(axis=1))[::-1][:A.topk].tolist()) & fin) /
                      len(set(np.argsort(M[p][k].max(axis=1))[::-1][:A.topk].tolist()) | fin)
                      for k in range(10)])
        J = np.array(J)
        print(f"{a:<10}{len(M):>4}  " + "".join(f"{v:<7.2f}" for v in J.mean(0)))
        drops = sum(1 for r in J if any(r[i] > 0.8 and r[i + 1] < r[i] - 0.05 for i in range(9)))
        print(f"{'':14}seeds that abandon an epitope after reaching 0.8 similarity: {drops}/{len(J)}")


if __name__ == "__main__":
    main()
