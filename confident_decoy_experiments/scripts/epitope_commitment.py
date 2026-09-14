#!/usr/bin/env python
"""When during recycling does the epitope commit?

WHY: the sequential-repulsion plan pushes a new seed off the previous seed's epitope at the recycle
where that epitope forms. That loop index is the whole design -- push before it and there is no
epitope to push away from, push after it and the basin is locked. This measures the curve so the
choice is made from data instead of guessed.

INPUT: the per-cycle dumps Protenix already writes when DUMP_CYCLE_DIR is set (see
protenix/model/protenix.py). Each file holds `exp_bin` -- the EXPECTED DISTOGRAM BIN for every
cross-chain residue pair, flattened -- plus `asym_id` so the pairs can be reshaped back into an
antigen x antibody matrix. No model changes, no full-distogram storage.

THE CHANGE METRIC IS THE LOSS ITSELF. For each antigen residue we take its vector of expected
distances to every antibody residue, and compare consecutive cycles with the sorted-W1 distance
(sort both, mean absolute difference), summed over antigen residues. That is exactly the objective
the repulsion would optimise, so "stabilised" means "stabilised as far as the loss can tell" -- the
criterion and the objective cannot disagree. Sorting also makes it blind to WHICH antibody residue
provides a contact, so a pure paratope rearrangement does not register as epitope change.

READING THE CURVE: it should rise while the pose is being decided and decay once it locks. The loop
to push at is where it has just finished falling -- the epitope exists (so there is something to
repel from) but the remaining recycles have not yet cemented it. Reported threshold-free-ish: the
first cycle where the change drops below a FRACTION of that run's own peak, swept over several
fractions so you can see whether the answer is stable. If 5%/10%/20% all pick the same cycle the
threshold is doing no work; if they disagree, "the commitment loop" is not well defined for that
target and the plan needs revisiting.

Usage:
  .../python scripts/epitope_commitment.py --dump-dir artifacts/commit/8slb --out artifacts/commit/8slb.csv
"""
import argparse
import glob
import os
import re

import numpy as np


def load_cycles(dump_dir, tag=None):
    """-> {pass_id: {cycle: (exp_bin_flat, asym_id)}}"""
    out = {}
    for f in sorted(glob.glob(f"{dump_dir}/*_cycle*.npz")):
        m = re.search(r"(.*)_pass(\d+)_cycle(\d+)\.npz$", os.path.basename(f))
        if not m:
            continue
        if tag and m.group(1) != tag:
            continue
        z = np.load(f)
        out.setdefault(int(m.group(2)), {})[int(m.group(3))] = (z["exp_bin"], z["asym_id"])
    return out


def cross_matrix(exp_flat, asym_id, ab_ids, ag_ids):
    """Rebuild the antigen x antibody expected-bin matrix from the flattened cross-chain vector.

    protenix saved `_exp[_cross]` where _cross = asym_id[None,:] != asym_id[:,None], i.e. row-major
    over all ordered off-diagonal-chain pairs. Rebuild the full [N,N] then slice.
    """
    n = len(asym_id)
    cross = asym_id[None, :] != asym_id[:, None]
    full = np.zeros((n, n), dtype=np.float32)
    full[cross] = exp_flat.astype(np.float32)
    ag = np.isin(asym_id, ag_ids)
    ab = np.isin(asym_id, ab_ids)
    return full[np.ix_(ag, ab)]          # [N_antigen, N_antibody]


def sorted_w1(A, B):
    """Sum over antigen residues of W1 between their distance vectors.

    W1 between two equal-length 1-D samples has the closed form 'sort both, mean |difference|'.
    The sort is what discards WHICH antibody residue supplies each distance, making this blind to
    the paratope and sensitive only to how buried each antigen residue is.
    """
    return float(np.abs(np.sort(A, axis=1) - np.sort(B, axis=1)).mean(axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ab-asym", default="0", help="comma-separated asym_id values for the ANTIBODY")
    ap.add_argument("--ag-asym", default="1", help="comma-separated asym_id values for the ANTIGEN")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    ab_ids = [int(x) for x in A.ab_asym.split(",")]
    ag_ids = [int(x) for x in A.ag_asym.split(",")]

    passes = load_cycles(A.dump_dir, A.tag)
    if not passes:
        raise SystemExit(f"no cycle dumps in {A.dump_dir}")
    rows = []
    for pid in sorted(passes):
        cyc = passes[pid]
        ks = sorted(cyc)
        mats = {}
        for k in ks:
            flat, aid = cyc[k]
            mats[k] = cross_matrix(flat, aid, ab_ids, ag_ids)
        ch = [(ks[i + 1], sorted_w1(mats[ks[i]], mats[ks[i + 1]])) for i in range(len(ks) - 1)]
        if not ch:
            continue
        peak = max(v for _, v in ch)
        print(f"\npass {pid}: {len(ks)} cycles, antigen x antibody = {mats[ks[0]].shape}")
        print(f"  {'cycle':>6}{'change vs prev':>16}{'% of peak':>11}")
        for c, v in ch:
            print(f"  {c:>6}{v:>16.2f}{100 * v / peak:>10.0f}%")
        for frac in (0.20, 0.10, 0.05):
            hit = [c for c, v in ch if v <= frac * peak]
            first = hit[0] if hit else None
            print(f"  first cycle with change <= {frac:.0%} of peak: {first}")
            rows.append(dict(pass_id=pid, frac=frac, commit_cycle=first,
                             peak=round(peak, 3),
                             peak_cycle=max(ch, key=lambda x: x[1])[0]))
        np.savez(f"{A.out}.pass{pid:02d}.npz",
                 cycles=np.array([c for c, _ in ch]),
                 change=np.array([v for _, v in ch]))

    import csv
    with open(A.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {A.out}")
    sel = {}
    for r in rows:
        sel.setdefault(r["pass_id"], {})[r["frac"]] = r["commit_cycle"]
    print("\nSTABILITY of the chosen cycle across thresholds (same answer = threshold does no work):")
    for pid, d in sorted(sel.items()):
        vals = [d[f] for f in (0.20, 0.10, 0.05)]
        print(f"  pass {pid}: 20% -> {vals[0]}, 10% -> {vals[1]}, 5% -> {vals[2]}"
              f"   {'STABLE' if len(set(vals)) == 1 else 'UNSTABLE'}")


if __name__ == "__main__":
    main()
