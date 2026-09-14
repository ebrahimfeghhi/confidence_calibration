#!/usr/bin/env python
"""Do the 5 samples of one seed look different THROUGH THE CONFIDENCE HEAD'S EYES?

THE PUZZLE. On 9x1h_CD, ipTM is ~constant within a seed (0.115-0.145 across seed 300014's five
samples) while those same poses span DockQ 0.044-0.643 and epitope Jaccard 0.31-0.86. Across all 262
PXM interfaces, 87% of ipTM variance is BETWEEN seeds and only 13% within, versus 40% between for
DockQ. Two explanations:
  (a) the head cannot RESOLVE these poses -- its input is too coarse to tell them apart, so flat ipTM
      is the correct response to inputs that really are nearly identical, or
  (b) the head can see the difference and ignores it, in which case the seed-level effect is real.

WHAT THE HEAD ACTUALLY SEES. Per protenix/model/modules/confidence.py, the only per-pose input is a
distance map: one representative atom per token (`distogram_rep_atom_mask`), all pairwise distances,
one-hot encoded into 39 bins from 3.25 A to 52.0 A in 1.25 A steps, everything beyond 52 A collapsed
into the top bin. No coordinates, no orientation, no chirality. This script reproduces exactly that
encoding and asks how much the five samples differ in it.

THE COMPARISON THAT MATTERS is cross-chain. Intra-chain distances are near-identical by construction
(the folds do not change -- per-chain lDDT SD is 0.007), so pooling all pairs would drown the signal.
The antibody-antigen block is where a change of epitope must show up, if it shows up at all.

METRIC: fraction of cross-chain token pairs assigned to a DIFFERENT bin between two poses, plus the
mean absolute bin shift. Reported against the same pairs' DockQ and epitope-Jaccard differences, so
"similar in the head's view" and "similar in reality" sit side by side.

Usage: .../foldbench/bin/python scripts/tmpl_distmap_within_seed.py --interface 9x1h_CD [--seeds ...]
"""
import argparse
import collections
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import _ost  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
# exactly the binning in ConfidenceHead.__init__
LOWER = np.arange(3.25, 52.0, 1.25)
N_BINS = len(LOWER)


def rep_coords(ent, chain):
    """One representative atom per residue (CB, or CA for glycine) -- the distogram rep atom."""
    ch = ent.FindChain(chain)
    out = []
    for r in ch.residues:
        a = r.FindAtom("CB")
        if not a.IsValid():
            a = r.FindAtom("CA")
        if a.IsValid():
            p = a.GetPos()
            out.append([p[0], p[1], p[2]])
    return np.array(out)


def binned(A, B):
    """Cross-chain distance matrix, one-hot bin index per pair (the head's actual input)."""
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return np.digitize(d, LOWER)          # 0 = below 3.25, N_BINS = beyond 52 A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="9x1h_CD")
    ap.add_argument("--seeds", default="")
    A = ap.parse_args()
    pdb, chains = A.interface.split("_")
    c1, c2 = chains[0], chains[1:]

    epi = {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        if r["pdb"] == pdb:
            epi[(int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])
    meta = {}
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] == pdb and {r["chain1"], r["chain2"]} == {c1, c2} and r["dockq"]:
            meta[(int(r["seed"]), int(r["sample"]))] = (float(r["pair_iptm"]), float(r["dockq"]))

    want = [int(s) for s in A.seeds.split(",")] if A.seeds else None
    byseed = collections.defaultdict(list)
    for k in meta:
        if want is None or k[0] in want:
            byseed[k[0]].append(k[1])

    _, io, _ = _ost()
    print(f"{A.interface}: the head sees {N_BINS} distance bins, 3.25-52.0 A in 1.25 A steps\n")
    print(f"{'seed':>8}{'pair':>8}{'d ipTM':>9}{'d DockQ':>10}{'d epiJ':>9}"
          f"{'% cross-chain pairs in a DIFFERENT bin':>40}{'mean |bin shift|':>18}")
    rows = []
    for seed in sorted(byseed):
        maps, keep = {}, []
        for smp in sorted(byseed[seed]):
            # PXMeter's mapping tells us which MODEL chains correspond to the reference pair
            jf = f"{P}/pxm_json/{pdb}_{seed}_{smp}.json"
            if not os.path.exists(jf):
                continue
            r2m = json.load(open(jf)).get("ref_to_model_chain_mapping", {})
            if c1 not in r2m or c2 not in r2m:
                continue
            cif = f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{smp}.cif"
            if not os.path.exists(cif):
                continue
            ent = io.LoadMMCIF(cif, fault_tolerant=True)
            Aa, Bb = rep_coords(ent, r2m[c1]), rep_coords(ent, r2m[c2])
            if len(Aa) == 0 or len(Bb) == 0:
                continue
            maps[smp] = binned(Aa, Bb)
            keep.append(smp)
        for a, b in itertools.combinations(keep, 2):
            X, Y = maps[a], maps[b]
            if X.shape != Y.shape:
                continue
            diff = (X != Y).mean()
            shift = np.abs(X.astype(int) - Y.astype(int)).mean()
            dip = abs(meta[(seed, a)][0] - meta[(seed, b)][0])
            ddq = abs(meta[(seed, a)][1] - meta[(seed, b)][1])
            dej = abs(epi.get((seed, a), np.nan) - epi.get((seed, b), np.nan))
            rows.append((seed, a, b, dip, ddq, dej, diff, shift))
            print(f"{seed:>8}{f'{a}-{b}':>8}{dip:>9.3f}{ddq:>10.3f}{dej:>9.2f}"
                  f"{100*diff:>39.1f}%{shift:>18.2f}")
    if rows:
        R = np.array([r[3:] for r in rows], float)
        print(f"\nover {len(rows)} within-seed pose pairs:")
        print(f"   median |d ipTM| {np.median(R[:,0]):.4f}   median |d DockQ| {np.median(R[:,1]):.3f}"
              f"   median |d epiJ| {np.nanmedian(R[:,2]):.2f}")
        print(f"   median share of cross-chain pairs in a different bin: {100*np.median(R[:,3]):.1f}%")
        ok = ~np.isnan(R[:, 2])
        if ok.sum() > 3 and R[ok, 3].std() > 1e-9:
            print(f"   r(bin-difference, |d DockQ|) = {np.corrcoef(R[ok,3], R[ok,1])[0,1]:+.3f}")
            print(f"   r(bin-difference, |d ipTM|)  = {np.corrcoef(R[ok,3], R[ok,0])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
