#!/usr/bin/env python
"""Convert a complex's free-run poses into pseudo-beta distograms, and report the storage cost.

Motivation: before templating on all 250 poses of 8che we need to know whether the template feature
tensor fits in memory. The distogram is the dominant term -- 39 one-hot bins per residue pair per
template -- so this measures the real footprint on the real poses rather than estimating from token
counts.

Uses the SAME pseudo-beta definition and bin edges the model uses (Templates._DGRAM_CONFIG:
3.25-50.75 A, 39 bins), so the numbers here are what the featurizer would produce, not an analogue.

Stores CB coordinates (or CA for glycine) rather than the binned distogram: the pairwise distances
are recoverable on demand and the coordinate form is N x 3 instead of N x N x 39, which is ~4 orders
of magnitude smaller and is what makes clustering 250 poses tractable.

Usage: .../python scripts/pose_distograms.py --pdb 8che [--save]
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from decoy_set import EXP, PRED, SEEDS, SAMPLES

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--save", action="store_true", help="write artifacts/<pdb>_posedgram.npz")
A = ap.parse_args()

MIN_BIN, MAX_BIN, N_BINS = 3.25, 50.75, 39     # Templates._DGRAM_CONFIG


def pseudo_beta(path):
    """(coords [N,3], chain_id [N], resnum [N]) using CB, or CA for glycine."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    xyz, ch, rn = [], [], []
    for chain in st[0]:
        for res in chain:
            want = "CA" if res.name == "GLY" else "CB"
            at = res.find_atom(want, "*") or res.find_atom("CA", "*")
            if at is None:
                continue
            xyz.append([at.pos.x, at.pos.y, at.pos.z]); ch.append(chain.name); rn.append(res.seqid.num)
    return np.array(xyz, dtype=np.float32), np.array(ch), np.array(rn, dtype=np.int32)


poses = [f"{PRED}/{A.pdb}/seed_{s}/predictions/{A.pdb}_sample_{k}.cif"
         for s in SEEDS for k in SAMPLES]
poses = [p for p in poses if os.path.exists(p)]
print(f"{A.pdb}: {len(poses)} poses found (seeds {min(SEEDS)}-{max(SEEDS)})")

coords, chains, resnums = [], None, None
for p in poses:
    xyz, ch, rn = pseudo_beta(p)
    if chains is None:
        chains, resnums = ch, rn
    elif len(xyz) != len(chains):
        print(f"  SKIP {os.path.basename(p)}: {len(xyz)} residues vs {len(chains)}"); continue
    coords.append(xyz)
coords = np.stack(coords)                      # [P, N, 3]
P, N, _ = coords.shape
print(f"parsed {P} poses x {N} pseudo-beta atoms")

f4 = 4  # bytes per float32
print()
print("storage, per pose and for all of them:")
print(f"  CB coordinates      [N,3]        {N*3*f4/1e6:8.2f} MB each   {P*N*3*f4/1e9:7.3f} GB total")
print(f"  pairwise distances  [N,N]        {N*N*f4/1e6:8.2f} MB each   {P*N*N*f4/1e9:7.3f} GB total")
print(f"  binned distogram    [N,N,{N_BINS}]   {N*N*N_BINS*f4/1e6:8.2f} MB each   "
      f"{P*N*N*N_BINS*f4/1e9:7.3f} GB total   (fp32)")
print(f"  same in bf16                     {N*N*N_BINS*2/1e6:8.2f} MB each   "
      f"{P*N*N*N_BINS*2/1e9:7.3f} GB total")
print()
print(f"  full template feature block [P,N,N,108] bf16: {P*N*N*108*2/1e9:.2f} GB")
print("  (108 = distogram 39 + unit_vector 3 + 2 masks + restype_i/j 64)")

# cross-chain block only -- the part that encodes the dock
uniq = list(dict.fromkeys(chains.tolist()))
cross = (chains[:, None] != chains[None, :])
print()
print(f"chains: {uniq}")
print(f"  cross-chain pairs: {cross.sum():,} of {N*N:,} ({100*cross.sum()/(N*N):.1f}%)")

if A.save:
    out = f"{EXP}/artifacts/{A.pdb}_posedgram.npz"
    np.savez_compressed(out, coords=coords, chains=chains, resnums=resnums,
                        poses=np.array([os.path.relpath(p, PRED) for p in poses]))
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB on disk)")
