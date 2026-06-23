"""Basin-to-basin cross-block distogram W1 matrix, under several residue-block
definitions, to (a) set the coverage X-shells precisely and (b) decide which block
makes the cleanest steering-loss target.

For each block we compute per-seed loop16 distograms (no GPU steering, just the
distogram head on saved reps), form basin centroids G / B1 / B2, and report the
W1 between centroids vs the within-basin spread. The block with the biggest
between/within ratio is the sharpest loss target.

Chain spans (7txw, L=609): A 0:219, B 219:438, C 438:609.
  A∪B × C  = the antibody-antigen interface (current loss)
  A × C, B × C = the two individual ab-ag interfaces
  A × B    = antibody-internal (NEGATIVE control — constant across basins)
  C × C    = antigen-internal (fold-conserved control)

Run (esm venv, GPU): python analysis/grad_steer/basin_w1_matrix.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import load, REPS, BASELINE  # noqa: E402

G = [10, 14, 33, 39]
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
LAB = {"G": G, "B1": B1, "B2": B2}

A, B, C = slice(0, 219), slice(219, 438), slice(438, 609)
BLOCKS = {
    "A∪B×C (current loss)": (slice(0, 438), C),
    "A×C": (A, C),
    "B×C": (B, C),
    "A×B (ab-internal ctrl)": (A, B),
    "C×C (ag-internal ctrl)": (C, C),
}


def w1_np(P, Q):  # [npairs, nbins] -> scalar mean 1D-W1 over ordered bins
    return np.abs(np.cumsum(P, -1) - np.cumsum(Q, -1)).sum(-1).mean()


def main():
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]

    @torch.no_grad()
    def full_disto(z_path):
        z = torch.from_numpy(np.load(z_path).astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z)
            z = model.parcae_coda(z, pair_attention_mask=pair_mask)
        z = z.float()
        logits = model.distogram_head(z + z.transpose(-2, -3))[0]   # [L,L,nbins]
        return F.softmax(logits, dim=-1)                            # [L,L,nbins]

    store = {n: [] for n in BLOCKS}
    print("computing per-seed distograms...", flush=True)
    for s in range(60):
        P = full_disto(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy")
        for n, (r, c) in BLOCKS.items():
            store[n].append(P[r, c, :].reshape(-1, P.shape[-1]).cpu().numpy())
        del P
        torch.cuda.empty_cache()
        if s % 15 == 0:
            print(f"  seed {s}", flush=True)
    Pb = full_disto(BASELINE / "reps" / "pairformer_z_loop16.npy")
    base = {n: Pb[r, c, :].reshape(-1, Pb.shape[-1]).cpu().numpy()
            for n, (r, c) in BLOCKS.items()}

    for n in BLOCKS:
        arr = np.stack(store[n])                                   # [60, npairs, nbins]
        cent = {b: arr[idx].mean(0) for b, idx in LAB.items()}
        within = {b: np.mean([w1_np(arr[i], cent[b]) for i in LAB[b]]) for b in LAB}
        gb1 = w1_np(cent["G"], cent["B1"])
        gb2 = w1_np(cent["G"], cent["B2"])
        b1b2 = w1_np(cent["B1"], cent["B2"])
        base_g = w1_np(base[n], cent["G"])
        ratio = gb1 / max(within["G"], 1e-9)                       # G-B1 separation / within-G
        print(f"\n=== {n} ===", flush=True)
        print(f"  centroid W1:   G-B1={gb1:.3f}  G-B2={gb2:.3f}  B1-B2={b1b2:.3f}", flush=True)
        print(f"  within-basin:  G={within['G']:.3f}  B1={within['B1']:.3f}  "
              f"B2={within['B2']:.3f}", flush=True)
        print(f"  baseline->G = {base_g:.3f}   |   G-B1 / within-G = {ratio:.1f}x  "
              f"(bigger = sharper loss target)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
