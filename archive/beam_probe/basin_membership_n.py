"""How many B1 reference seeds do you need to reliably tell that a new run is
NOT in the B1 basin? Subsample n B1 seeds -> centroid; score every held-out seed
by distance-to-centroid; measure how well that separates held-out B1 (in-basin)
from G+B2 (out-of-basin). Sweep n. loop-16 distograms.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import load, cross_distogram_from_z, AB, C, GOOD, REPS  # noqa: E402
from basins import B1, B2  # noqa: E402

NPAIRS = (AB.stop - AB.start) * (C.stop - C.start)
LOOP = 16
RNG = np.random.default_rng(0)


def cdf(model, pair_mask, seed):
    z = torch.from_numpy(np.load(
        REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{LOOP:02d}.npy"
    ).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)
    out = P.cumsum(-1).reshape(-1)
    del z, P
    return out


def main():
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    print(f"computing loop-{LOOP} distograms for 60 seeds...", flush=True)
    V = torch.stack([cdf(model, pair_mask, s) for s in range(60)])     # [60, D] GPU fp32
    print(f"V {tuple(V.shape)}", flush=True)

    b1 = np.array(B1)
    out_idx = np.array(GOOD + B2)                                       # "not B1"

    def dist(ref, q):
        c = V[ref].mean(0)
        return ((V[q] - c).abs().sum(1) / NPAIRS).cpu().numpy()

    print(f"\n{'n_B1':>4}  {'AUC':>6}  {'clean-sep%':>10}  {'margin(minOut-maxIn)':>20}", flush=True)
    R = 300
    for n in (2, 3, 4, 5, 6, 8, 10, 15, 20):
        aucs, cleans, margins = [], [], []
        for _ in range(R):
            ref = RNG.choice(b1, size=n, replace=False)
            held_b1 = np.array([s for s in b1 if s not in set(ref)])
            q = np.concatenate([held_b1, out_idx])
            y = np.concatenate([np.zeros(len(held_b1)), np.ones(len(out_idx))])
            s = dist(ref, q)
            aucs.append(roc_auc_score(y, s))
            max_in = s[y == 0].max()
            min_out = s[y == 1].min()
            cleans.append(1.0 if min_out > max_in else 0.0)
            margins.append(min_out - max_in)
        print(f"{n:>4}  {np.mean(aucs):6.3f}  {100*np.mean(cleans):9.1f}%  "
              f"{np.mean(margins):+20.3f}", flush=True)
    print("\n(clean-sep% = fraction of subsamples where a single threshold perfectly "
          "splits held-out B1 from G+B2; margin>0 = separable)", flush=True)


if __name__ == "__main__":
    main()
