"""Run-1 dynamics: 16 z0-only particles (no mask, no dropout), full 17 loops,
forward-only. Logs per-run per-loop VELOCITY (W1 between consecutive loops) and
distance-to-running-medoid -> the settling/commitment curve (knee = best loop to
steer away from B1/B2). At loop 16 also captures INTRA-chain blocks (AB×AB, C×C)
and clusters the 16 into 2 modes (label-free) to test whether the modes differ
only in the CROSS block (reorientation) or intra-chain too (refold).

All from run 1 only. Saves CSVs + plots. Run in tmux gpu0, CUDA_VISIBLE_DEVICES=7.
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, single_rep_from_hidden, lift_to_lm_z,  # noqa: E402
                          _loop_body, load_hidden, BASELINE, w1, AB, C, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
B = 16
CROSS_NP = (AB.stop - AB.start) * (C.stop - C.start)


def readout_logits(model, z, pair_mask):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    return model.distogram_head(z + z.transpose(-2, -3))               # [B,L,L,nbins]


def block(logits, r, c):
    P = F.softmax(logits[:, r, c, :], dim=-1)
    return P.reshape(P.shape[0], -1, P.shape[-1])                      # [B, npairs, nbins]


def per_run_velocity(cross_t, cross_p):                                # [B,np,nb] -> [B]
    d = (cross_t.cumsum(-1) - cross_p.cumsum(-1)).abs().sum(-1)        # [B,np]
    return (d.mean(-1)).cpu().numpy()


def pairwise_and_medoid(cross):                                        # [B,np,nb]
    V = cross.cumsum(-1).reshape(B, -1)
    W = torch.zeros(B, B)
    for i in range(B):
        W[i] = (V - V[i]).abs().sum(1).cpu() / CROSS_NP
    W = W.numpy()
    med = int(W.sum(1).argmin())
    return W, med, W[med]


def main():
    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)                    # diagnostic only
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE))
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]

    z = torch.empty(B, L, L, Cdim, device=z_init.device, dtype=z_init.dtype)
    for i in range(B):
        torch.manual_seed(i)
        z[i] = model._init_pair_state(z_init)[0].to(z.dtype)
    s_lm = base_s_lm.repeat(B, 1, 1)
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    rows, prev = [], None
    cross16 = intra_ab = intra_c = None
    print(f"running {B} z0-only particles x {N_LOOPS+1} loops...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for lp in range(N_LOOPS + 1):
                lm_z = lift_to_lm_z(model, s_lm)
                z = _loop_body(model, z, z_init, lm_z, pm, a, b_mat)
                logits = readout_logits(model, z, pm)
                cross = block(logits, AB, C).float()                   # [B,CROSS_NP,nb]
                vel = per_run_velocity(cross, prev) if prev is not None else [np.nan] * B
                _, med, dmed = pairwise_and_medoid(cross)
                for i in range(B):
                    rows.append(dict(run=i, loop=lp, velocity=round(float(vel[i]), 5),
                                     dist_medoid=round(float(dmed[i]), 5)))
                if lp == N_LOOPS:
                    cross16 = cross.cpu()
                    intra_ab = block(logits, AB, AB).float().cpu()
                    intra_c = block(logits, C, C).float().cpu()
                prev = cross
                del logits
                torch.cuda.empty_cache()
    print(f"forward done {time.time()-t0:.0f}s; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
          flush=True)
    del z, prev
    torch.cuda.empty_cache()

    with open(OUT / "pass1_dynamics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "loop", "velocity", "dist_medoid"])
        w.writeheader(); w.writerows(rows)

    # ---- label-free 2-mode split of run 1, then compare modes intra vs cross ----
    Vc = cross16.cuda()
    cdf = Vc.cumsum(-1).reshape(B, -1)
    W = np.zeros((B, B))
    for i in range(B):
        W[i] = ((cdf - cdf[i]).abs().sum(1).cpu().numpy()) / CROSS_NP
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(W)
    A_idx = np.where(lab == 0)[0]; Bn_idx = np.where(lab == 1)[0]

    def mean_block(stack, idx):
        return stack[idx].mean(0).cuda()                              # [np,nb]
    crossA, crossB = mean_block(cross16, A_idx), mean_block(cross16, Bn_idx)
    abA, abB = mean_block(intra_ab, A_idx), mean_block(intra_ab, Bn_idx)
    cA, cB = mean_block(intra_c, A_idx), mean_block(intra_c, Bn_idx)
    w_cross = w1(crossA, crossB).item()
    w_ab = w1(abA, abB).item()
    w_c = w1(cA, cB).item()

    # diagnostic label of each mode via centroids
    def classify(idx):
        P = mean_block(cross16, idx)
        return min({"G": w1(P, Pg).item(), "B1": w1(P, Pb1).item(), "B2": w1(P, Pb2).item()}.items(),
                   key=lambda kv: kv[1])
    print(f"\n2-mode split: modeA n={len(A_idx)} ({classify(A_idx)}), "
          f"modeB n={len(Bn_idx)} ({classify(Bn_idx)})", flush=True)
    print(f"mode-vs-mode W1:  CROSS(AB×C)={w_cross:.4f}   intra AB×AB={w_ab:.4f}   "
          f"intra C×C={w_c:.4f}", flush=True)
    print("  (reorientation premise holds if CROSS >> intra)", flush=True)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loops = sorted({r["loop"] for r in rows})

    def series(run, key):
        return [next(r[key] for r in rows if r["run"] == run and r["loop"] == lp) for lp in loops]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i in range(B):
        ax[0].plot(loops, series(i, "velocity"), color="0.7", lw=0.8)
    mean_vel = [np.nanmean([next(r["velocity"] for r in rows if r["run"] == i and r["loop"] == lp)
                            for i in range(B)]) for lp in loops]
    ax[0].plot(loops, mean_vel, "b-o", lw=2, label="mean velocity")
    ax[0].set_xlabel("loop"); ax[0].set_ylabel("velocity (W1 loop i vs i-1)")
    ax[0].set_title("run-1 settling: knee = best loop to steer"); ax[0].legend()
    for i in range(B):
        ax[1].plot(loops, series(i, "dist_medoid"), color="0.7", lw=0.8)
    ax[1].set_xlabel("loop"); ax[1].set_ylabel("distance to running medoid")
    ax[1].set_title("run-1 differentiation (B1 core vs drifters)")
    plt.tight_layout(); plt.savefig(OUT / "pass1_dynamics.png", dpi=130)
    print("\nsaved pass1_dynamics.csv and pass1_dynamics.png", flush=True)
    print("mean velocity by loop:", " ".join(f"{v:.3f}" for v in mean_vel[1:]), flush=True)


if __name__ == "__main__":
    main()
