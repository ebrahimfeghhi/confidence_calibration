"""Nudge-source twin of pass1_dynamics.py. 16 particles, FIXED z0 (shared initial
pair state), each with a different 5% random nudge on the baseline s_lm (no mask,
no dropout, no grad). Full 17 loops, forward-only. Logs per-run per-loop VELOCITY
(W1 between consecutive loops) and distance-to-running-medoid -> the fan-out /
commitment curve, to check whether the knee (loop where the 16 peel apart) sits
where the z0-source version put it (~loops 3->5-6).

Saves CSV + plot. Run in tmux gpu0, CUDA_VISIBLE_DEVICES=0.
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
INIT_FRAC = 0.05
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
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()  # [1,L,dim]
    s0n = base_s_lm.norm().item()
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]

    # FIXED z0: one shared initial pair state for all 16 particles (seed 0)
    torch.manual_seed(0)
    z0_one = model._init_pair_state(z_init)[0]
    z = z0_one.unsqueeze(0).repeat(B, 1, 1, 1).contiguous().to(z_init.dtype)

    # diversity source = a DIFFERENT 5% nudge on s_lm per particle (z0 held fixed)
    s_lm = base_s_lm.repeat(B, 1, 1).clone()
    for i in range(B):
        torch.manual_seed(1000 + i)
        d = torch.randn_like(base_s_lm)                               # [1,L,dim]
        s_lm[i] = base_s_lm[0] + (d / d.norm() * INIT_FRAC * s0n)[0]
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    rows, prev = [], None
    cross16 = intra_ab = intra_c = None
    print(f"running {B} nudged particles (5%, fixed z0) x {N_LOOPS+1} loops...", flush=True)
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

    with open(OUT / "pass1_nudge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "loop", "velocity", "dist_medoid"])
        w.writeheader(); w.writerows(rows)

    # ---- label-free 2-mode split, then compare modes intra vs cross ----
    Vc = cross16.cuda()
    cdf = Vc.cumsum(-1).reshape(B, -1)
    W = np.zeros((B, B))
    for i in range(B):
        W[i] = ((cdf - cdf[i]).abs().sum(1).cpu().numpy()) / CROSS_NP
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(W)
    A_idx = np.where(lab == 0)[0]; Bn_idx = np.where(lab == 1)[0]

    def mean_block(stack, idx):
        return stack[idx].mean(0).cuda()                              # [np,nb]

    def classify(idx):
        P = mean_block(cross16, idx)
        return min({"G": w1(P, Pg).item(), "B1": w1(P, Pb1).item(), "B2": w1(P, Pb2).item()}.items(),
                   key=lambda kv: kv[1])
    print(f"\n2-mode split: modeA n={len(A_idx)} ({classify(A_idx)}), "
          f"modeB n={len(Bn_idx)} ({classify(Bn_idx)})", flush=True)

    # ---- fan-out summary (cloud radius = mean dist-to-medoid per loop) ----
    print("\nloop  meanVel   cloud-radius(mean dist_medoid)  std", flush=True)
    for lp in sorted({int(r["loop"]) for r in rows}):
        vel = [r["velocity"] for r in rows if int(r["loop"]) == lp and r["velocity"] not in ("", "nan")]
        dm = [r["dist_medoid"] for r in rows if int(r["loop"]) == lp]
        mv = np.nanmean(vel) if vel else float("nan")
        print(f"{lp:3d}  {mv:7.4f}   {np.mean(dm):.4f}   std={np.std(dm):.4f}", flush=True)

    # ---- plot ----
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
    ax[0].set_title("nudge-source settling"); ax[0].legend()
    for i in range(B):
        ax[1].plot(loops, series(i, "dist_medoid"), color="0.7", lw=0.8)
    mean_dm = [np.mean([next(r["dist_medoid"] for r in rows if r["run"] == i and r["loop"] == lp)
                        for i in range(B)]) for lp in loops]
    ax[1].plot(loops, mean_dm, "r-o", lw=2, label="cloud radius")
    ax[1].set_xlabel("loop"); ax[1].set_ylabel("distance to running medoid")
    ax[1].set_title("nudge-source fan-out (knee = commitment loop)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "pass1_nudge.png", dpi=130)
    print("\nsaved pass1_nudge.csv and pass1_nudge.png", flush=True)


if __name__ == "__main__":
    main()
