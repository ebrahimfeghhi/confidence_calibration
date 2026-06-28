"""Arm B: DIRECTED gradient repulsion. Steer each particle's s_lm to maximize W1
distance from BOTH run-1 basin centroids (B1+B2), short horizon K=3, the proven
recipe (lr=0.3, ~6 steps) that previously reached G. 16 particles, distinct z0
(same as noise arm), small per-particle s_lm init noise for diversity. Eval each
at full horizon: basin (vs G/B1/B2), escaped (vs run-1 cloud), valid (sharpness).

Compare to the noise arm (3/16 escaped, 0 G). Run in tmux gpu0, CUDA_VISIBLE_DEVICES=7.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,  # noqa
                          lift_to_lm_z, distogram_from_s_lm, cross_distogram_from_z, w1,
                          BASELINE, N_LOOPS, AB, C)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
B = 16
CROSS_NP = (AB.stop - AB.start) * (C.stop - C.start)
K = 3            # short steering horizon (proven)
STEPS = 6        # few steps (15 collapses)
LR = 0.3         # proven (0.1 trap, 0.5 supervised)


def entropy(P):  # [.,np,nb] -> scalar mean bin-entropy
    return (-(P * (P + 1e-9).log()).sum(-1)).mean().item()


def main():
    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, _, _ = load_centroids(model, pair_mask)                       # G centroid: diagnostic
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    base_norm = base_s_lm.norm().item()

    # run-1 B1+B2 centroids from the cloud (cluster into 2 modes, label-free)
    cloud = torch.from_numpy(np.load(OUT / "pass1_loop16_distograms.npy").astype(np.float32)).cuda()
    cl_cdf = cloud.cumsum(-1).reshape(cloud.shape[0], -1)
    Wm = np.zeros((cloud.shape[0], cloud.shape[0]))
    for i in range(cloud.shape[0]):
        Wm[i] = (cl_cdf - cl_cdf[i]).abs().sum(1).cpu().numpy() / CROSS_NP
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(Wm)
    g0, g1 = np.where(lab == 0)[0], np.where(lab == 1)[0]
    big, small = (g0, g1) if len(g0) >= len(g1) else (g1, g0)
    def cen(idx):
        P = cloud[idx].mean(0)
        return (P / P.sum(-1, keepdim=True)).detach()
    cB1, cB2 = cen(big), cen(small)                                   # B1 dense, B2 second mode
    print(f"centroids from run-1: B1 n={len(big)}, B2 n={len(small)}", flush=True)

    # escape threshold + sharpness reference from run-1 cloud
    nn = []
    for i in range(cloud.shape[0]):
        d = (cl_cdf - cl_cdf[i]).abs().sum(1) / CROSS_NP
        d[i] = 1e9
        nn.append(d.min().item())
    tau = max(nn) * 1.3
    ent_ref = (-(cloud * (cloud + 1e-9).log()).sum(-1)).mean(-1).mean().item()
    print(f"escape tau={tau:.3f}  ent_ref={ent_ref:.3f}", flush=True)

    print(f"\n{'p':>2} {'repel0':>7} {'repelF':>7} {'nearest':>8} {'dG':>6} {'dB1':>6} {'dB2':>6} "
          f"{'dCloud':>7} {'esc':>4} {'ent':>6} {'valid':>6}", flush=True)
    t0 = time.time()
    summ = dict(esc=0, valid=0, G=0, B1=0, B2=0)
    rows = []
    for i in range(B):
        z0 = fixed_z0(model, z_init, seed=i)                          # distinct z0 (same as noise arm)
        torch.manual_seed(2000 + i)
        d = torch.randn_like(base_s_lm)
        s_lm = (base_s_lm + 0.1 * base_norm * d / d.norm()).detach().requires_grad_(True)
        opt = torch.optim.Adam([s_lm], lr=LR)
        repel0 = None
        for step in range(STEPS):
            with torch.enable_grad():
                P, _ = distogram_from_s_lm(model, s_lm, z_init, pair_mask, z0, total_steps=K)
                rep = w1(P, cB1) + w1(P, cB2)
                loss = -rep
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([s_lm], 10.0)
            opt.step()
            if repel0 is None:
                repel0 = rep.item()
            if step > 0 and step % 2 == 0:
                for gpar in opt.param_groups:
                    gpar["lr"] *= 0.5
        repelF = rep.item()
        with torch.no_grad():
            P17, _ = distogram_from_s_lm(model, s_lm.detach(), z_init, pair_mask, z0,
                                         total_steps=N_LOOPS + 1)
        dG, dB1, dB2 = w1(P17, Pg).item(), w1(P17, cB1).item(), w1(P17, cB2).item()
        near = min([("G", dG), ("B1", dB1), ("B2", dB2)], key=lambda kv: kv[1])[0]
        pc = P17.cumsum(-1).reshape(-1)
        dcloud = ((cl_cdf - pc).abs().sum(1) / CROSS_NP).min().item()
        ent = entropy(P17)
        esc = dcloud > tau
        valid = esc and (ent < ent_ref * 1.2)
        summ["esc"] += esc; summ["valid"] += valid; summ[near] += 1
        rows.append((i, dG, esc, valid, near))
        print(f"{i:>2} {repel0:>7.3f} {repelF:>7.3f} {near:>8} {dG:>6.3f} {dB1:>6.3f} {dB2:>6.3f} "
              f"{dcloud:>7.3f} {str(esc):>4} {ent:>6.3f} {str(valid):>6}", flush=True)
        del opt, s_lm, P17
        torch.cuda.empty_cache()

    print(f"\nforward+steer done {time.time()-t0:.0f}s; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
          flush=True)
    print(f"\nARM B (directed repulsion): {summ['esc']}/16 escaped, {summ['valid']}/16 valid, "
          f"nearest G={summ['G']} B1={summ['B1']} B2={summ['B2']}", flush=True)
    print(f"  G-region hits (dG<0.3): {sum(1 for r in rows if r[1] < 0.3)}; "
          f"best dG={min(r[1] for r in rows):.3f}", flush=True)
    print("  (noise arm baseline: 3/16 escaped, 3/16 valid, 0 G)", flush=True)


if __name__ == "__main__":
    main()
