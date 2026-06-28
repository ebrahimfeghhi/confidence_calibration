"""Noise arm: force 16 particles out of BOTH default basins with a persistent
isotropic s_lm perturbation (gradient-free), swept across 4 magnitudes. Same z0
as run 1 (seeds 0-15) so it's a controlled with/without-perturbation comparison.

Judge at loop 16: escaped the run-1 cloud (B1+B2)? valid (distogram sharpness vs
run-1)? nearest centroid (G/B1/B2, diagnostic)? Reports dose-response by magnitude.

Run in tmux gpu0, CUDA_VISIBLE_DEVICES=7.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, single_rep_from_hidden, lift_to_lm_z,  # noqa: E402
                          _loop_body, load_hidden, BASELINE, w1, AB, C, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
B = 16
CROSS_NP = (AB.stop - AB.start) * (C.stop - C.start)
ALPHAS = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5,
          1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]      # magnitude sweep across the 16


def cross_distogram_batched(model, z, pair_mask):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))
    return F.softmax(logits, dim=-1)[:, AB, C, :].reshape(z.shape[0], -1, logits.shape[-1])


def entropy(P):                                          # [B,np,nb] -> [B] mean bin-entropy
    return (-(P * (P + 1e-9).log()).sum(-1)).mean(-1)


def alpha_at(amax, t):                                   # hold to loop12, anneal to 0 by 16
    return amax * (1.0 if t <= 12 else max(0.0, (16 - t) / 4.0))


def main():
    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE))   # [1,L,Cdim]
    base_norm = base_s_lm.norm().item()
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]
    print(f"||base_s_lm||={base_norm:.2f}", flush=True)

    # run-1 cloud (16 loop16 distograms) for escape + sharpness reference
    cloud = torch.from_numpy(np.load(OUT / "pass1_loop16_distograms.npy").astype(np.float32)).cuda()
    cl_cdf = cloud.cumsum(-1).reshape(cloud.shape[0], -1)
    # internal kNN spread of run-1 cloud -> escape threshold
    nn = []
    for i in range(cloud.shape[0]):
        d = (cl_cdf - cl_cdf[i]).abs().sum(1) / CROSS_NP
        d[i] = 1e9
        nn.append(d.min().item())
    tau = max(nn) * 1.3
    ent_ref = entropy(cloud).mean().item()
    print(f"run-1 cloud: escape tau={tau:.3f}  mean entropy={ent_ref:.3f}", flush=True)

    # particles: same z0 as run 1, persistent random direction, swept magnitude
    z = torch.empty(B, L, L, Cdim, device=z_init.device, dtype=z_init.dtype)
    dirs = torch.empty(B, L, Cdim, device=z_init.device)
    for i in range(B):
        torch.manual_seed(i)
        z[i] = model._init_pair_state(z_init)[0].to(z.dtype)          # same z0 as run 1
        torch.manual_seed(1000 + i)
        d = torch.randn(L, Cdim, device=z_init.device)
        dirs[i] = d / d.norm()                                        # unit Frobenius direction
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()
    amax = torch.tensor(ALPHAS, device=z_init.device)

    print("running noise arm (16 particles, swept magnitude)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for t in range(N_LOOPS + 1):
                scale = (amax * alpha_at(1.0, t) * base_norm).view(B, 1, 1)
                s_lm = base_s_lm + scale * dirs                       # [B,L,Cdim]
                lm_z = lift_to_lm_z(model, s_lm)
                z = _loop_body(model, z, z_init, lm_z, pm, a, b_mat)
                del lm_z, s_lm
    final = cross_distogram_batched(model, z, pm).float()             # [B,np,nb]
    torch.cuda.synchronize()
    print(f"forward {time.time()-t0:.0f}s; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
          flush=True)

    ent = entropy(final).cpu().numpy()
    fcdf = final.cumsum(-1).reshape(B, -1)
    print(f"\n{'p':>2} {'amax':>5} {'nearest':>8} {'dG':>6} {'dB1':>6} {'dB2':>6} "
          f"{'dCloud':>7} {'esc':>4} {'ent':>6} {'valid':>6}", flush=True)
    summ = {}
    for i in range(B):
        P = final[i]
        dG, dB1, dB2 = w1(P, Pg).item(), w1(P, Pb1).item(), w1(P, Pb2).item()
        near = min([("G", dG), ("B1", dB1), ("B2", dB2)], key=lambda kv: kv[1])[0]
        dcloud = ((cl_cdf - fcdf[i]).abs().sum(1) / CROSS_NP).min().item()
        esc = dcloud > tau
        valid = esc and (ent[i] < ent_ref * 1.2)
        print(f"{i:>2} {ALPHAS[i]:>5.2f} {near:>8} {dG:>6.3f} {dB1:>6.3f} {dB2:>6.3f} "
              f"{dcloud:>7.3f} {str(esc):>4} {ent[i]:>6.3f} {str(valid):>6}", flush=True)
        g = summ.setdefault(ALPHAS[i], dict(n=0, esc=0, valid=0, G=0, B1=0, B2=0))
        g["n"] += 1; g["esc"] += esc; g["valid"] += valid; g[near] += 1

    print("\nDose-response by magnitude:", flush=True)
    print(f"{'amax':>5} {'escaped':>8} {'valid':>6} {'nearG':>6} {'nearB1':>7} {'nearB2':>7}", flush=True)
    for amx in sorted(summ):
        g = summ[amx]
        print(f"{amx:>5.2f} {g['esc']:>3}/{g['n']} {g['valid']:>6} {g['G']:>6} {g['B1']:>7} {g['B2']:>7}",
              flush=True)
    tot_esc = sum(g["esc"] for g in summ.values())
    tot_valid = sum(g["valid"] for g in summ.values())
    print(f"\nTOTAL: {tot_esc}/16 escaped run-1 cloud, {tot_valid}/16 valid-escaped; "
          f"goal = 0 in B1/B2 (i.e. 16/16 escaped+valid)", flush=True)
    np.save(OUT / "noise_arm_loop16.npy", final.half().cpu().numpy())
    print("saved noise_arm_loop16.npy", flush=True)


if __name__ == "__main__":
    main()
