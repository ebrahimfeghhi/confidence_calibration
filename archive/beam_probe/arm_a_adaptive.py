"""Arm A: direction-adaptive blind noise (gradient-free). 16 slots, distinct z0,
fixed magnitude alpha=1.5 (the band where the sweep saw escapes). Across rounds,
SEARCH over directions: freeze a slot once it achieves a VALID escape, re-roll the
failures with new directions biased toward any direction that already worked
(ES-lite). Tests whether selecting/adapting DIRECTION beats one blind batch
(noise arm got 3/16), without gradients.

Run in tmux gpu0, CUDA_VISIBLE_DEVICES=7.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, fixed_z0, single_rep_from_hidden, lift_to_lm_z,  # noqa
                          _loop_body, load_hidden, BASELINE, w1, AB, C, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
B = 16
ROUNDS = 3
ALPHA = 1.5
CROSS_NP = (AB.stop - AB.start) * (C.stop - C.start)


def cross_batched(model, z, pm):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pm)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))
    return F.softmax(logits, dim=-1)[:, AB, C, :].reshape(z.shape[0], -1, logits.shape[-1])


def alpha_at(t):
    return ALPHA * (1.0 if t <= 12 else max(0.0, (16 - t) / 4.0))


def run_batch(model, z_init, base_s_lm, base_norm, pm, a, b_mat, z0s, dirs):
    n = len(z0s)
    L, Cdim = z_init.shape[1], z_init.shape[-1]
    z = torch.stack([z0s[i][0] for i in range(n)]).to(z_init.dtype)
    D = torch.stack(dirs)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for t in range(N_LOOPS + 1):
                s_lm = base_s_lm + (alpha_at(t) * base_norm) * D
                lm_z = lift_to_lm_z(model, s_lm)
                z = _loop_body(model, z, z_init, lm_z, pm, a, b_mat)
                del lm_z, s_lm
    return cross_batched(model, z, pm).float()


def main():
    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, _, _ = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    base_norm = base_s_lm.norm().item()
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]

    cloud = torch.from_numpy(np.load(OUT / "pass1_loop16_distograms.npy").astype(np.float32)).cuda()
    cl_cdf = cloud.cumsum(-1).reshape(cloud.shape[0], -1)
    nn = []
    for i in range(cloud.shape[0]):
        d = (cl_cdf - cl_cdf[i]).abs().sum(1) / CROSS_NP
        d[i] = 1e9
        nn.append(d.min().item())
    tau = max(nn) * 1.3
    ent_ref = (-(cloud * (cloud + 1e-9).log()).sum(-1)).mean(-1).mean().item()
    print(f"tau={tau:.3f} ent_ref={ent_ref:.3f} alpha={ALPHA}", flush=True)

    z0s = [fixed_z0(model, z_init, seed=i) for i in range(B)]
    def rand_dir(seed):
        torch.manual_seed(seed)
        d = torch.randn(L, Cdim, device=z_init.device)
        return d / d.norm()
    dirs = [rand_dir(3000 + i) for i in range(B)]
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    done = [None] * B            # store (dG, near) when a slot achieves valid escape
    good_dirs = []               # directions that produced valid escapes (for ES bias)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for rnd in range(ROUNDS):
        active = [i for i in range(B) if done[i] is None]
        if not active:
            break
        pm_a = pair_mask.expand(len(active), *pair_mask.shape[1:]).contiguous()
        finalP = run_batch(model, z_init, base_s_lm, base_norm, pm_a,
                           a, b_mat, [z0s[i] for i in active], [dirs[i] for i in active])
        ent = (-(finalP * (finalP + 1e-9).log()).sum(-1)).mean(-1).cpu().numpy()
        fcdf = finalP.cumsum(-1).reshape(len(active), -1)
        newly = 0
        for j, i in enumerate(active):
            P = finalP[j]
            dG = w1(P, Pg).item()
            dcloud = ((cl_cdf - fcdf[j]).abs().sum(1) / CROSS_NP).min().item()
            esc = dcloud > tau
            valid = esc and (ent[j] < ent_ref * 1.2)
            if valid:
                done[i] = (round(dG, 3), round(dcloud, 3))
                good_dirs.append(dirs[i].clone())
                newly += 1
            else:
                # re-roll: ES bias toward a known-good direction if any, else fresh random
                if good_dirs and rng.random() < 0.5:
                    base = good_dirs[rng.integers(len(good_dirs))]
                    torch.manual_seed(7000 + rnd * B + i)
                    nd = base + 0.5 * torch.randn_like(base)
                    dirs[i] = nd / nd.norm()
                else:
                    dirs[i] = rand_dir(9000 + rnd * B + i)
        tot = sum(1 for d in done if d is not None)
        print(f"round {rnd}: ran {len(active)}, +{newly} valid escapes, cumulative {tot}/16  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del finalP, fcdf
        torch.cuda.empty_cache()

    tot = sum(1 for d in done if d is not None)
    gh = sum(1 for d in done if d is not None and d[0] < 0.3)
    bestG = min([d[0] for d in done if d is not None], default=float("nan"))
    print(f"\nARM A (direction-adaptive noise, {ROUNDS} rounds, alpha={ALPHA}): "
          f"{tot}/16 valid escapes, G-region hits (dG<0.3)={gh}, best dG={bestG}", flush=True)
    print("  (noise arm single batch: 3/16; arm B: see its log)", flush=True)
    print(f"  escaped slots (dG,dCloud): {[d for d in done if d is not None]}", flush=True)


if __name__ == "__main__":
    main()
