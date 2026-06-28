"""Simple batched staged-repulsion (no anchor accumulation).

B particles = baseline s_lm + B different 5% nudges, one shared fixed z0. Run ONE
recycling trajectory; at each checkpoint loop in Ks (default 3,6,9) pause and push
the particles APART (a few Adam steps on s_lm maximizing each particle's cross-block
distogram distance from the other B-1, T=1 truncated: grad through the single loop
at the checkpoint, carry z detached). Then continue the trajectory with the updated
s_lm. After loop 16, decode all B and score vs G/B1/B2 centroids.

Gradient is chunked over particles (detach the "others") so memory is independent of
B: each chunk recomputes only its own checkpoint loop with grad and repels from the
full detached set. Forward-only advance is batched over all B.

Run:  CUDA_VISIBLE_DEVICES=N python repel_staged_batch.py [--B 16 --chunk 4 \
        --Ks 3,6,9 --Mpush 3 --lr 0.3 --frac 0.05 --seed 0 --tag staged]
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, fixed_z0, single_rep_from_hidden,  # noqa: E402
                          lift_to_lm_z, load_hidden, _loop_body, w1, BASELINE, AB, C, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
REAL_G = 0.08


def flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a == name and cast == bool:
            return True
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name and cast != bool:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


def readout_block(model, z, pair_mask):
    """z [b,L,L,C] -> cross-block (AB x C) distogram [b, npairs, nbins]. Differentiable.
    Same block selection/order as forward_grad.cross_distogram_from_z, kept batched."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))      # [b,L,L,nbins]
    P = F.softmax(logits[:, AB, C, :], dim=-1)                  # [b,nAB,nC,nbins]
    return P.reshape(P.shape[0], -1, P.shape[-1])               # [b,npairs,nbins]


def mean_pairwise_w1(P):                                        # [b,np,nb] -> scalar
    cdf = P.cumsum(-1)
    b = P.shape[0]
    tot, n = 0.0, 0
    for i in range(b):
        d = (cdf[i:i + 1] - cdf[i + 1:]).abs().sum(-1).mean(-1)   # [b-i-1]
        tot += float(d.sum()); n += d.numel()
    return tot / max(n, 1)


def main():
    B = flag("--B", 16, int)
    chunk = flag("--chunk", 4, int)
    Ks = set(int(x) for x in flag("--Ks", "3,6,9", str).split(","))
    Mpush = flag("--Mpush", 3, int)
    lr = flag("--lr", 0.3, float)
    frac = flag("--frac", 0.05, float)
    seed = flag("--seed", 0, int)
    tag = flag("--tag", "staged", str)
    print(f"config: B={B} chunk={chunk} Ks={sorted(Ks)} Mpush={Mpush} lr={lr} "
          f"frac={frac} seed={seed} tag={tag}", flush=True)

    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)

    base = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()   # [1,L,256]
    s0n = base.norm().item()

    z0 = fixed_z0(model, z_init, seed=seed)                     # [1,L,L,Cdim]
    z = z0.repeat(B, 1, 1, 1).contiguous().to(z_init.dtype)
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    s_lm = base.repeat(B, 1, 1).clone()
    for i in range(B):
        torch.manual_seed(1000 + 100 * seed + i)
        d = torch.randn_like(base)
        s_lm[i] = base[0] + (d / d.norm() * frac * s0n)[0]
    s_lm = s_lm.detach().requires_grad_(True)
    opt = torch.optim.Adam([s_lm], lr=lr)

    torch.cuda.reset_peak_memory_stats()
    spread_log = []
    t0 = time.time()
    print(f"running 1 trajectory, {N_LOOPS+1} loops, push at {sorted(Ks)}...", flush=True)
    for lp in range(N_LOOPS + 1):
        if lp in Ks:
            z_prev = z.detach()
            pre = post = None
            for ps in range(Mpush):
                # detached targets at current s_lm, computed in CHUNKS (avoid the
                # B-wide forward peak stacking on the carried state + grad graph)
                P_parts = []
                with torch.no_grad():
                    for c0 in range(0, B, chunk):
                        sl = slice(c0, min(c0 + chunk, B))
                        lm_t = lift_to_lm_z(model, s_lm[sl])
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            z_t = _loop_body(model, z_prev[sl], z_init, lm_t, pm[sl], a, b_mat)
                        P_parts.append(readout_block(model, z_t, pm[sl]).detach())
                        del z_t, lm_t
                        torch.cuda.empty_cache()
                P_all = torch.cat(P_parts, 0)                            # [B,np,nb]
                del P_parts
                if ps == 0:
                    pre = mean_pairwise_w1(P_all)
                cdf_all = P_all.cumsum(-1)                                # [B,np,nb]
                opt.zero_grad()
                for c0 in range(0, B, chunk):
                    sl = slice(c0, min(c0 + chunk, B))
                    lm_c = lift_to_lm_z(model, s_lm[sl])
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        z_c = checkpoint(_loop_body, model, z_prev[sl], z_init, lm_c,
                                         pm[sl], a, b_mat, use_reentrant=False)   # ckpt = fit grad
                    P_c = readout_block(model, z_c, pm[sl])              # [c,np,nb] grad
                    cdf_c = P_c.cumsum(-1)
                    dist = (cdf_c[:, None] - cdf_all[None]).abs().sum(-1).mean(-1)   # [c,B]
                    for r, gi in enumerate(range(c0, min(c0 + chunk, B))):
                        dist[r, gi] = 0.0                                # exclude self
                    md = dist.sum(1) / (B - 1)                           # [c] mean over others
                    (-md.sum()).backward()
                    del P_c, z_c, cdf_c, dist, md
                opt.step()
                torch.cuda.empty_cache()
            # post-push spread + advance trajectory with updated s_lm (forward-only)
            with torch.no_grad():
                lm_all = lift_to_lm_z(model, s_lm)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = _loop_body(model, z_prev, z_init, lm_all, pm, a, b_mat)
                post = mean_pairwise_w1(readout_block(model, z, pm))
            spread_log.append((lp, pre, post))
            print(f"  loop {lp:2d} push: spread {pre:.4f} -> {post:.4f}", flush=True)
        else:
            with torch.no_grad():
                lm_all = lift_to_lm_z(model, s_lm)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = _loop_body(model, z, z_init, lm_all, pm, a, b_mat)
        torch.cuda.empty_cache()

    # ---- decode + score all B at the final state ----
    with torch.no_grad():
        P_fin = readout_block(model, z, pm)                              # [B,np,nb]
    rows, realg = [], 0
    for i in range(B):
        wg = float(w1(P_fin[i], Pg)); wb1 = float(w1(P_fin[i], Pb1)); wb2 = float(w1(P_fin[i], Pb2))
        basin = min([("G", wg), ("B1", wb1), ("B2", wb2)], key=lambda x: x[1])[0]
        realg += wg < REAL_G
        rows.append(dict(particle=i, basin=basin, wg=round(wg, 4),
                         wb1=round(wb1, 4), wb2=round(wb2, 4)))
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    with open(OUT / f"staged_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["particle", "basin", "wg", "wb1", "wb2"])
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    dist = Counter(r["basin"] for r in rows)
    best = min(rows, key=lambda r: r["wg"])
    print(f"\n[{tag}] done {dt:.0f}s, peak {peak:.1f} GB. basins={dict(dist)} "
          f"REAL_G(wg<{REAL_G})={realg}/{B}  best wg={best['wg']} (particle {best['particle']}, "
          f"basin {best['basin']})", flush=True)
    print("spread (pre->post) by checkpoint:",
          " ".join(f"L{lp}:{pre:.3f}->{post:.3f}" for lp, pre, post in spread_log), flush=True)
    print("saved", OUT / f"staged_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
