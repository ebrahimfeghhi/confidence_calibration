"""Per-loop BASIN trajectory for the slm-nudge / fixed-z0 particles.

Question this answers
---------------------
For the particles that END in the good basin G (M=5-style setup, here 16 nudge
particles), do their distograms already sit in G by loop K=3, or do they start
perturbed (in B1) and only migrate into G at a LATER loop?

Setup is IDENTICAL to pass1_nudge.py (fixed shared z0, each particle a different
5% random nudge on the baseline s_lm, forward-only, 17 loops). The ONLY addition:
at every loop we classify each particle's AB×C cross-block distogram by W1 distance
to the precomputed G / B1 / B2 centroids, so we get a per-loop basin label and the
per-loop W1-to-G. Then we report, for each G-ender, the first loop it entered G.

Saves CSV. Run in tmux gpuN, CUDA_VISIBLE_DEVICES=N.
"""
import sys
import csv
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
INIT_FRAC = 0.05


def readout_logits(model, z, pair_mask):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    return model.distogram_head(z + z.transpose(-2, -3))               # [B,L,L,nbins]


def cross_block(logits):
    P = F.softmax(logits[:, AB, C, :], dim=-1)
    return P.reshape(P.shape[0], -1, P.shape[-1])                      # [B, npairs, nbins]


def w1_to(P, Q):
    """P [B,np,nb], Q [np,nb] centroid -> [B] mean per-pair W1."""
    return (P.cumsum(-1) - Q.unsqueeze(0).cumsum(-1)).abs().sum(-1).mean(-1)


def main():
    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)                    # [np,nb] each
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = base_s_lm.norm().item()
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)

    # FIXED z0 shared across all particles (seed 0) — identical to pass1_nudge.py
    torch.manual_seed(0)
    z0_one = model._init_pair_state(z_init)[0]
    z = z0_one.unsqueeze(0).repeat(B, 1, 1, 1).contiguous().to(z_init.dtype)

    # diversity = a different 5% nudge on s_lm per particle (z0 held fixed)
    s_lm = base_s_lm.repeat(B, 1, 1).clone()
    for i in range(B):
        torch.manual_seed(1000 + i)
        d = torch.randn_like(base_s_lm)
        s_lm[i] = base_s_lm[0] + (d / d.norm() * INIT_FRAC * s0n)[0]
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    rows = []
    print(f"running {B} nudged particles (5%, fixed z0) x {N_LOOPS+1} loops...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for lp in range(N_LOOPS + 1):
                lm_z = lift_to_lm_z(model, s_lm)
                z = _loop_body(model, z, z_init, lm_z, pm, a, b_mat)
                P = cross_block(readout_logits(model, z, pm)).float()  # [B,np,nb]
                wG = w1_to(P, Pg).cpu().numpy()
                wB1 = w1_to(P, Pb1).cpu().numpy()
                wB2 = w1_to(P, Pb2).cpu().numpy()
                labs = np.array(["G", "B1", "B2"])[np.argmin(
                    np.stack([wG, wB1, wB2]), axis=0)]
                for i in range(B):
                    rows.append(dict(run=i, loop=lp,
                                     w1_G=round(float(wG[i]), 5),
                                     w1_B1=round(float(wB1[i]), 5),
                                     w1_B2=round(float(wB2[i]), 5),
                                     basin=labs[i]))
                torch.cuda.empty_cache()
    print(f"forward done {time.time()-t0:.0f}s; peak "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    with open(OUT / "nudge_basin_traj.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["run", "loop", "w1_G", "w1_B1", "w1_B2", "basin"])
        wr.writeheader(); wr.writerows(rows)

    # ---- analysis: who ends in G, and when did they first enter G? ----
    def at(run, lp, key):
        return next(r[key] for r in rows if r["run"] == run and r["loop"] == lp)

    enders = {b: [i for i in range(B) if at(i, N_LOOPS, "basin") == b]
              for b in ("G", "B1", "B2")}
    print(f"\nfinal-loop basin: G={enders['G']}  B1={enders['B1']}  B2={enders['B2']}",
          flush=True)

    print("\nG-enders — first loop entering each basin label, and W1-to-G snap:", flush=True)
    print(f"  {'run':>3} {'firstG':>7} {'traj (loop:basin)':>20}", flush=True)
    for i in enders["G"]:
        traj = [at(i, lp, "basin") for lp in range(N_LOOPS + 1)]
        firstG = next((lp for lp in range(N_LOOPS + 1) if traj[lp] == "G"), None)
        # compact run-length trajectory
        compact, prev = [], None
        for lp, bsn in enumerate(traj):
            if bsn != prev:
                compact.append(f"{lp}:{bsn}"); prev = bsn
        print(f"  {i:>3} {str(firstG):>7}   {' '.join(compact)}", flush=True)
        wg = [at(i, lp, "w1_G") for lp in range(N_LOOPS + 1)]
        print(f"        w1_G by loop: "
              + " ".join(f"{lp}:{wg[lp]:.3f}" for lp in range(N_LOOPS + 1)), flush=True)

    # ---- the headline: basin composition AT loop K=3 vs final ----
    for K in (3, 5, 8, N_LOOPS):
        comp = {b: sum(1 for i in range(B) if at(i, K, "basin") == b)
                for b in ("G", "B1", "B2")}
        gk = [i for i in range(B) if at(i, K, "basin") == "G"]
        print(f"\nloop {K:2d} basin composition: {comp}   (G runs: {gk})", flush=True)
    print("\n--> compare 'G runs at loop 3' vs 'G-enders at loop 16' to see if G is "
          "reached BY K=3 or only LATER.", flush=True)
    print("\nsaved nudge_basin_traj.csv", flush=True)


if __name__ == "__main__":
    main()
