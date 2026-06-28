"""Fresh-vs-persistent noise on s_lm, batched B=16 forward-only (no gradient).

Idea (user): instead of a single persistent s_lm perturbation (old noise_arm: gets
re-absorbed), add FRESH Gaussian noise to s_lm at EVERY recycling loop (like dropout,
which is load-bearing). Run B particles in parallel, fixed shared z0, and see how many
land in G / how the basins spread. Sweep magnitude; include a persistent control.

Saves:
  noise_fresh_scores.csv          (arm, frac, particle, basin, wg, wb1, wb2)
  noise_fresh_distograms.pt       fresh arm only: {f"frac{f}_k3": P[B,np,nb] fp16,
                                                   f"frac{f}_final": P[B,np,nb] fp16}
  K3 = distogram after 3 loops; final = after N_LOOPS+1 loops.

Run: CUDA_VISIBLE_DEVICES=N python noise_fresh_batch.py [--B 16 --fracs 0.05,0.1,0.25,0.5,1.0]
"""
import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, fixed_z0, single_rep_from_hidden,  # noqa: E402
                          load_hidden, lift_to_lm_z, _loop_body, w1, BASELINE, AB, C, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
K3 = 3  # "K=3" save point: distogram after 3 loops


def flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


@torch.no_grad()
def readout_block(model, z, pair_mask):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))
    P = F.softmax(logits[:, AB, C, :], dim=-1)
    return P.reshape(P.shape[0], -1, P.shape[-1])               # [b,npairs,nbins]


def main():
    B = flag("--B", 16, int)
    fracs = [float(x) for x in flag("--fracs", "0.05,0.1,0.25,0.5,1.0", str).split(",")]
    seed0 = flag("--seed", 0, int)
    print(f"config: B={B} fracs={fracs} z0_seed={seed0} (fresh + persistent arms)", flush=True)

    print("loading...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)

    base = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()    # [1,L,256]
    s0n = base.norm().item()
    z0 = fixed_z0(model, z_init, seed=seed0)                                # [1,L,L,C]
    z0B = z0.repeat(B, 1, 1, 1).contiguous().to(z_init.dtype)
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()

    def noisy_slm(frac, loop, persistent_cache=None):
        """B particles = base + fresh per-particle unit-noise * frac*s0n. Reproducible."""
        if persistent_cache is not None and loop > 0:
            return persistent_cache              # persistent: reuse loop-0 draw
        torch.manual_seed(424242 + 1000 * int(frac * 1000) + loop)
        d = torch.randn(B, *base.shape[1:], device=base.device, dtype=base.dtype)
        d = d / d.flatten(1).norm(dim=1).view(B, 1, 1)           # per-particle unit
        return base + d * frac * s0n                             # [B,L,256]

    @torch.no_grad()
    def run_arm(frac, fresh):
        z = z0B.clone()
        pers = None
        saved = {}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for lp in range(N_LOOPS + 1):
                if fresh:
                    s_lm = noisy_slm(frac, lp)
                else:
                    if pers is None:
                        pers = noisy_slm(frac, 0)
                    s_lm = pers
                lm = lift_to_lm_z(model, s_lm)
                z = _loop_body(model, z, z_init, lm, pm, a, b_mat)
                if lp + 1 == K3:                                 # after 3 loops
                    saved["k3"] = readout_block(model, z, pm).half().cpu()
        P_fin = readout_block(model, z, pm)
        saved["final"] = P_fin.half().cpu()
        rows = []
        for i in range(B):
            wg, wb1, wb2 = w1(P_fin[i], Pg).item(), w1(P_fin[i], Pb1).item(), w1(P_fin[i], Pb2).item()
            basin = min([("G", wg), ("B1", wb1), ("B2", wb2)], key=lambda x: x[1])[0]
            rows.append(dict(arm="fresh" if fresh else "persist", frac=frac, particle=i,
                             basin=basin, wg=round(wg, 4), wb1=round(wb1, 4), wb2=round(wb2, 4)))
        return rows, saved

    all_rows, disto = [], {}
    t0 = time.time()
    for frac in fracs:
        for fresh in (True, False):
            rows, saved = run_arm(frac, fresh)
            all_rows.extend(rows)
            nG = sum(r["basin"] == "G" for r in rows)
            nRG = sum(r["wg"] < 0.08 for r in rows)
            dist = {bs: sum(r["basin"] == bs for r in rows) for bs in ("G", "B1", "B2")}
            print(f"  frac={frac:<5} {'fresh' if fresh else 'persist':>7}: basins={dist} "
                  f"realG(wg<.08)={nRG}/{B}  bestwg={min(r['wg'] for r in rows):.3f}", flush=True)
            if fresh:                                            # save distograms for fresh arm
                disto[f"frac{frac}_k3"] = saved["k3"]
                disto[f"frac{frac}_final"] = saved["final"]
            with open(OUT / "noise_fresh_scores.csv", "w", newline="") as f:   # incremental flush
                w = csv.DictWriter(f, fieldnames=["arm", "frac", "particle", "basin", "wg", "wb1", "wb2"])
                w.writeheader(); w.writerows(all_rows)
            torch.cuda.empty_cache()
    torch.save(disto, OUT / "noise_fresh_distograms.pt")
    sz = (OUT / "noise_fresh_distograms.pt").stat().st_size / 1e6
    print(f"\ndone {time.time()-t0:.0f}s. wrote noise_fresh_scores.csv + "
          f"noise_fresh_distograms.pt ({sz:.0f} MB, {len(disto)} tensors)", flush=True)


if __name__ == "__main__":
    main()
