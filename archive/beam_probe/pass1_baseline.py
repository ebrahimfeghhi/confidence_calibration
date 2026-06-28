"""Pass 1 of the two-pass plan: run 16 particles with identical no-mask
conditioning + no dropout, varying ONLY z0, full 17 loops, forward-only. Capture
per-loop cross-block distograms -> define the B1 baseline (per-loop medoid +
wobble threshold). Classify the 16 at loop 16 vs existing G/B1/B2 centroids to
confirm they land in the default basin.

Saves: per-loop stats CSV, per-loop medoid distograms, loop16 distograms, a plot.
Run: CUDA_VISIBLE_DEVICES=7 ESM_PYTHON ... pass1_baseline.py
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
NPAIRS = (AB.stop - AB.start) * (C.stop - C.start)


def cross_distogram_batched(model, z, pair_mask):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))           # [B,L,L,nbins]
    P = F.softmax(logits, dim=-1)[:, AB, C, :].reshape(z.shape[0], -1, logits.shape[-1])
    return P                                                          # [B,npairs,nbins]


@torch.no_grad()
def run_capture(model, z, z_init, s_lm, pm, a, b_mat, n_loops):
    distos = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n_loops):
            lm_z = lift_to_lm_z(model, s_lm)
            z = _loop_body(model, z, z_init, lm_z, pm, a, b_mat)
            P = cross_distogram_batched(model, z, pm)
            distos.append(P.half().cpu())
            del P, lm_z
            t = time.time()
    return z, distos


def pairwise_w1(V):                                                   # V [n,D] CDF (GPU)
    n = V.shape[0]
    W = torch.zeros(n, n)
    for i in range(n):
        W[i] = (V - V[i]).abs().sum(1).cpu() / NPAIRS
    return W.numpy()


def main():
    print("loading model + centroids...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE))  # no-mask conditioning
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]

    z = torch.empty(B, L, L, Cdim, device=z_init.device, dtype=z_init.dtype)
    for i in range(B):
        torch.manual_seed(i)
        z[i] = model._init_pair_state(z_init)[0].to(z.dtype)         # distinct z0, no perturbation
    s_lm = base_s_lm.repeat(B, 1, 1)                                  # identical conditioning, no mask
    pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()
    print(f"running {B} particles x {N_LOOPS+1} loops (z0-only)...", flush=True)
    t0 = time.time()
    z, distos = run_capture(model, z, z_init, s_lm, pm, a, b_mat, N_LOOPS + 1)
    torch.cuda.synchronize()
    print(f"forward done in {time.time()-t0:.0f}s; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
          flush=True)
    del z
    torch.cuda.empty_cache()

    distos = torch.stack(distos)                                      # [17, B, npairs, nbins] fp16 CPU

    # per-loop baseline stats
    rows, medoid_distos = [], []
    for lp in range(distos.shape[0]):
        V = distos[lp].cuda().float()                                # [B,npairs,nbins]
        cdf = V.cumsum(-1).reshape(B, -1)
        Wm = pairwise_w1(cdf)
        med = int(Wm.sum(1).argmin())
        d = Wm[med]
        d_others = np.delete(d, med)
        mu, sig, mx = float(d_others.mean()), float(d_others.std()), float(d_others.max())
        rows.append(dict(loop=lp, medoid=med, mu=round(mu, 5), sigma=round(sig, 5),
                         max_dist=round(mx, 5), thr_3sigma=round(mu + 3 * sig, 5),
                         thr_max=round(mx, 5)))
        medoid_distos.append(distos[lp, med].numpy())
        del V, cdf
        torch.cuda.empty_cache()

    # classify the 16 at loop 16 vs existing centroids
    print("\nloop-16 classification of the 16 z0-only runs:", flush=True)
    V16 = distos[-1].cuda().float()
    counts = {"G": 0, "B1": 0, "B2": 0}
    for i in range(B):
        P = V16[i]
        ds = {"G": w1(P, Pg).item(), "B1": w1(P, Pb1).item(), "B2": w1(P, Pb2).item()}
        lab = min(ds, key=ds.get)
        counts[lab] += 1
        print(f"  run {i:2d}: nearest={lab}  (G {ds['G']:.3f}  B1 {ds['B1']:.3f}  B2 {ds['B2']:.3f})",
              flush=True)
    print(f"  --> counts: {counts}", flush=True)

    with open(OUT / "pass1_baseline_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(OUT / "pass1_medoid_distograms.npy", np.stack(medoid_distos).astype(np.float16))
    np.save(OUT / "pass1_loop16_distograms.npy", distos[-1].numpy().astype(np.float16))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lp = [r["loop"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(lp, [r["mu"] for r in rows], "-o", label="mean within-basin W1 (wobble)")
    plt.fill_between(lp, [r["mu"] - r["sigma"] for r in rows], [r["mu"] + r["sigma"] for r in rows],
                     alpha=0.2)
    plt.plot(lp, [r["thr_3sigma"] for r in rows], "--", label="threshold mean+3sigma")
    plt.plot(lp, [r["max_dist"] for r in rows], ":", label="max within-basin")
    plt.xlabel("loop"); plt.ylabel("W1 to medoid")
    plt.title("Pass-1 baseline (z0-only) wobble by loop"); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pass1_baseline.png", dpi=130)
    print("\nsaved pass1_baseline_stats.csv, medoid/loop16 distograms, pass1_baseline.png", flush=True)


if __name__ == "__main__":
    main()
