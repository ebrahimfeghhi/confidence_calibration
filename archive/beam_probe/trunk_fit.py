"""Empirical VRAM fit test for the forward-only batched trunk driver (the beam's
core). Builds distinct-z0 particles, per-particle s_lm conditioning (lm_z
recomputed each loop), runs the recycling loop under no_grad, and sweeps batch
size B upward until OOM to find the real ceiling at this L.

Uses n_loops=2 (peak transient recurs every loop, so 2 captures it). Run:
CUDA_VISIBLE_DEVICES=N ESM_PYTHON ... trunk_fit.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confornet"))
from forward_grad import (load, compute_z_init, single_rep_from_hidden, lift_to_lm_z,  # noqa: E402
                          _loop_body, load_hidden, BASELINE)


@torch.no_grad()
def trunk_loops(model, z, z_init, s_lm, pair_mask, a, b_mat, n_loops):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_loops):
            lm_z = lift_to_lm_z(model, s_lm)            # [B,L,L,256] — recomputed per loop
            z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
    return z


def main():
    print("loading model (chunk_size=8)...", flush=True)
    model, builder, feat, chain_infos, cond = load(chunk_size=8)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    base_s_lm = single_rep_from_hidden(model, load_hidden(BASELINE))   # [1,L,256] fp32
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    L, Cdim = z_init.shape[1], z_init.shape[-1]
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"z_init {tuple(z_init.shape)}  pair_mask {tuple(pair_mask.shape)} "
          f"{pair_mask.dtype}  s_lm {tuple(base_s_lm.shape)}  total VRAM {total:.1f} GB",
          flush=True)

    def run_batch(B, n_loops=2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        z = torch.empty(B, L, L, Cdim, device=z_init.device, dtype=z_init.dtype)
        for i in range(B):
            torch.manual_seed(i)
            z[i] = model._init_pair_state(z_init)[0].to(z.dtype)        # distinct z0
        s_lm = base_s_lm.repeat(B, 1, 1) + 0.1 * torch.randn(B, L, Cdim, device=z_init.device)
        pm = pair_mask.expand(B, *pair_mask.shape[1:]).contiguous()
        z = trunk_loops(model, z, z_init, s_lm, pm, a, b_mat, n_loops)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del z, s_lm, pm
        torch.cuda.empty_cache()
        return peak

    best, last_peak = 0, 0.0
    for B in [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
        try:
            t0 = time.time()
            peak = run_batch(B)
            marg = (peak - 0.0)  # rough
            print(f"B={B:4d}  peak={peak:6.1f} GB  ({time.time()-t0:4.0f}s)  fits", flush=True)
            best, last_peak = B, peak
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"B={B:4d}  OOM", flush=True)
                torch.cuda.empty_cache()
                break
            raise
    # crude per-particle marginal from the two largest fitting points isn't tracked here;
    # the sweep itself gives the ceiling.
    print(f"\nLargest fitting batch (n_loops=2, L={L}): B={best}  (peak {last_peak:.1f} GB / "
          f"{total:.0f} GB)", flush=True)


if __name__ == "__main__":
    main()
