#!/usr/bin/env python
"""LoRA on the pairformer: make an UNCOMMITTED trunk state commit to one of its own poses.

THE TARGET. 9x1h_EF seed 300014 holds several docking arrangements at once -- its five diffusion
samples span DockQ 0.044-0.634 from a single frozen pair rep, while ipTM spans only 0.117-0.151 and
cannot tell them apart. We optimise toward sample 3 (DockQ 0.634), a pose THIS SEED ALREADY PRODUCES,
so the experiment resolves a superposition rather than implanting a belief the model never held.

WHY LoRA ON THE PAIRFORMER RATHER THAN EDITING z DIRECTLY. Optimising z as a free tensor will find
some tensor that satisfies the loss, but almost certainly one the pairformer could never emit, and
nothing learned about it would transfer. Here the adapters are trained and z is RE-DERIVED by the
pairformer, so the result is a pairformer output by construction -- on-manifold, and pair-specific
(triangle attention can move an individual contact, which a global channel affine cannot).

This runs from the dump alone: protenix.py:364 calls pairformer_stack(s, z, ...) with no MSA and no
template features, and both s and z are in the dump. The MSA module is NOT re-run, so this is one
pairformer pass rather than a full recycle -- the same injection point the ConforNet hook uses two
lines above that call.

48 NOISE SAMPLES, CHUNKED. Protenix trains the diffusion module with diffusion_batch_size=48 and
diffusion_chunk_size=4 (configs_base.py:122). Matching that matters here specifically: z dominates
the trajectory at HIGH noise, which is where the dock is chosen, and 4 draws barely sample that
regime. Chunks are accumulated through a detached z_new/s_new and then backpropagated into the
adapters in ONE pairformer backward -- so the 48 samples cost memory like 4, not like 48.

DECODE EVERY --decode-every STEPS with the real 200-step sampler and the confidence head, at a fixed
noise seed so checkpoints are comparable. Recorded per decode: RMSD of each sample to the target, the
spread across samples (the "did it commit" number), and ipTM (does manufactured commitment to a GOOD
pose raise confidence the way commitment to a decoy does?).

Usage:
  steer_lora.py --dump artifacts/dump9x1h/reps/seed300014 --target-sample 3 \
      --steps 100 --decode-every 5 --lora-rank 8 --out artifacts/steerlora9x1h
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
from steer_reps import build_model, load_dump          # noqa: E402  (shared, already verified)


class LoRAHook(nn.Module):
    """out <- out + (x @ A.T) @ B.T * (alpha/r).  B is zero-init so the adapter starts as identity.

    Attached with register_forward_hook so the base module is untouched and can be restored; the
    base weight stays frozen and only A/B are optimised.
    """

    def __init__(self, lin, r, alpha, device):
        super().__init__()
        fi, fo = lin.in_features, lin.out_features
        self.A = nn.Parameter(torch.randn(r, fi, device=device) * (1.0 / fi ** 0.5))
        self.B = nn.Parameter(torch.zeros(fo, r, device=device))
        self.scale = alpha / r

    def __call__(self, module, inp, out):
        x = inp[0]
        return out + ((x.to(self.A.dtype) @ self.A.T) @ self.B.T) * self.scale


def attach_lora(stack, rank, alpha, device, name_filter=""):
    hooks, mods = [], []
    for name, m in stack.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if name_filter and name_filter not in name:
            continue
        h = LoRAHook(m, rank, alpha, device)
        m.register_forward_hook(h)
        hooks.append(h); mods.append(name)
    return hooks, mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--target-sample", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--decode-every", type=int, default=5)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--lora-filter", default="", help="substring filter on module names")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="pairformer attention chunk; inference uses 256-512 at this token count. "
                         "None materialises the full triangle attention and OOMs at 1474 tokens.")
    ap.add_argument("--blocks-per-ckpt", type=int, default=1,
                    help="gradient checkpointing depth in the pairformer")
    ap.add_argument("--n-sample-loss", type=int, default=48, help="matches diffusion_batch_size")
    ap.add_argument("--chunk", type=int, default=4, help="matches diffusion_chunk_size")
    ap.add_argument("--n-sample-decode", type=int, default=5)
    ap.add_argument("--ref-dump", default="")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    model, configs = build_model(A.ckpt_dir, A.model_name, dev,
                                 blocks_per_ckpt=A.blocks_per_ckpt)
    feat, z0, s0, s_inputs, coords = load_dump(A.dump, dev)
    tgt = coords[A.target_sample]
    z0b, s0b = z0, s0          # dump is un-batched throughout; see prepare_cache cat
    zn0, sn0 = z0.norm().item(), s0.norm().item()
    print(f"tokens {z0.shape[0]}  atoms {tgt.shape[0]}  target = sample {A.target_sample}")

    if A.ref_dump:
        _, zr, sr, _, _ = load_dump(A.ref_dump, dev)
        print(f"YARDSTICK two real trunk states: ||dz||/||z|| {(z0-zr).norm().item()/zn0:.4f}   "
              f"||ds||/||s|| {(s0-sr).norm().item()/sn0:.4f}")
        del zr, sr
        torch.cuda.empty_cache()

    hooks, names = attach_lora(model.pairformer_stack, A.lora_rank, A.lora_alpha, dev, A.lora_filter)
    params = [p for h in hooks for p in h.parameters()]
    print(f"LoRA on {len(names)} Linear modules in pairformer_stack, rank {A.lora_rank}: "
          f"{sum(p.numel() for p in params):,} trainable params")
    opt = torch.optim.Adam(params, lr=A.lr)

    from protenix.model.generator import (sample_diffusion_training, TrainingNoiseSampler,
                                          InferenceNoiseScheduler)
    noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler.to_dict())
    sched = InferenceNoiseScheduler(**configs.inference_noise_scheduler.to_dict())
    noise_schedule = sched(N_step=configs.sample_diffusion.N_step, device=dev)
    label = {"coordinate": tgt,
             "coordinate_mask": torch.ones(tgt.shape[0], device=dev)}
    sd2 = configs.sigma_data ** 2
    tnp = tgt.cpu().numpy()

    def pairformer():
        return model.pairformer_stack(
            s0b, z0b, pair_mask=None,
            triangle_multiplicative=configs.triangle_multiplicative,
            triangle_attention=configs.triangle_attention,
            inplace_safe=False, chunk_size=A.chunk_size)

    def diffusion_loss(zc, sc, n):
        _, x_den, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler, denoise_net=model.diffusion_module,
            label_dict=label, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=sc, z_trunk=zc,
            pair_z=None, p_lm=None, c_l=None, N_sample=n)
        w = (sigma ** 2 + sd2) / ((sigma * configs.sigma_data) ** 2)
        mse = ((x_den - tgt.unsqueeze(0)) ** 2).sum(-1).mean(-1)
        return (w * mse).mean()

    def decode(step, zc, sc):
        torch.manual_seed(1234)                       # identical noise draws at every decode
        with torch.no_grad():
            x = model.sample_diffusion(
                denoise_net=model.diffusion_module, input_feature_dict=feat,
                s_inputs=s_inputs, s_trunk=sc, z_trunk=zc, pair_z=None,
                p_lm=None, c_l=None, N_sample=A.n_sample_decode,
                noise_schedule=noise_schedule, inplace_safe=False)
            conf = model.run_confidence_head(
                input_feature_dict=feat, s_inputs=s_inputs, s_trunk=sc,
                z_trunk=zc, pair_mask=None, x_pred_coords=x,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                chunk_size=A.chunk_size)
        xa = x.detach().float().cpu().numpy()
        xa = xa.reshape(-1, xa.shape[-2], xa.shape[-1])
        # Save BEFORE scoring: an earlier attempt died in the RMSD maths and lost the coordinates,
        # which are the expensive part. Diagnostics first, because "SVD did not converge" on a 3x3
        # means NaN/Inf upstream, not a numerical edge case.
        np.savez(f"{A.out}/decode{step:04d}.npz", coordinate=xa)
        nbad = int((~np.isfinite(xa)).sum())
        print(f"   [decode {step:>4}] coords {xa.shape}  nonfinite {nbad}  "
              f"range [{np.nanmin(xa):.1f},{np.nanmax(xa):.1f}]")
        if nbad:
            print(f"   [decode {step:>4}] SKIPPED RMSD -- {nbad} non-finite coordinates")
            return dict(step=step, rmsd=None, rmsd_mean=None, rmsd_spread=None, nonfinite=nbad)
        rms = []
        b = tnp - tnp.mean(0)
        for k in range(xa.shape[0]):                  # superpose out the sampler's free global frame
            a = xa[k] - xa[k].mean(0)
            try:
                U, _, Vt = np.linalg.svd((a.T @ b).astype(np.float64))
                R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                rms.append(float(np.sqrt(((a @ R - b) ** 2).sum(1).mean())))
            except np.linalg.LinAlgError as e:
                print(f"   [decode {step:>4}] sample {k}: superposition failed ({e})")
                rms.append(float("nan"))
        np.savez(f"{A.out}/decode{step:04d}.npz", coordinate=xa, rmsd=np.array(rms))
        try:
            torch.save({k: v.cpu() for k, v in (conf[0] if isinstance(conf, tuple) else conf).items()
                        if torch.is_tensor(v)}, f"{A.out}/conf{step:04d}.pt")
        except Exception:
            pass
        print(f"   [decode {step:>4}] RMSD to target mean {np.nanmean(rms):6.2f} A  "
              f"spread {np.nanmax(rms)-np.nanmin(rms):5.2f}  per-sample {[round(v,1) for v in rms]}")
        return dict(step=step, rmsd=rms, rmsd_mean=float(np.nanmean(rms)),
                    rmsd_spread=float(np.nanmax(rms) - np.nanmin(rms)))

    # ---- reference decode: the RAW dumped trunk state, no pairformer pass at all ----------------
    # Step 0 is NOT the original behaviour: the pairformer is residual, so one extra pass moves z even
    # with the adapters at zero. Without this reference we could not separate "steering worked" from
    # "an eleventh pairformer pass sharpened things". Saved as step -1.
    hist = [decode(-1, z0b, s0b)]
    hist[-1].update(rel_z=0.0, rel_s=0.0, note="raw dumped z/s, no pairformer pass")
    print(f"   [ref   raw] this is the seed's original behaviour; step 0 adds one pairformer pass")
    json.dump(hist, open(f"{A.out}/history.json", "w"), indent=1)

    t0 = time.time()
    for step in range(A.steps + 1):
        # ---- one pairformer pass; z/s are re-derived by the model, not edited ------------------
        s_new, z_new = pairformer()
        if step % A.decode_every == 0:
            with torch.no_grad():
                rz = (z_new - z0b).norm().item() / zn0
                rs = (s_new - s0b).norm().item() / sn0
            d = decode(step, z_new.detach(), s_new.detach())
            d.update(rel_z=rz, rel_s=rs)
            hist.append(d)
            json.dump(hist, open(f"{A.out}/history.json", "w"), indent=1)
            torch.save({f"h{i}": h.state_dict() for i, h in enumerate(hooks)},
                       f"{A.out}/lora{step:04d}.pt")
            print(f"   [ckpt  {step:>4}] ||dz||/||z|| {rz:.5f}  ||ds||/||s|| {rs:.5f}")
        if step == A.steps:
            break

        # ---- 48 noise draws in chunks, accumulated through a detached z/s ----------------------
        # The gradient is collected on the detached copies first, then pushed through the pairformer
        # ONCE. Backpropagating each chunk through the pairformer instead would cost 12x the time and
        # require retaining its graph the whole way.
        zd = z_new.detach().requires_grad_(True)
        sd = s_new.detach().requires_grad_(True)
        tot, nchunk = 0.0, 0
        for i in range(0, A.n_sample_loss, A.chunk):
            n = min(A.chunk, A.n_sample_loss - i)
            loss = diffusion_loss(zd, sd, n) * (n / A.n_sample_loss)
            loss.backward()
            tot += loss.item(); nchunk += 1
        opt.zero_grad(set_to_none=True)
        torch.autograd.backward([z_new, s_new], [zd.grad, sd.grad])
        opt.step()
        del zd, sd, z_new, s_new
        if step % 5 == 0:
            print(f"   step {step:>4}  loss {tot:.4f}  "
                  f"({(time.time()-t0)/max(step,1):.0f}s/step)  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    json.dump(hist, open(f"{A.out}/history.json", "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> {A.out}")


if __name__ == "__main__":
    main()
