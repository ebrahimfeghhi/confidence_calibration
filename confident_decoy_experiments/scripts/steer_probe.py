#!/usr/bin/env python
"""Which calling convention does sample_diffusion want for the dumped trunk state?

WHY. Feeding the dump straight into sample_diffusion produced 170,250 non-finite coordinates -- every
value of all 5 samples. The dump squeezed leading 1-dims off z_trunk/s_trunk (so they are [N,N,c] and
[N,c]) but stored relp un-batched as [N,N,139]. prepare_cache concatenates those two, so at inference
they must have had matching rank; batching z alone fails that concat, and un-batching everything
passes the concat but yields NaN. One of the other inputs therefore needs a batch dim, and guessing
has cost three GPU launches already.

This probes the combinations directly with N_step=5 and N_sample=1 -- seconds per variant instead of
a 200-step decode -- and reports which produce finite coordinates. It does not optimise anything.

Usage: steer_probe.py --dump artifacts/dump9x1h/reps/seed300014
"""
import argparse
import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
from steer_reps import build_model, load_dump          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--n-step", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=256)
    A = ap.parse_args()
    dev = "cuda"

    model, configs = build_model(A.ckpt_dir, A.model_name, dev)
    feat, z0, s0, s_inputs, coords = load_dump(A.dump, dev)
    print(f"dump shapes: z {tuple(z0.shape)}  s {tuple(s0.shape)}  "
          f"s_inputs {tuple(s_inputs.shape)}  relp {tuple(feat['relp'].shape)}  "
          f"coords {tuple(coords.shape)}")
    print(f"dump finite? z {torch.isfinite(z0).all().item()}  s {torch.isfinite(s0).all().item()}  "
          f"s_inputs {torch.isfinite(s_inputs).all().item()}  "
          f"coords {torch.isfinite(coords).all().item()}")

    from protenix.model.generator import InferenceNoiseScheduler
    sched = InferenceNoiseScheduler(**configs.inference_noise_scheduler.to_dict())
    ns = sched(N_step=A.n_step, device=dev)

    def attempt(bz, bs, bi, brelp):
        z = z0.unsqueeze(0) if bz else z0
        s = s0.unsqueeze(0) if bs else s0
        si = s_inputs.unsqueeze(0) if bi else s_inputs
        f = dict(feat)
        if brelp:
            f["relp"] = feat["relp"].unsqueeze(0)
        torch.manual_seed(0)
        with torch.no_grad():
            x = model.sample_diffusion(
                denoise_net=model.diffusion_module, input_feature_dict=f,
                s_inputs=si, s_trunk=s, z_trunk=z, pair_z=None, p_lm=None, c_l=None,
                N_sample=1, noise_schedule=ns, inplace_safe=False)
        xa = x.detach().float().cpu().numpy()
        nb = int((~np.isfinite(xa)).sum())
        return xa.shape, nb, (float(np.nanmin(xa)), float(np.nanmax(xa))) if nb < xa.size else None

    print(f"\n{'z':>5} {'s':>5} {'s_in':>5} {'relp':>5}   {'out shape':<22} {'nonfinite':>10}  range")
    for bz, bs, bi, brelp in itertools.product([False, True], repeat=4):
        if bz != brelp:
            continue          # prepare_cache cats z_trunk with relpe(relp): ranks must match
        try:
            shp, nb, rng = attempt(bz, bs, bi, brelp)
            tag = "OK" if nb == 0 else f"{nb}"
            print(f"{str(bz):>5} {str(bs):>5} {str(bi):>5} {str(brelp):>5}   {str(shp):<22} "
                  f"{tag:>10}  {rng}")
        except Exception as e:
            print(f"{str(bz):>5} {str(bs):>5} {str(bi):>5} {str(brelp):>5}   "
                  f"{'EXC':<22} {type(e).__name__}: {str(e)[:70]}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
