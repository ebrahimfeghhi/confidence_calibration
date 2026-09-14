#!/usr/bin/env python
"""Smoke test for the _trunk_override decode path.

THE FAILURE THIS EXISTS TO CATCH: handing the diffusion module the WRONG pair representation --
z_pre (before the final pairformer stack) instead of z (after). That runs perfectly happily and
produces confidently wrong structures, so it would silently waste the entire sweep.

METHOD: run the trunk, take the stop-hook state, run the final pairformer stack OURSELVES with NO
affine, and decode through _trunk_override. Then decode the same complex through the ordinary
runner.predict path at the same seed. If the override is faithful the two must agree to CUDA noise.

Also asserts the hand-prepped feature dict has the same keys forward() would have produced.

Usage: .../python scripts/smoke_override.py --pdb 8qf4 --gpu 3
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confornet_protenix as C

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", default="8qf4")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--seed", type=int, default=7)
A = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)
os.environ["LAYERNORM_TYPE"] = "openfold"

out = f"{C.EXP}/artifacts/confornet/_smoke"
os.makedirs(out, exist_ok=True)
runner, _ = C.build_runner(A.pdb, f"{C.JDIR}/{A.pdb}-final-updated.json",
                           f"{out}/decoded", A.seed, 200, 2)
model = runner.model
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

data, atom_array = C.get_batch(runner.configs)
from protenix.utils.torch_utils import to_device
data = to_device(data, runner.device)

# ---- feature-prep assert -------------------------------------------------------------------------
before = set(data["input_feature_dict"].keys())
from protenix.model.protenix import update_input_feature_dict
data["input_feature_dict"] = update_input_feature_dict(
    model.relative_position_encoding.generate_relp(data["input_feature_dict"]))
added = set(data["input_feature_dict"].keys()) - before
expect = {"relp", "d_lm", "v_lm", "pad_info"}
assert added == expect, f"feature prep added {added}, expected {expect}"
print(f"feature-prep assert OK: added {sorted(added)}", flush=True)

from runner.inference import update_inference_configs
n_token = int(data["N_token"].item())
runner.update_model_configs(update_inference_configs(runner.configs, n_token))
n_cycle = model.configs.model.N_cycle
chunk = None
print(f"{A.pdb}: N_token={n_token} N_cycle={n_cycle}", flush=True)


def coords(pred):
    return pred["coordinate"].float().cpu().numpy()


# ---- A: ordinary path ------------------------------------------------------------------------------
from protenix.utils.seed import seed_everything
seed_everything(seed=A.seed, deterministic=False)
ref = coords(runner.predict(copy.deepcopy(data)))
print(f"ordinary runner.predict : coords {ref.shape}", flush=True)

# ---- B: override path, NO affine -------------------------------------------------------------------
# Same seed, so the trunk draw matches A. Identity in place of the affine, so the only difference
# between the two is WHERE the pairformer stack was executed from.
seed_everything(seed=A.seed, deterministic=False)
s_inputs, s_pre, z_pre, drop = C.run_trunk_prefix(
    runner, data, n_cycle, chunk, A.seed, runner.configs.mc_dropout_apply_rate)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    s_out, z_out = model.pairformer_stack(
        s_pre, z_pre, pair_mask=None,
        triangle_multiplicative=model.configs.triangle_multiplicative,
        triangle_attention=model.configs.triangle_attention,
        inplace_safe=False, chunk_size=chunk)
model._trunk_override = (s_inputs, s_out.detach(), z_out.detach())
try:
    with torch.no_grad():
        ovr = coords(runner.predict(copy.deepcopy(data)))
finally:
    model._trunk_override = None
print(f"override runner.predict : coords {ovr.shape}  (mc_dropout={drop})", flush=True)

# ---- verdict ---------------------------------------------------------------------------------------
n = min(ref.shape[-2], ovr.shape[-2])
d = np.abs(ref[..., :n, :] - ovr[..., :n, :])
print()
print(f"mean |delta| = {d.mean():8.4f} A")
print(f"max  |delta| = {d.max():8.4f} A")
print()
if d.mean() < 1.0:
    print("PASS: override reproduces the ordinary path -> the diffusion module is receiving the")
    print("      post-pairformer z, and the plumbing is faithful.")
else:
    print("FAIL: override diverges from the ordinary path. Most likely the wrong tensor is being")
    print("      handed over (z_pre instead of z), or the trunk draw did not match. DO NOT SWEEP.")
