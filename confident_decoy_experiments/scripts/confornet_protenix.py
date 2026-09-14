#!/usr/bin/env python
"""ConforNets (arXiv 2604.18559) ported from OpenFold3 to Protenix.

WHAT A CONFORNET IS: one learnable affine on the pair representation, z @ W.T + b, with W a c_z x c_z
matrix (128x128 here) initialised to the identity. ~16.5k parameters. It has no spatial structure --
the SAME channel mixing is applied at every residue pair (i,j) -- so the open question this script
exists to answer is whether a global channel affine can relocate a discrete docking pose, or whether
it can only blur the fold. The reference benchmarks are membrane transporters and fold-switchers,
i.e. large INTRA-chain conformational change, never inter-chain docking.

WHERE IT IS INJECTED: at the last recycle, immediately before the final pairformer stack (the
reference implementation's PAIR_PRE mode). Every earlier recycle runs under no_grad, so the whole
training graph is one pairformer stack. See the hooks in protenix/model/protenix.py.

A FRESH TRUNK EVERY STEP: MSA subsampling (min_size.test=1, so sample_size ~ randint(1, n+1) redrawn
every recycle) and MC dropout (p=0.4 on the recycling carry-over, rolled once per forward pass) are
both stock Protenix inference behaviour, and both stay ON here. So the trunk is re-run every
optimisation step and each step sees a DIFFERENT z_pre. The ConforNet is therefore fit to the
distribution of trunk states the model actually produces, not to one draw from it -- which is also
what makes the training distribution match the decode distribution, since decoding is stochastic in
exactly the same way. Within a step the single trunk is shared across all k ConforNets, so the affine
is the only thing separating them.

This costs roughly 10x a cached step: 9 full recycles under no_grad, then the 10th recycle's
pre-pairformer work, then one grad-enabled final stack per ConforNet. Nothing is wasted -- the stop
hook returns before the final pairformer, so the stack we backprop through is the one the trunk would
have run anyway. --fixed-trunk restores the cache-once behaviour for a cheap smoke test, but a
ConforNet trained that way is fit to a single trunk draw and should not be trusted to transfer.

OBJECTIVE (dist_cdf_mse): distogram logits -> softmax -> cumsum over bins -> CDF, then maximise mean
pairwise MSE between the k CDFs. Label-free. Chosen over the reference's coord_mse because it needs
no diffusion rollout in the graph, which is both cheaper and lower-memory. Loss is reported NEGATIVE
(it is minimised), so a run that is working shows loss going DOWN and pairwise_cdf_mse going UP.

SEEDING: each step reseeds from (seed, step) so the trunk draw is a reproducible function of the step
index rather than of how many CUDA calls happened to precede it. CUDA is not bit-reproducible anyway,
so this makes the MSA/dropout draws replayable, not the outputs.

MEMORY: the reference warns ~600 aa on an 80 GB card because OpenFold3 backprops through 48
pairformer blocks with no checkpointing. Protenix's PairformerStack honours blocks_per_ckpt whenever
grad is enabled, and --blocks-per-ckpt (default 1) turns that on. 8che is 942 tokens, so this is the
lever that decides whether it fits at all. UNMEASURED -- run --probe-memory first.

Usage (nothing here runs a sweep; one complex, one seed, one optimisation):
  .../python scripts/confornet_protenix.py --pdb 8che --gpu 3 --k 2 --steps 10 --probe-memory
"""
import argparse
import copy
import csv
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

PROTENIX_ROOT = "/home/ebrahim/Protenix"
JDIR = f"{PROTENIX_ROOT}/foldbench_abag/jsons"


# --------------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------------
class ConforNet(nn.Module):
    """Learnable affine x @ W.T + b, initialised to identity. Verbatim port of confornet/core/confornet.py."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32, NOT bf16. An earlier version cast W to x.dtype to fix a matmul dtype error, which also
        # rounded W to bf16 on every forward. Near the identity diagonal's 1.0 the bf16 step is
        # 2^-7 = 0.0078, while an Adam update at lr=1e-3 is ~1e-3 -- so diagonal learning was quantised
        # away. Measured on a completed run: 39.5% of diagonal entries moved LESS than one bf16 tick
        # across the entire optimisation, i.e. the affine could learn channel MIXING (off-diagonal,
        # magnitude ~0.01 where the bf16 step is ~3e-5) but was frozen on channel GAIN.
        # The reference has no autocast anywhere and keeps W in fp32, so this is also the faithful
        # behaviour -- the bf16 cast was our deviation, not theirs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (torch.matmul(x.float(), self.W.t()) + self.b).to(x.dtype)

    def reset_to_identity(self):
        with torch.no_grad():
            self.W.copy_(torch.eye(self.dim, device=self.W.device, dtype=self.W.dtype))
            self.b.zero_()


def make_confornets(k: int, c_z: int, device, init_noise: float = 0.01):
    """k ConforNets, each identity + N(0, init_noise). The noise is what breaks the symmetry: k
    identical identity maps would have identical gradients and would never separate."""
    nets = []
    for _ in range(k):
        cn = ConforNet(c_z).to(device=device, dtype=torch.float32)
        cn.reset_to_identity()
        with torch.no_grad():
            cn.W.add_(torch.randn_like(cn.W) * init_noise)
            cn.b.add_(torch.randn_like(cn.b) * init_noise)
        for p in cn.parameters():
            p.requires_grad_(True)
        nets.append(cn)
    return nets


# --------------------------------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------------------------------
def distogram_cdf(logits: torch.Tensor) -> torch.Tensor:
    """softmax over bins then cumsum -> CDF in [0,1]. Comparing CDFs rather than the raw softmax
    makes the loss sensitive to HOW FAR apart two distance distributions are, not merely to whether
    they put mass in different bins: two poses differing by one bin score as nearly identical, two
    poses differing by twenty do not."""
    return F.softmax(logits.float(), dim=-1).cumsum(dim=-1)


def pairwise_mse(tensors: list) -> torch.Tensor:
    """Mean pairwise MSE over k tensors, returned NEGATIVE so that minimising it maximises spread."""
    k = len(tensors)
    if k < 2:
        raise ValueError(f"pairwise diversity needs k >= 2, got k={k}")
    flat = [t.reshape(-1) for t in tensors]
    total = tensors[0].new_zeros(())
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            total = total + F.mse_loss(flat[i], flat[j])
            n_pairs += 1
    return -total / n_pairs


# --------------------------------------------------------------------------------------------------
# protenix plumbing
# --------------------------------------------------------------------------------------------------
def build_runner(pdb: str, json_path: str, dump_dir: str, seed: int, n_step: int, n_sample: int):
    """Mirror runner/inference.py:run()'s two-pass config build, then construct the runner.

    parse_sys_args() reads sys.argv, so the arguments are staged there rather than passed.
    """
    sys.path.insert(0, PROTENIX_ROOT)
    os.chdir(PROTENIX_ROOT)  # the checkpoint/cache paths in the configs are relative to the repo root
    from collections.abc import Mapping

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs, parse_sys_args
    from runner.inference import InferenceRunner, update_gpu_compatible_configs

    sys.argv = [
        "confornet_protenix.py",
        "--model_name", "protenix-v2",
        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
        "--input_json_path", json_path,
        "--dump_dir", dump_dir,
        "--seeds", str(seed),
        "--use_template", "true",
        "--sample_diffusion.N_sample", str(n_sample),
        "--sample_diffusion.N_step", str(n_step),
    ]
    arg_str = parse_sys_args()

    configs = parse_configs(
        configs={**configs_base, **{"data": data_configs}, **inference_configs},
        arg_str=arg_str, fill_required_with_null=True)
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_configs[configs.model_name])
    configs = parse_configs(configs=base_configs, arg_str=arg_str, fill_required_with_null=True)
    configs = update_gpu_compatible_configs(configs)
    return InferenceRunner(configs), configs


def get_batch(configs):
    """First (and only) item of the inference dataloader."""
    from protenix.data.inference.infer_dataloader import get_inference_dataloader

    for batch in get_inference_dataloader(configs=configs):
        data, atom_array, err = batch[0]
        if err:
            raise RuntimeError(f"dataloader reported: {err}")
        return data, atom_array
    raise RuntimeError("dataloader yielded nothing")


def run_trunk_prefix(runner, data, n_cycle, chunk_size, seed, mc_dropout_apply_rate):
    """Run the recycling trunk and return its state just before the final pairformer.

    Stochastic: MSA rows are resubsampled every recycle (Protenix's stock test-time behaviour, since
    min_size.test=1), and MC dropout is rolled ONCE for the whole pass exactly as _main_inference_loop
    does it, so 40% of passes get dropout on all recycles rather than 40% of recycles getting it.

    Returns (s_inputs, s_pre, z_pre), all detached. Uses the _trunk_stop_before_last_pairformer hook,
    so the final pairformer stack is left for the caller to run WITH grad.
    """
    import random

    from protenix.utils.seed import seed_everything

    model = runner.model
    seed_everything(seed=seed, deterministic=False)
    mc_dropout = random.random() < mc_dropout_apply_rate
    model._trunk_stop_before_last_pairformer = True
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s_inputs, s_pre, z_pre = model.get_pairformer_output(
                input_feature_dict=data["input_feature_dict"],
                N_cycle=n_cycle,
                inplace_safe=False,   # a tensor we hand back must not be scribbled on later
                chunk_size=chunk_size,
                mc_dropout=mc_dropout,
            )
    finally:
        model._trunk_stop_before_last_pairformer = False
    return s_inputs.detach(), s_pre.detach(), z_pre.detach(), mc_dropout


def final_stack_cdf(model, cn, s_pre, z_pre, chunk_size):
    """One ConforNet's forward: affine on z, final pairformer stack, distogram head, CDF.

    Returns (cdf, s, z). The s and z are exactly what a decode needs, so at a save step we stash them
    rather than recomputing -- the forward has to happen anyway for the loss.

    This is the entire differentiable path. inplace_safe=False is mandatory -- the in-place variants
    overwrite tensors autograd still needs.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z = cn(z_pre)
        s, z = model.pairformer_stack(
            s_pre, z,
            pair_mask=None,
            triangle_multiplicative=model.configs.triangle_multiplicative,
            triangle_attention=model.configs.triangle_attention,
            inplace_safe=False,
            chunk_size=chunk_size,
        )
        logits = model.distogram_head(z)
    return distogram_cdf(logits), s, z


def decode_from_trunk(runner, data, atom_array, s_inputs, s, z, seed, tag):
    """Diffuse from an ALREADY-COMPUTED trunk state; run no trunk of our own.

    s and z are this optimisation step's own post-affine, post-pairformer tensors, stashed at the
    forward. Handing them over via _trunk_override means:
      * the poses come from the trunk the affine was actually trained against, not a re-run;
      * we skip 10 recycles per decode (the dominant cost of the whole run);
      * diffusion, the confidence heads and the cif dumper all run through their normal paths.

    NO seed_everything here. An earlier version reseeded to a fixed value before every decode, a
    leftover from a design that cached one trunk for the whole run. It survived the switch to
    per-step trunks and quietly collapsed 40 trunk draws down to 10 -- every decode in a run saw the
    same MSA subsample, dropout roll and diffusion noise. The reference seeds once per run and lets
    the stream continue; so do we.
    """
    model = runner.model
    data = copy.deepcopy(data)   # runner.predict() deletes template/msa keys from the dict it is given
    model._trunk_override = (s_inputs, s, z)
    try:
        prediction = runner.predict(data)
    finally:
        model._trunk_override = None
    runner.dumper.dump(
        dataset_name="", pdb_id=tag, seed=seed, pred_dict=prediction, atom_array=atom_array,
        entity_poly_type={k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"})
    return prediction


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="8che")
    ap.add_argument("--gpu", type=int, required=True)
    # Defaults below are the paper's optimisation recipe verbatim (sec. "Optimization details"):
    # k=2, 20 steps, Adam lr 1e-3 halved every 5 steps, gradients clipped to norm 10 PER ConforNet.
    # k=2 because their benchmarks all have exactly two reference conformational states -- that is a
    # property of their benchmark, not of the method, and is the first thing worth varying here since
    # an antibody-antigen target has no reason to have two.
    ap.add_argument("--k", type=int, default=2, help="number of ConforNets trained jointly")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--halve-lr-every", type=int, default=5, help="0 disables; paper uses 5")
    ap.add_argument("--init-noise", type=float, default=0.01)
    ap.add_argument("--save-steps", default="5,10,15,20",
                    help="decode after this many optimisation UPDATES. The stash is taken at the forward, where the affine has had exactly `step` updates, so these are update counts, not loop indices. 5/10/15/20 = one snapshot at the end of each learning-rate block.")
    ap.add_argument("--n-sample", type=int, default=5, help="diffusion samples per decode")
    ap.add_argument("--n-step", type=int, default=200, help="diffusion steps per decode")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--chunk-size", type=int, default=0, help="0 = None (fastest, most memory)")
    ap.add_argument("--blocks-per-ckpt", type=int, default=1,
                    help="pairformer activation checkpointing during backprop; 0 disables")
    ap.add_argument("--fixed-trunk", action="store_true",
                    help="run the trunk ONCE and reuse it for every step. Cheap smoke test only: the "
                         "ConforNet is then fit to a single MSA/dropout draw.")
    ap.add_argument("--mc-dropout-rate", type=float, default=None,
                    help="override the per-pass MC dropout probability (default: the config's 0.4)")
    ap.add_argument("--probe-memory", action="store_true",
                    help="run the trunk, then ONE forward+backward, print peak VRAM, exit")
    ap.add_argument("--tag", default="")
    A = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)
    os.environ["LAYERNORM_TYPE"] = "openfold"
    # NOTE: PROTENIX_MSA_STRATEGY is deliberately left unset. Unset means Protenix's stock random
    # subsampling -- sample_size ~ randint(1, n+1), redrawn every recycle. Setting it to "topk" would
    # pin a deterministic row set and remove the very stochasticity this run is training against.

    tag = A.tag or f"{A.pdb}_cn{A.k}"
    outdir = f"{EXP}/artifacts/confornet/{tag}"
    os.makedirs(outdir, exist_ok=True)
    chunk_size = A.chunk_size or None
    save_steps = ([int(x) for x in A.save_steps.split(",") if x != ""]
                  if A.save_steps else [A.steps - 1])

    runner, configs = build_runner(
        A.pdb, f"{JDIR}/{A.pdb}-final-updated.json", f"{outdir}/decoded",
        A.seed, A.n_step, A.n_sample)
    model = runner.model
    model.eval()
    if A.mc_dropout_rate is not None:
        runner.configs.mc_dropout_apply_rate = A.mc_dropout_rate
    drop_rate = runner.configs.mc_dropout_apply_rate

    data, atom_array = get_batch(runner.configs)
    from protenix.utils.torch_utils import to_device
    data = to_device(data, runner.device)

    # Protenix.forward does two feature-prep steps before it ever reaches the trunk (protenix.py
    # :1216-1217): generate_relp adds "relp" (the relative position encoding the trunk indexes at
    # get_pairformer_output line ~215) and update_input_feature_dict adds d_lm / v_lm / pad_info
    # (Algorithm 5 lines 1-3, consumed by the AtomAttentionEncoder). Calling get_pairformer_output
    # directly bypasses forward, so these must be done by hand or the input embedder dies on
    # KeyError: 'd_lm'. Both are pure recomputation, so the decode path re-running them via forward
    # is harmless.
    from protenix.model.protenix import update_input_feature_dict
    data["input_feature_dict"] = update_input_feature_dict(
        model.relative_position_encoding.generate_relp(data["input_feature_dict"]))

    from runner.inference import update_inference_configs
    n_token = int(data["N_token"].item())
    runner.update_model_configs(update_inference_configs(runner.configs, n_token))
    n_cycle = model.configs.model.N_cycle
    print(f"{A.pdb}: N_token={n_token} N_cycle={n_cycle} k={A.k} chunk_size={chunk_size} "
          f"msa=subsampled dropout_rate={drop_rate} "
          f"trunk={'FIXED (one draw)' if A.fixed_trunk else 'resampled every step'}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    s_inputs, s_pre, z_pre, drop_on = run_trunk_prefix(
        runner, data, n_cycle, chunk_size, A.seed, drop_rate)   # step 0; see block note below
    print(f"trunk 0: z_pre={tuple(z_pre.shape)} s_pre={tuple(s_pre.shape)} mc_dropout={drop_on}  "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)

    c_z = z_pre.shape[-1]
    if A.blocks_per_ckpt:
        model.pairformer_stack.blocks_per_ckpt = A.blocks_per_ckpt

    # Separate namespace from the trunk seeds (which occupy [seed, seed+steps) on the low integers),
    # so the phi init noise can never be drawn from the same RNG state as a trunk draw.
    torch.manual_seed(1_000_000 + A.seed)
    nets = make_confornets(A.k, c_z, runner.device, A.init_noise)

    # Freeze the base model. The optimiser only ever holds phi, so the weights were never UPDATED --
    # but without this, backward still populates .grad on all 464M Protenix params, costing ~1.9 GiB
    # of buffers and the backward compute to fill them. Frozen in implementation, not just in effect.
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for cn in nets for p in cn.parameters()], lr=A.lr)

    if A.probe_memory:
        torch.cuda.reset_peak_memory_stats()
        loss = pairwise_mse([final_stack_cdf(model, cn, s_pre, z_pre, chunk_size) for cn in nets])
        loss.backward()
        print(f"one fwd+bwd at k={A.k}: loss={loss.item():.6e}  "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        return

    # --- optimisation -----------------------------------------------------------------------------
    csv_path = f"{outdir}/training_loss.csv"
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["step", "loss", "pairwise_cdf_mse", "lr", "mc_dropout", "peak_gib"])

    # A.steps+1 iterations to perform A.steps updates: the LAST iteration's forward exists only so we
    # can decode a 20-update affine, and its own update would never be decoded from, so it is skipped.
    #
    # At iteration t the affine has had exactly t updates, because the stash is taken at the FORWARD,
    # before that iteration's optimiser step. So stashing at t in {5,10,15,20} yields affines with
    # 5/10/15/20 updates -- one snapshot at the end of each learning-rate block. The files are labelled
    # by UPDATE COUNT (= t), so nobody has to reason about the indexing later.
    for step in range(A.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        is_save = step in save_steps
        is_last = step == A.steps

        # A fresh trunk draw per step: new MSA subsample, new dropout roll. One trunk is shared by
        # all k ConforNets within the step, so the affine stays the only thing separating them --
        # if each got its own draw, the diversity loss would be rewarding trunk noise instead.
        if step > 0 and not A.fixed_trunk:
            # Trunk seed is --seed + step, so a run consumes the contiguous block [seed, seed+steps].
            # Non-overlap is the DRIVER's job: it spaces run seeds by at least `steps`.
            s_inputs, s_pre, z_pre, drop_on = run_trunk_prefix(
                runner, data, n_cycle, chunk_size, A.seed + step, drop_rate)

        cdfs, stash = [], []
        for cn in nets:
            cdf, s_out, z_out = final_stack_cdf(model, cn, s_pre, z_pre, chunk_size)
            cdfs.append(cdf)
            if is_save:
                # Detached copies + a snapshot of the weights that produced them. Diffusing HERE
                # would hold the pairformer activations and the diffusion allocations at once; the
                # stash lets backward free the graph first, so the two peaks never overlap.
                stash.append((s_out.detach(), z_out.detach(),
                              copy.deepcopy({k: v.detach().cpu() for k, v in cn.state_dict().items()})))
        loss = pairwise_mse(cdfs)
        loss_val = loss.item()
        del cdfs

        if not is_last:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for cn in nets:
                torch.nn.utils.clip_grad_norm_(cn.parameters(), A.grad_clip)
            opt.step()
        del loss          # frees the graph on the final iteration, where backward never runs

        lr = opt.param_groups[0]["lr"]
        peak = torch.cuda.max_memory_allocated() / 2**30
        wr.writerow([step, f"{loss_val:.8f}", f"{-loss_val:.8f}", f"{lr:.6e}",
                     int(drop_on), f"{peak:.2f}"])
        fh.flush()   # a long optimisation must be monitorable without waiting for it to end
        print(f"step {step:>3}: loss={loss_val:.6e} cdf_mse={-loss_val:.6e} "
              f"lr={lr:.2e} drop={int(drop_on)} peak={peak:.1f} GiB", flush=True)

        # Halve AFTER updates 5, 10, 15 -> clean 5/5/5/5 blocks. Not after update 20: nothing follows.
        n_updates = step + 1
        if A.halve_lr_every and n_updates % A.halve_lr_every == 0 and n_updates < A.steps:
            for g in opt.param_groups:
                g["lr"] /= 2.0

        if is_save:
            for i, (s_o, z_o, sd) in enumerate(stash):
                with torch.no_grad():
                    decode_from_trunk(runner, data, atom_array, s_inputs, s_o, z_o,
                                      A.seed, f"{tag}_step{step}_cn{i}")
                torch.save(sd, f"{outdir}/{tag}_step{step}_cn{i}.pt")
            print(f"  decoded {A.k} x {A.n_sample} samples at {step} updates", flush=True)
        del stash

    fh.close()
    json.dump({"pdb": A.pdb, "k": A.k, "steps": A.steps, "lr": A.lr, "seed": A.seed,
               "n_token": n_token, "n_cycle": n_cycle, "save_steps": save_steps,
               "objective": "dist_cdf_mse", "fixed_trunk": A.fixed_trunk,
               "msa": "subsampled (stock, topk off)", "mc_dropout_apply_rate": drop_rate},
              open(f"{outdir}/config.json", "w"), indent=1)
    print(f"\ndone -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
