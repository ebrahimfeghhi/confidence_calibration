"""Phase 0: a *differentiable* ESMFold2-Fast trunk -> cross-block distogram path,
with the per-residue single LM rep `s_lm [1,L,256]` as the autograd leaf.

Why this exists
---------------
`ESMFold2Model.forward` is `@torch.inference_mode()`, but `_run_one_loop` and all
the submodules are not. So we rebuild the pre-loop tensors (reusing the existing
`build_conditioning`), inject a *leaf* LM rep, run the recycling trunk under
`enable_grad`, and read the gradient of a distogram loss back to `s_lm`.

Target = the single rep at `LanguageModelShim.forward` (the [1,L,256] tensor that
`base_z_mlp`'s SingleToPair lifts to the [1,L,L,256] pair `lm_z` injected every
loop). Perturbing the single rep is ~600x lower-dim than the pair tensor, stays on
the SingleToPair manifold, and is per-residue. See the plan/memories for rationale.

Determinism: `lm_dropout=0` (skipped here) AND a *fixed* initial pair state `z0`
(drawn once, reused) — `_init_pair_state` is random, so without pinning it the loss
would be noisy across steps.

Run (esm venv, GPU): python steer/core/forward_grad.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
from complex_io import load_complex, build_conditioning  # noqa: E402
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR  # noqa: E402

N_LOOPS = 16                                    # parcae runs N_LOOPS + 1 times

# ----- per-target config: chain slices (antibody rows AB, antigen cols C) + captured-rep
#       paths. Defaults to 7txw so every existing import keeps working; call
#       configure_target(name) BEFORE load() to switch (load()/cross_distogram_from_z
#       read these module globals at call time). NOTE chain ORDER is not assumable:
#       7txw is antibody-first, 8im0 is antigen-first with a single VHH chain.
TARGETS = {
    "7txw": dict(ab=slice(0, 438), c=slice(438, 609),
                 reps=ESM_PREDICTIONS / "7txw_reps" / "7txw",
                 baseline=ESM_PREDICTIONS / "7txw_reps_baseline" / "7txw" / "seed_0",
                 good=[10, 14, 33, 39]),
    "8im0": dict(ab=slice(235, 362), c=slice(0, 235),
                 reps=None,
                 baseline=ESM_PREDICTIONS / "8im0_reps_baseline" / "8im0" / "seed_0",
                 good=None),
    # 8c5h: antigen(His-tagged, 132)=tokens 0:132, nanobody(125)=tokens 132:257
    "8c5h": dict(ab=slice(132, 257), c=slice(0, 132),
                 reps=None,
                 baseline=ESM_PREDICTIONS / "8c5h_reps_baseline" / "8c5h" / "seed_0",
                 good=None),
    # --- Tier-1 failures (single antibody + single antigen, contiguous AB). slices = JSON chain order.
    "8ts0": dict(ab=slice(0, 450), c=slice(450, 619),      # Fab(236+214) | ag(169)
                 reps=None, baseline=ESM_PREDICTIONS / "8ts0_reps_baseline" / "8ts0" / "seed_0", good=None),
    "8z8v": dict(ab=slice(585, 708), c=slice(0, 585),      # ag(585) | VHH(123)
                 reps=None, baseline=ESM_PREDICTIONS / "8z8v_reps_baseline" / "8z8v" / "seed_0", good=None),
    "8urf": dict(ab=slice(0, 446), c=slice(446, 605),      # Fab(214+232) | ag(159)
                 reps=None, baseline=ESM_PREDICTIONS / "8urf_reps_baseline" / "8urf" / "seed_0", good=None),
    "8vzo": dict(ab=slice(551, 1006), c=slice(0, 551),     # ag(551) | Fab(240+215)
                 reps=None, baseline=ESM_PREDICTIONS / "8vzo_reps_baseline" / "8vzo" / "seed_0", good=None),
    "8jyr": dict(ab=slice(202, 642), c=slice(0, 202),      # ag(202) | Fab(222+218)
                 reps=None, baseline=ESM_PREDICTIONS / "8jyr_reps_baseline" / "8jyr" / "seed_0", good=None),
    "8vvl": dict(ab=slice(268, 707), c=slice(0, 268),      # ag(268) | Fab(226+213)
                 reps=None, baseline=ESM_PREDICTIONS / "8vvl_reps_baseline" / "8vvl" / "seed_0", good=None),
}
_TARGET = "7txw"
REPS = TARGETS["7txw"]["reps"]
BASELINE = TARGETS["7txw"]["baseline"]
AB, C = TARGETS["7txw"]["ab"], TARGETS["7txw"]["c"]
GOOD = TARGETS["7txw"]["good"]


def configure_target(name):
    """Switch module target (chain slices + rep paths); returns the cfg dict."""
    global REPS, BASELINE, AB, C, GOOD, _TARGET
    cfg = TARGETS[name]
    AB, C = cfg["ab"], cfg["c"]
    BASELINE, REPS, GOOD, _TARGET = cfg["baseline"], cfg["reps"], cfg["good"], name
    return cfg


# ----------------------------------------------------------------------------- setup
def load(chunk_size: int = 32, keep_esmc: bool = False, kernel_backend: str = None):
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(chunk_size if chunk_size > 0 else None)   # chunk_size<=0 -> None (no chunking; faster at L<600)
    if kernel_backend:                     # "fused" -> Triton trimul kernel on the O(L^3) op (bf16 fwd+bwd); speeds steer & decode
        model.set_kernel_backend(kernel_backend)
    model.requires_grad_(False)            # only the chosen leaf gets grad; no param-grad buffers
    if getattr(model, "_esmc", None) is not None and not keep_esmc:
        model._esmc.to("cpu")              # LM reps come from saved npy; free ~6B weights
        torch.cuda.empty_cache()
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), _TARGET)
    feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    return model, builder, feat, chain_infos, cond


def compute_z_init(model, cond):
    """Replicate forward():950-962 (the deterministic pair-state reference)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        xi = cond["x_inputs"]
        z_init = model.z_init_1(xi).unsqueeze(2) + model.z_init_2(xi).unsqueeze(1)
        z_init = z_init + cond["rel_pos"] + cond["token_bonds_encoding"]
    return z_init


def fixed_z0(model, z_init, seed: int = 0):
    """Draw the random initial pair state ONCE, deterministically, to reuse."""
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    z0 = model._init_pair_state(z_init)
    torch.random.set_rng_state(g)
    return z0


# ------------------------------------------------------------------- LM rep -> leaf
def single_rep_from_hidden(model, hidden):
    """Split LanguageModelShim.forward at the [1,L,256] single rep (the leaf point).

    hidden: [1, L, 81, 2560] ESMC rep (the saved esmc_hidden_states)."""
    lm = model.language_model
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        s = lm.base_z_linear(hidden)                 # [1,L,81,256]
        w = lm.base_z_combine.softmax(0)             # [81]
        s_lm = (w @ s).squeeze(-2)                    # [1,L,256]
    return s_lm.float()


def lift_to_lm_z(model, s_lm):
    """s_lm [1,L,256] -> pair lm_z [1,L,L,256] via base_z_mlp (SingleToPair+LN)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.language_model.base_z_mlp(s_lm)


def load_hidden(path: Path):
    arr = np.load(path / "reps" / "esmc_hidden_states.npy")
    return torch.from_numpy(arr.astype(np.float32)).cuda()


# --------------------------------------------------------------------- trunk (grad)
def _loop_body(model, z, z_init, lm_z_i, pair_mask, a, b_mat):
    """One recycling iteration (lm_dropout=0, no MSA) = `_run_one_loop`'s body."""
    if model.lm_encoder is not None:
        refined = model.lm_encoder(lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask)
        z_inject = z_init + refined.to(z_init.dtype)
    else:
        z_inject = z_init + lm_z_i.to(z_init.dtype)
    injected = model.parcae_input_norm(z_inject)
    z = a * z + F.linear(injected.to(z.dtype), b_mat)
    return model.folding_trunk(z, pair_attention_mask=pair_mask)


def run_trunk(model, z_init, lm_z, pair_mask, z0, total_steps, collect_per_loop=False, use_ckpt=True):
    """Recycling trunk with **per-loop gradient checkpointing** so only the 17
    loop-boundary z's are retained (each loop is recomputed in backward). Returns
    (z_final, per_loop_leaves). per_loop leaves give per-loop attribution.

    use_ckpt=False drops the outer per-loop checkpoint (no recompute in backward) —
    faster when VRAM allows (small L). NOTE: the model's inner per-block checkpoint
    in FoldingTrunk.forward is unconditional, so this only removes the outer level."""
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(device=z_init.device, dtype=z_init.dtype)
    b_mat = b.to(device=z_init.device, dtype=z_init.dtype)
    z = z0
    per_loop = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(total_steps):
            lm_z_i = lm_z
            if collect_per_loop:                      # fresh leaf at this loop's value
                lm_z_i = lm_z.detach().clone().requires_grad_(True)
                lm_z_i.retain_grad()
                per_loop.append(lm_z_i)
            if use_ckpt:
                z = checkpoint(_loop_body, model, z, z_init, lm_z_i, pair_mask, a, b_mat,
                               use_reentrant=False)
            else:
                z = _loop_body(model, z, z_init, lm_z_i, pair_mask, a, b_mat)
    return z, per_loop


def cross_distogram_from_z(model, z, pair_mask):
    """z (trunk output) -> readout/coda -> distogram_head -> softmax -> AB×C block.
    Differentiable version of distogram_kl_basins.cross_distogram. Returns [npairs,nbins]."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()
    logits = model.distogram_head(z + z.transpose(-2, -3))[0]      # [L,L,nbins]
    P = F.softmax(logits, dim=-1)[AB, C, :].reshape(-1, logits.shape[-1])
    return P


def final_trunk_z(model, z_init, lm_z, pair_mask, z0, steps):
    """Forward-only recycling -> final trunk pair-state z (what BOTH heads read).
    Same z0/trajectory as the steering's disto_fwd, so a structure decoded from this z
    follows the steered (escaped) distogram instead of a re-rolled forward pass."""
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    z = z0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(steps):
            z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
    return z.float()


def final_trunk_z_batched(model, z_init, lm_z_B, pair_mask, z0, steps):
    """Batched forward-only launch: B leaves' pair conditioning lm_z_B [B,L,L,256]
    -> final trunk z [B,L,L,256], all in one pass (grad-free). Same trajectory as
    final_trunk_z but the leading batch dim B carries independent leaves; the shared
    z_init/z0/a/b broadcast over it. Use to launch many steered s_lm's to loop 17 at
    once before decoding."""
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    B = lm_z_B.shape[0]
    z = z0.expand(B, -1, -1, -1).contiguous()      # shared initial pair state per leaf
    pm = pair_mask.expand(B, -1, -1).contiguous() if pair_mask.shape[0] == 1 else pair_mask  # fused trimul needs mask batched to B (no broadcast)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(steps):
            z = _loop_body(model, z, z_init, lm_z_B, pm, a, b_mat)
    return z.float()


def decode_from_ztrunk(model, builder, feat, chain_infos, cond, z, n_samples, n_steps, name, seed):
    """Diffuse a 3D structure from a GIVEN trunk z (tensor) -> decoded samples (each
    .complex.to_mmcif(), .iptm). Port of the per-loop diffuse_from_z, taking the live
    steering z_trunk so the structure is forced onto the steered orientation."""
    tok_mask = feat["token_attention_mask"]
    atm_mask = feat["atom_attention_mask"]
    disto_idx = feat["distogram_atom_idx"]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        zc = model.parcae_readout(z)
        zc = model.parcae_coda(zc, pair_attention_mask=cond["pair_mask"])
    zc = zc.float()
    distogram_logits = model.distogram_head(zc + zc.transpose(-2, -3))
    torch.manual_seed(seed)
    with torch.no_grad():
        structure_output = model.structure_head.sample(
            z_trunk=zc, s_inputs=cond["x_inputs"], s_trunk=None,
            relative_position_encoding=cond["rel_pos"],
            ref_pos=feat["ref_pos"], ref_charge=feat["ref_charge"], ref_mask=atm_mask,
            ref_element=cond["ref_element_oh"], ref_atom_name_chars=cond["ref_name_oh"],
            ref_space_uid=feat["ref_space_uid"], tok_idx=cond["atom_to_token"],
            asym_id=feat["asym_id"], residue_index=feat["residue_index"],
            entity_id=feat["entity_id"], token_index=feat["token_index"], sym_id=feat["sym_id"],
            token_attention_mask=tok_mask, num_diffusion_samples=n_samples,
            num_sampling_steps=n_steps, noise_scale=None, step_scale=None,
            return_atom_repr=False, denoising_early_exit_rmsd=None,
        )
        sample_coords = structure_output["sample_atom_coords"]
        output = {"distogram_logits": distogram_logits, "sample_atom_coords": sample_coords}
        confidence_output = model.confidence_head(
            s_inputs=cond["x_inputs"].detach(), z=zc.detach().float(),
            x_pred=sample_coords.detach(), distogram_atom_idx=disto_idx,
            token_attention_mask=tok_mask, atom_to_token=cond["atom_to_token"],
            atom_attention_mask=atm_mask, asym_id=feat["asym_id"], mol_type=feat["mol_type"],
            num_diffusion_samples=n_samples,
            relative_position_encoding=cond["rel_pos"].detach(),
            token_bonds_encoding=cond["token_bonds_encoding"].detach(),
        )
    output.update(confidence_output)
    output["atom_pad_mask"] = atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
    output["residue_index"] = feat["residue_index"]
    output["entity_id"] = feat["entity_id"]
    results = builder.decode(output, feat, chain_infos,
                             num_diffusion_samples=n_samples, complex_id=name)
    return results if isinstance(results, list) else [results]


def decode_from_ztrunk_batched(model, builder, feat, chain_infos, cond, z_B,
                               n_samples, n_steps, name, seed):
    """Batched decode of B independent trunk states of the SAME target.

    z_B: [B,L,L,·]. Only the expensive, sequential diffusion sampler is batched
    across the B structures (effective batch B*n_samples, structure-major/sample-minor);
    the cheap confidence_head + builder.decode stay per-structure so their single-
    structure contract is untouched. Since all B share one target, the conditioning is
    identical and just `expand`s over B. Returns list[B] of per-structure sample lists.

    Seed note: one torch.manual_seed(seed) draws the whole batch's diffusion noise, so
    per-structure noise differs from the batch-1 path (same method, reseeded samples)."""
    B = z_B.shape[0]
    tok_mask = feat["token_attention_mask"]
    atm_mask = feat["atom_attention_mask"]
    disto_idx = feat["distogram_atom_idx"]

    def eb(t):                                   # expand a leading-1 tensor to batch B
        return t.expand(B, *([-1] * (t.dim() - 1))) if torch.is_tensor(t) and t.shape[0] == 1 else t

    pm_B = cond["pair_mask"].expand(B, -1, -1).contiguous()   # fused trimul needs mask batched to B (no broadcast)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        zc = model.parcae_readout(z_B)
        zc = model.parcae_coda(zc, pair_attention_mask=pm_B)
    zc = zc.float()                              # [B,L,L,·]
    distogram_logits = model.distogram_head(zc + zc.transpose(-2, -3))      # [B,L,L,nbins]
    torch.manual_seed(seed)
    with torch.no_grad():
        structure_output = model.structure_head.sample(
            z_trunk=zc, s_inputs=eb(cond["x_inputs"]), s_trunk=None,
            relative_position_encoding=eb(cond["rel_pos"]),
            ref_pos=eb(feat["ref_pos"]), ref_charge=eb(feat["ref_charge"]), ref_mask=eb(atm_mask),
            ref_element=eb(cond["ref_element_oh"]), ref_atom_name_chars=eb(cond["ref_name_oh"]),
            ref_space_uid=eb(feat["ref_space_uid"]), tok_idx=eb(cond["atom_to_token"]),
            asym_id=eb(feat["asym_id"]), residue_index=eb(feat["residue_index"]),
            entity_id=eb(feat["entity_id"]), token_index=eb(feat["token_index"]), sym_id=eb(feat["sym_id"]),
            token_attention_mask=eb(tok_mask), num_diffusion_samples=n_samples,
            num_sampling_steps=n_steps, noise_scale=None, step_scale=None,
            return_atom_repr=False, denoising_early_exit_rmsd=None,
        )
        coords_all = structure_output["sample_atom_coords"]     # [B*n_samples, n_atoms, 3], structure-major
    out = []
    for bi in range(B):                          # confidence + mmcif per structure (cheap, unchanged contract)
        coords_bi = coords_all[bi * n_samples:(bi + 1) * n_samples]
        dl_bi = distogram_logits[bi:bi + 1] if torch.is_tensor(distogram_logits) else distogram_logits
        output = {"distogram_logits": dl_bi, "sample_atom_coords": coords_bi}
        with torch.no_grad():
            confidence_output = model.confidence_head(
                s_inputs=cond["x_inputs"].detach(), z=zc[bi:bi + 1].detach().float(),
                x_pred=coords_bi.detach(), distogram_atom_idx=disto_idx,
                token_attention_mask=tok_mask, atom_to_token=cond["atom_to_token"],
                atom_attention_mask=atm_mask, asym_id=feat["asym_id"], mol_type=feat["mol_type"],
                num_diffusion_samples=n_samples,
                relative_position_encoding=cond["rel_pos"].detach(),
                token_bonds_encoding=cond["token_bonds_encoding"].detach(),
            )
        output.update(confidence_output)
        output["atom_pad_mask"] = atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
        output["residue_index"] = feat["residue_index"]
        output["entity_id"] = feat["entity_id"]
        results = builder.decode(output, feat, chain_infos,
                                 num_diffusion_samples=n_samples, complex_id=name)
        out.append(results if isinstance(results, list) else [results])
    return out


def distogram_from_s_lm(model, s_lm, z_init, pair_mask, z0,
                        total_steps=N_LOOPS + 1, collect_per_loop=False):
    """Full path: leaf s_lm -> lm_z -> trunk -> cross-block distogram P."""
    lm_z = lift_to_lm_z(model, s_lm)
    z, per_loop = run_trunk(model, z_init, lm_z, pair_mask, z0, total_steps,
                            collect_per_loop=collect_per_loop)
    P = cross_distogram_from_z(model, z, pair_mask)
    return P, per_loop


# --------------------------------------------------------------------------- losses
def w1(P, Q):
    """Mean over pairs of 1D Wasserstein-1 over ORDERED distance bins (= L1 of CDFs)."""
    return (P.cumsum(-1) - Q.cumsum(-1)).abs().sum(-1).mean()


def good_reference(model, pair_mask, z0, z_init, seeds=GOOD):
    """P_G_ref = mean over good seeds of their SAVED loop16 cross-block distogram
    (the actual dropout-on G basin), in probability space. Detached target."""
    Ps = []
    for s in seeds:
        z = torch.from_numpy(
            np.load(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)
        ).cuda()
        with torch.no_grad():
            Ps.append(cross_distogram_from_z(model, z, pair_mask))
    Pg = torch.stack(Ps).mean(0)
    return (Pg / Pg.sum(-1, keepdim=True)).detach()


# ----------------------------------------------------------------------------- main
def main():
    torch.manual_seed(0)
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)
    print(f"z_init {tuple(z_init.shape)}  z0 {tuple(z0.shape)}", flush=True)

    Pg = good_reference(model, pair_mask, z0, z_init)
    print(f"G reference distogram {tuple(Pg.shape)}", flush=True)

    # --- START FROM THE BASELINE (no mask, no dropout) — the canonical B1 point ---
    hidden = load_hidden(BASELINE)
    print(f"baseline esmc hidden {tuple(hidden.shape)}", flush=True)
    s_lm = single_rep_from_hidden(model, hidden).detach().requires_grad_(True)
    print(f"s_lm leaf {tuple(s_lm.shape)}  ||s_lm||={s_lm.norm().item():.3e}", flush=True)

    def loss_at(x):
        with torch.no_grad():
            P, _ = distogram_from_s_lm(model, x, z_init, pair_mask, z0)
            return w1(P, Pg).item()

    with torch.enable_grad():
        P, _ = distogram_from_s_lm(model, s_lm, z_init, pair_mask, z0)
        loss = w1(P, Pg)
    loss.backward()
    g = s_lm.grad
    L0 = loss.item()
    gn = g.norm().item()
    print(f"\nloss W1(P_baseline, P_G) = {L0:.5f}  (baseline starts in B1, far from G)",
          flush=True)
    print(f"grad: ||g||={gn:.4e}  finite={torch.isfinite(g).all().item()}", flush=True)

    # --- (1) noise floor: same input, run loss twice; how much does CUDA jitter move W1? ---
    jit = abs(loss_at(s_lm) - loss_at(s_lm))
    print(f"\nnoise floor (|W1 run-to-run| on identical input) = {jit:.2e}", flush=True)

    # --- (2) along-gradient check: step -eps*ghat; predicted drop = eps*||g|| (1st order).
    #         Large eps so signal clears the jitter AND survives bf16 rounding. ---
    print("\nalong-gradient check  (step x = s_lm - eps*ghat):", flush=True)
    ghat = g / (gn + 1e-12)
    print(f"  {'eps':>7} {'predicted dL':>13} {'actual dL':>11} {'ratio':>7}", flush=True)
    for eps in (0.05, 0.2, 0.5, 1.0):
        actual = loss_at(s_lm - eps * ghat) - L0
        pred = -eps * gn
        print(f"  {eps:7.2f} {pred:13.4e} {actual:11.4e} {actual/pred:7.2f}", flush=True)
    print("  (ratio ~1 at small eps = grad correct; <1 at large eps = curvature; "
          "actual dL >> noise floor)", flush=True)

    # --- per-loop attribution: which loop's injected rep moves the loss ---
    print("\nper-loop gradient norms (||d loss / d lm_z_i||):", flush=True)
    s_lm2 = s_lm.detach().requires_grad_(True)
    with torch.enable_grad():
        P, per_loop = distogram_from_s_lm(model, s_lm2, z_init, pair_mask, z0,
                                          collect_per_loop=True)
        loss2 = w1(P, Pg)
    loss2.backward()
    norms = [pl.grad.norm().item() if pl.grad is not None else 0.0 for pl in per_loop]
    for i, n in enumerate(norms):
        bar = "#" * int(60 * n / (max(norms) + 1e-12))
        print(f"  loop {i:2d}  {n:.4e}  {bar}", flush=True)
    print(f"\n  peak loop = {int(np.argmax(norms))}  "
          f"(expect mid/late ~3-12 per b1-default-basin-late-escape)", flush=True)
    print("\nPhase 0 OK.", flush=True)


if __name__ == "__main__":
    main()
