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
def load(chunk_size: int = 32, keep_esmc: bool = False):
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(chunk_size if chunk_size > 0 else None)
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


def run_trunk(model, z_init, lm_z, pair_mask, z0, total_steps, collect_per_loop=False):
    """Recycling trunk with **per-loop gradient checkpointing** so only the 17
    loop-boundary z's are retained (each loop is recomputed in backward). Returns
    (z_final, per_loop_leaves). per_loop leaves give per-loop attribution."""
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
            z = checkpoint(_loop_body, model, z, z_init, lm_z_i, pair_mask, a, b_mat,
                           use_reentrant=False)
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
