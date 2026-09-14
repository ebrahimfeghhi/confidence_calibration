# How diffusion works in ESMFold2 (token-level + atom-level)

A code-anchored tutorial on ESMFold2's diffusion structure head: the EDM denoising
framework, the two coupled resolutions (per-residue **tokens** and per-atom **atoms**),
the full sampling loop, and every parameter you can turn.

All `file:line` references point at the installed transformers port:

```
/home/ebrahim/esm/.venv/lib/python3.12/site-packages/transformers/models/esmfold2/
    modeling_esmfold2_common.py      # all the modules below
    configuration_esmfold2.py        # all the defaults below
```

Shapes in this doc are the **verified** ones for 7txw (609 tokens, 4640 padded atoms),
captured by the runnable probe in the [appendix](#appendix-runnable-shape-probe).

---

## 1. The big picture

ESMFold2 predicts structure in three stages:

```
 sequence ──► ESMC language model ──► pairformer trunk ──► diffusion structure head ──► 3D coords
                                       (24 trimul layers)   (this document)            + confidence head
```

The trunk produces two conditioning tensors the diffusion head consumes:

| tensor | meaning | shape (7txw) | dim |
|--------|---------|--------------|-----|
| `z_trunk` | **pair** representation (residue × residue) | `(1, 609, 609, 256)` | `c_z = 256` |
| `s_inputs` | per-**token** input features (`InputsEmbedder`) | `(1, 609, 451)` | `c_s_inputs = 451` |

Crucially, **the diffusion head never sees the LM representation directly** — it
conditions only on `z_trunk` plus deterministic input features (`s_inputs`, reference
atom geometry, relative-position encoding). `s_trunk` exists in the API but is `None` in
the shipped inference path. (This is exactly why `run/esm/run_7txw_loop_diffusion.py` can
re-diffuse from a *saved* per-loop `z` without re-running ESMC.)

The head is a **denoising diffusion model over 3D atom coordinates**. It starts from pure
Gaussian noise in `ℝ^(A×3)` and iteratively denoises toward a structure, at every step
conditioning on `z`/`s`. Entry point: `DiffusionStructureHead.sample()`
(`modeling_esmfold2_common.py:1767`).

---

## 2. The EDM denoising framework

ESMFold2 uses the **EDM** formulation (Karras et al., 2022) — the same family AF3 uses —
**not** DDPM. Two things define it: a continuous noise level σ, and a *preconditioned*
denoiser.

### Noise level σ
Coordinates at noise level σ are `x = x_clean + σ·ε`, `ε ~ N(0, I)`. There is no discrete
β-schedule; σ is a real number swept from large (≈pure noise) to ≈0. The data scale is
`sigma_data = 16.0` (Å-ish units; `configuration_esmfold2.py:100`). In the code the noise
level is the variable `t` / `t_hat` (it is σ, not a DDPM timestep).

### Preconditioned denoiser
Rather than predict noise directly, the network `F_θ` is wrapped so that the **denoiser**
`D_θ(x; σ)` is well-conditioned at every σ:

```
D_θ(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ( c_in(σ)·x ; c_noise(σ) )
```

ESMFold2's exact choices (`DiffusionModule.forward`, `modeling_esmfold2_common.py:1509`):

| coefficient | formula | code |
|-------------|---------|------|
| `c_in`  | `1/√(σ² + σ_data²)` | `r_noisy = x_noisy / √(t²+σ²)` — `:1557` |
| `c_noise` | `0.25·ln(σ / σ_data)` | `t_noise = 0.25·log(t/σ)` — `:1410` (in `DiffusionConditioning`) |
| `c_skip` | `σ_data² / (σ_data² + σ²)` | `out = σ²/(σ²+t²)·x_noisy …` — `:1607` |
| `c_out` | `σ_data·σ / √(σ_data² + σ²)` | `… + σ·t/√(σ²+t²)·r_update` — `:1608` |

Intuition: at **high σ** (`c_skip→0`, `c_out` large) the network output dominates — it must
hallucinate global structure from noise. At **low σ** (`c_skip→1`) the denoiser mostly
passes the input through and only refines — small corrections near a finished structure.
`c_in` keeps the network's input unit-variance regardless of σ; `c_noise` gives it a
tamed (log) encoding of the noise level.

### How the model was trained (context)
Training draws σ from a log-normal: `σ = σ_data · exp(μ + s·N(0,1))` with `μ =
train_noise_log_mean = -1.2`, `s = train_noise_log_std = 1.5`
(`configuration_esmfold2.py:124-126`), and regresses `D_θ` onto the aligned ground-truth
coordinates. **This HF port is inference-only** — `DiffusionStructureHead` exposes only
`sample()` (no training `forward`), so those two numbers are documentation/provenance,
not used at inference.

---

## 3. Conditioning (`DiffusionConditioning`, `:1329`)

Before denoising, the trunk outputs are turned into the per-step conditioning tensors
`s` (token) and `z` (pair). `forward` at `:1369`:

- **Pair `z`** (`:1387-1395`): concatenate `z_trunk` with the relative-position encoding,
  LayerNorm, project `2·c_z → c_z`, then two `TransitionLayer` residual blocks.
  `z` does **not** depend on σ, so it is **cached across all sampling steps**
  (`inference_cache["z"]`) — computed once, reused every step. Shape stays
  `(1, 609, 609, 256)`.
- **Single `s`** (`:1398-1416`): project `s_inputs` (`451 → c_token`), then **add the noise
  embedding**: `c_noise(σ)` → `FourierEmbedding` (`:376`) → LayerNorm → linear, broadcast
  over residues (`:1413`); then two transition blocks. This is the **only** place the
  noise level σ enters the token stream. Shape `(1, 609, 768)`.

---

## 4. The two resolutions: atom level vs token level

This is the heart of the head. Each denoising step runs **atom → token → atom**
(`DiffusionModule.forward`, `:1560-1602`):

```
x_noisy (A,3) ─c_in─► r_noisy ─► [ATOM ENCODER] ─scatter─► a (token) ─+s─► [TOKEN TRANSFORMER ×12]
                                       │ skips (q,c,attn)                         │
                                       └──────────────────────────► [ATOM DECODER] ◄── gather ┘
                                                                          │
                                                                     r_update (A,3) ─c_skip/c_out─► x_denoised
```

| level | rep | dim | blocks | heads | attention | code |
|-------|-----|-----|--------|-------|-----------|------|
| **atom** | `q` (per-atom) | `c_atom = 128` | 3 enc + 3 dec | 4 | sliding-window (128) **3D-RoPE** | `SWAAtomTransformer:701` |
| **token** | `a`, `s` (per-residue) | `c_token = 768` | 12 | 16 | global, **pair-biased** by `z` + AdaLN on `s` | `DiffusionTransformer:1253` |
| **pair** | `z` | `c_z = 256` | (from trunk) | — | used as attention **bias** only; not updated here | `AttentionPairBias:1005` |

### 4a. Atom encoder (`ESMFold2AtomEncoder`, `:771`)
1. Build per-atom features (`:853-862`): reference position, charge, mask, element one-hot,
   atom-name characters → linear to `c_atom = 128` → LayerNorm. These are σ-independent and
   **cached** (`c_base`).
2. **Inject the noisy coordinates** (`:892-898`): `coords_linear` maps a 6-D vector
   `[r_noisy, pred_r1]` (the current noisy coords plus a second slot, zeros at inference)
   into atom space and adds it. *This is how the structure being denoised enters the
   network.*
3. `SWAAtomTransformer`: `atom_num_blocks = 3` blocks of **sliding-window attention**
   (`swa_window_size = 128`) with **3D rotary position embeddings** built from `ref_pos`
   and `ref_space_uid` (`spatial_rope_base_frequency = 20.0`,
   `n_spatial_rope_pairs_per_axis = 2`, `n_uid_rope_pairs = 10`). Atoms attend locally —
   cheap, and geometry-aware via 3D RoPE.
4. **Scatter atoms → tokens** (`scatter_atom_to_token`, `:921`): pool the atom reps into one
   `c_token = 768` vector per residue → `a`, shape `(1, 609, 768)`. The per-atom `q` and `c`
   and the attention indices are returned as **skip connections** for the decoder.

### 4b. Token transformer (`DiffusionTransformer`, `:1253`)
`token_num_blocks = 12` blocks, each (`:1309-1318`):
- `AttentionPairBias` (`:1005`): global self-attention over the 609 tokens, `token_num_heads
  = 16`, where the **pair rep `z` provides an additive per-head attention bias** — this is
  how pairwise/geometric information from the trunk shapes which residues attend to which.
- `ConditionedTransitionBlock` (`:1208`): a feed-forward block whose LayerNorm is
  **adaptive (AdaLN)**, modulated by the conditioning `s` (which carries the σ embedding).

Before this stage the conditioned `s` is also added into `a` via a zero-initialized
projection (`s_to_token`, `:1577`). Output stays `(1, 609, 768)`.

### 4c. Atom decoder (`ESMFold2AtomDecoder`, `:933`)
1. **Gather tokens → atoms** (`gather_token_to_atom`, `:981`): broadcast each token vector
   back to its atoms, project to `c_atom`, and add to the encoder's `q` skip.
2. Another `SWAAtomTransformer` (3 blocks) reusing the cached attention params.
3. `output_linear` → a **per-atom 3-vector update** `r_update`, shape `(1, 4640, 3)` (`:996`).

### 4d. Assemble the denoised coordinates (`:1605-1608`)
```
x_denoised = c_skip(σ)·x_noisy + c_out(σ)·r_update
```
i.e. the network's per-atom update is blended with the noisy input by the EDM coefficients.

---

## 5. The sampling loop (`sample()`, `:1767`)

```python
schedule = inference_noise_schedule(steps)          # Karras power-law, :1675
if max_inference_sigma: schedule = cap(schedule)     # drop σ>cap, prepend cap, :1816-1818
x = schedule[0] * randn(B, A, 3)                      # start from noise, :1823
for σ_tm, σ_t, γ in steps:                            # high σ → low σ
    x, prev = center_random_augmentation(x, ...)      # Alg 19, :1718 / :1842
    t_hat   = σ_tm * (1 + γ)                           # churn up,  :1847
    eps_std = noise_scale * √(t_hat² - σ_tm²)          # SDE noise, :1848
    x_noisy = x + eps_std * randn_like(x)              #            :1849
    x_denoised = DiffusionModule(x_noisy, t_hat, z, s) # the denoiser
    x_noisy = weighted_rigid_align(x_noisy, x_denoised)# Kabsch,    :1890
    x = x_noisy + step_scale * (σ_t - t_hat) * (x_noisy - x_denoised)/t_hat   # step, :1899
```

### 5a. The Karras schedule (`inference_noise_schedule`, `:1675`)
```
σ_i = σ_data · ( s_max^(1/p) + (i/(N-1))·(s_min^(1/p) - s_max^(1/p)) )^p ,   then append 0
```
With defaults `inference_s_max = 160`, `inference_s_min = 4e-4`, `inference_p = 8`,
`inference_num_steps = 68`: the top is `σ_data·s_max = 16·160 = 2560`, the bottom is
`σ_data·s_min ≈ 0.0064`, with a power-law (`p = 8`) spacing that spends most steps at low σ.

> **Cap.** `inference_noise_schedule` returns the *uncapped* schedule (top = 2560), but
> `sample()` then clips it to `max_inference_sigma = 256` (default): drop all σ > 256 and
> prepend 256 (`:1816-1818`). So the loop actually starts at σ = 256, and the number of
> denoise steps run is **fewer** than `num_sampling_steps` whenever the raw schedule
> exceeds the cap. (Verified: with `num_sampling_steps=2`, raw schedule = `[2560, 0.0064,
> 0]` → capped run uses `[256, 0.0064, 0]`.)

### 5b. Churn — the stochastic ("temperature") knob
EDM's stochastic sampler optionally *re-noises* before each denoise ("churn"):
- `γ = gamma_0 (0.605)` where `σ > gamma_min (1.107)`, else `0` (`:1828-1832`). This bumps
  the working noise level from `σ_tm` up to `t_hat = σ_tm·(1+γ)`.
- The injected noise has std `eps_std = noise_scale · √(t_hat² − σ_tm²)` (`:1848`).

So **`noise_scale` (λ) is the temperature**: `λ = 0` (default) → `eps_std = 0` → a
deterministic **probability-flow ODE** whose only randomness is the initial draw; `λ > 0`
→ a genuine **SDE** that re-randomizes each high-σ step and explores more (it can hop
between basins). Note churn only fires where `γ > 0`, i.e. at the **high-σ** steps where
global shape (e.g. an antibody–antigen orientation) is decided — not in the final
low-σ refinement.

### 5c. The reverse step — `step_scale`
`x = x_noisy + step_scale·(σ_t − t_hat)·(x_noisy − x_denoised)/t_hat` (`:1899`) is the
first-order (Euler) EDM/ODE step. `step_scale = η` (default `1.0`) scales the step length.

### 5d. Frame handling — augmentation + Kabsch
- `_center_random_augmentation` (Alg 19, `:1718`): every step re-centers the coords at the
  atom centroid and applies a **random rotation + translation**. Structure is SE(3) so this
  doesn't change geometry, but it decorrelates the frame across steps and matches training
  augmentation.
- `_weighted_rigid_align` (Kabsch SVD, `:1742`): aligns `x_noisy` onto `x_denoised` *before*
  the Euler update so the step direction is computed in a consistent frame (`:1890-1894`).

### 5e. Early exit
`denoising_early_exit_rmsd` (`:1901-1921`): if consecutive `x_denoised` predictions agree
within an RMSD threshold, stop early. `None` (default) runs the full schedule.

---

## 6. Full parameter reference

### Architecture (`DiffusionModuleConfig`, `configuration_esmfold2.py:97`; `AtomAttentionConfig:63`)
| param | default | effect |
|-------|---------|--------|
| `sigma_data` | `16.0` | data scale; sets all EDM coefficients |
| `c_atom` | `128` | atom hidden dim |
| `c_token` | `768` | token hidden dim |
| `c_z` | `256` | pair (bias) dim |
| `c_s_inputs` | `451` | input-feature dim |
| `fourier_dim` | `256` | σ Fourier-embedding width |
| `atom_num_blocks` / `atom_num_heads` | `3` / `4` | atom transformer depth/heads (enc & dec each) |
| `token_num_blocks` / `token_num_heads` | `12` / `16` | token transformer depth/heads |
| `swa_window_size` | `128` | atom sliding-window attention span |
| `spatial_rope_base_frequency` | `20.0` | 3D-RoPE spatial base freq |
| `n_spatial_rope_pairs_per_axis` | `2` | 3D-RoPE spatial pairs/axis |
| `n_uid_rope_pairs` / `uid_rope_base_frequency` | `10` / `10000.0` | 3D-RoPE per-molecule-id encoding |

### Training noise (provenance only; inference-only port)
| param | default |
|-------|---------|
| `train_noise_log_mean` (μ) | `-1.2` |
| `train_noise_log_std` (s) | `1.5` |

### Sampler / schedule (`DiffusionStructureHeadConfig`, `:116`)
| param | default | effect |
|-------|---------|--------|
| `gamma_0` | `0.605` | churn amount where σ > `gamma_min` |
| `gamma_min` | `1.107` | σ threshold below which no churn |
| `noise_scale` (λ) | `0.0` | **temperature**: 0 = deterministic ODE; >0 = stochastic SDE |
| `step_scale` (η) | `1.0` | Euler step-length multiplier |
| `inference_s_max` | `160.0` | top of raw schedule = `σ_data·s_max = 2560` |
| `inference_s_min` | `4e-4` | bottom = `σ_data·s_min ≈ 0.0064` |
| `inference_p` | `8.0` | power-law curvature of the schedule |
| `inference_num_steps` | `68` | schedule length (denoise steps before cap) |

### `sample()` call-time args (`:1767`)
| arg | default | effect |
|-----|---------|--------|
| `num_diffusion_samples` | `1` | independent samples drawn in parallel (batch dim) |
| `num_sampling_steps` | `None`→`68` | overrides `inference_num_steps` for this call |
| `max_inference_sigma` | `256.0` | caps the schedule top (see §5a) |
| `noise_scale` / `step_scale` | `None`→config | per-call overrides of λ / η |
| `use_inference_cache` | `True` | cache σ-independent tensors (`z`, atom features, RoPE) across steps |
| `return_atom_repr` / `return_token_repr` | `False` | also return internal reps |
| `denoising_early_exit_rmsd` | `None` | early-stop threshold (§5e) |

---

## 7. Connection to the TTS noise sweep

The 7txw experiments (`docs/latent_structure_rep.md`, Idea 1) turn exactly these knobs.
`run/esm/run_7txw_loop_diffusion.py` forwards `--noise_scale`, `--step_scale`,
`--num_sampling_steps` (and `--num_diffusion_samples`, `--sample_batch`) straight into
`structure_head.sample()`:

- **`noise_scale`** is the lever for Idea 1. At the default `0.0` the sampler is a
  deterministic ODE, so a committed-bad seed's saved final-loop `z` always denoises to the
  same wrong basin; the only diversity across samples is the initial noise draw. Raising
  `noise_scale` turns on churn at the high-σ steps (where orientation is decided), letting
  the sampler explore whether a *suppressed* good basin is still reachable from that `z`.
- **`num_sampling_steps = 16`** (vs the 68 default) is used in that harness so the only
  variable vs the noise-0 baseline is `noise_scale`.
- **`max_inference_sigma`** is left at 256; it caps where the schedule starts.

See the noise-sweep run (`7txw_noise{0.5,1.0,1.5,2.0}`), scorer `eval/score_7txw_noise.py`,
and plot `analysis/plot_noise_sweep.py`.

---

## Appendix: runnable shape probe

Minimal end-to-end call (reuses the conditioning-build helpers from the loop-diffusion
harness). Run on a free GPU with the esm venv:

```python
import sys, numpy as np, torch
sys.path.insert(0, "/home/ebrahim/tts_abag/run/esm"); sys.path.insert(0, "/home/ebrahim/tts_abag")
from run_7txw_loop_diffusion import load_complex, build_conditioning
from esm.models.esmfold2 import ESMFold2InputBuilder
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from paths import ESM_PREDICTIONS, JSON_DIR

model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval(); model.set_chunk_size(32)
builder = ESMFold2InputBuilder()
feat, _ = builder.prepare_input(load_complex(str(JSON_DIR), "7txw"), seed=0, device=model.device)
with torch.no_grad(): cond = build_conditioning(model, feat)

z = torch.from_numpy(np.load(ESM_PREDICTIONS/"7txw_reps"/"7txw"/"seed_0"/"reps"/"pairformer_z_loop16.npy")).cuda().float()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    z = model.parcae_coda(model.parcae_readout(z), pair_attention_mask=cond["pair_mask"])

out = model.structure_head.sample(
    z_trunk=z.float(), s_inputs=cond["x_inputs"], s_trunk=None,
    relative_position_encoding=cond["rel_pos"], ref_pos=feat["ref_pos"],
    ref_charge=feat["ref_charge"], ref_mask=feat["atom_attention_mask"],
    ref_element=cond["ref_element_oh"], ref_atom_name_chars=cond["ref_name_oh"],
    ref_space_uid=feat["ref_space_uid"], tok_idx=cond["atom_to_token"],
    asym_id=feat["asym_id"], residue_index=feat["residue_index"], entity_id=feat["entity_id"],
    token_index=feat["token_index"], sym_id=feat["sym_id"],
    token_attention_mask=feat["token_attention_mask"],
    num_diffusion_samples=1, num_sampling_steps=2)
print(out["sample_atom_coords"].shape)   # (1, 4640, 3)
```

Verified shapes for 7txw (609 tokens, 4640 padded atoms; `num_diffusion_samples=1`):

```
INPUTS
  z_trunk (pair):                 (1, 609, 609, 256)
  s_inputs (per-token):           (1, 609, 451)
  relative_position_encoding:     (1, 609, 609, 256)
  ref_pos (per-atom):             (1, 4640, 3)
  atom_to_token:                  (1, 4640)

PER DENOISE STEP (internal)
  conditioning      -> s (1, 609, 768),  z (1, 609, 609, 256)
  atom_encoder      -> a (1, 609, 768),  q (1, 4640, 128),  c (1, 4640, 128)    # token, atom, atom
  token_transformer -> a (1, 609, 768)
  atom_decoder      -> r_update (1, 4640, 3)
  diffusion_module  -> x_denoised (1, 4640, 3)

OUTPUT
  sample_atom_coords:             (1, 4640, 3)

Karras schedule (num_sampling_steps=2, uncapped): [2560.0, 0.0064, 0.0]
```

With `num_diffusion_samples = S` the leading batch dim of every per-step / atom tensor
becomes `S` (the pair `z` stays `(1, …)` and is broadcast), which is how one trunk `z`
yields many independent samples.
