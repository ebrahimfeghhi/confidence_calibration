# How the steering gradients are computed: `s_lm` vs `z0`

A code-anchored explanation of the gradient computation behind the 7txw steering
experiments — how we differentiate a short-horizon distogram loss back to two different
internal tensors of ESMFold2-Fast's recurrent trunk, and *why* the two gradients behave so
differently. All code references are to `analysis/grad_steer/forward_grad.py` unless noted.

This doc is self-contained; every fact needed is inlined. Math renders on GitHub.

---

## 0. What we are differentiating, and why it needs custom plumbing

We want the gradient of a scalar loss `L` (how far the predicted antibody×antigen geometry
is from the good basin G) with respect to one internal tensor — either the **LM single
representation `s_lm`** or the **initial recurrent pair state `z0`** — with *all model
weights frozen*. We then take gradient steps on that tensor to push the prediction into G.

The stock `ESMFold2Model.forward` is wrapped in `@torch.inference_mode()`, which forbids
autograd entirely. So `forward_grad.py` rebuilds the trunk forward outside inference mode:

- `load()` (line 49) loads the model, calls `model.requires_grad_(False)` so **no parameter
  gradients** are ever allocated, and moves the 6B-param ESMC LM to CPU (we start from a
  cached `s_lm`, so ESMC is not in the graph for these experiments).
- The tensor being steered is made the sole autograd **leaf**:
  `leaf = tensor.detach().clone().requires_grad_(True)`.
- The recurrence runs under `with torch.enable_grad()` + bf16 autocast, with per-loop
  gradient checkpointing (§5).

**Notation.**
- `L` — number of tokens (609 for 7txw). `d = 256` — pair-channel width.
- `s_lm` — LM single rep, shape `[1, L, 256]`.
- `z_init` — the deterministic, data-derived pair embedding injected every loop, `[1, L, L, 256]`
  (`compute_z_init`, line 64). Same every loop; depends only on the inputs, never on the leaf.
- `z_t` — the recurrent **pair state** at loop `t`, `[1, L, L, 256]`. `z_0 = z0`.
- `K` — the steering **horizon**: we read the loss after `K` recycling loops.
- `a` — per-channel SSM decay, a length-256 vector with every entry in `(0,1)`
  (`_discretized_dynamics`, modeling file: `a = exp(-delta * exp(log_a))`).
- `P_K` — the antibody×antigen cross-distogram read from `z_K` (a per-pair distribution over
  distance bins). `P_G` — the mature-G target distogram. `L = W1(P_K, P_G)`.

---

## 1. The shared forward path: leaf → distogram → loss

Both levers feed the same recurrence. One recycling loop is `_loop_body` (line 107):

```python
refined  = model.lm_encoder(lm_z, pair_attention_mask=pair_mask)   # lm_z = lift of s_lm
z_inject = z_init + refined
injected = model.parcae_input_norm(z_inject)                        # LayerNorm
z        = a * z + F.linear(injected, b_mat)                        # the SSM update
z        = model.folding_trunk(z, pair_attention_mask=pair_mask)    # 24 pair-update blocks
```

So the recurrence is, writing `FT` for the 24-block folding trunk and bundling the input
terms into a single tensor `u`:

$$z_{t+1} = \mathrm{FT}\!\big(\,a \odot z_t + u\,\big), \qquad u = W_b\,\mathrm{LN}\!\big(z_\text{init} + \mathrm{LMenc}(\text{lm\_z})\big)$$

Two facts that drive everything:
1. **`u` does not depend on the state `z_t`** — it is rebuilt from `z_init` and the LM signal,
   which are fixed across loops. So `∂u/∂z_t = 0`.
2. **`u` is re-added at every loop**, while `z0` only sets the loop-0 value.

After `K` loops, `cross_distogram_from_z` (line 140) maps `z_K` to `P_K`
(`parcae_readout → parcae_coda → distogram_head(z + zᵀ) → softmax → AB×C block`), and the
loss is `w1(P_K, P_G)` (line 163). `distogram_from_s_lm` (line 152) chains lift → trunk →
distogram; `run_trunk` (line 119) is the loop driver.

Both branches differ only in **which tensor is the leaf and where it enters `u` or `z_0`.**

---

## 2. The `s_lm` gradient (persistent lever)

Here the leaf is `s_lm`. It enters through `lift_to_lm_z` (line 95,
`lm_z = base_z_mlp(s_lm)`, a SingleToPair + LayerNorm), so `u` is a function of `s_lm` and is
injected **identically at every loop**. `z0` is held fixed.

Because `u` enters at every loop, `z_K` depends on `s_lm` through `K` separate injection
sites. By the chain rule the gradient is a **sum over those sites**:

$$\frac{\partial L}{\partial s_{lm}} = \frac{\partial L}{\partial z_K}\;\Bigg(\sum_{t=0}^{K-1} \frac{\partial z_K}{\partial u}\bigg|_{\text{site }t}\Bigg)\;\frac{\partial u}{\partial s_{lm}}$$

The contribution of the injection at loop `t` is attenuated by the loops *after* it:

$$\frac{\partial z_K}{\partial u}\bigg|_{\text{site }t} \approx \Bigg(\prod_{s=t+1}^{K-1} J_{\mathrm{FT},s}\,\mathrm{diag}(a)\Bigg) J_{\mathrm{FT},t}$$

The key point: the **last injection** (`t = K-1`) is attenuated by *nothing* — it lands one
folding-trunk step before the readout. So no matter how large `K` is, there is always a
strong, barely-attenuated term in the sum. **The `s_lm` gradient does not vanish with `K`.**
This is the formal reason s_lm steering transfers at every horizon: a steered `s_lm` reshapes
the input that gets re-injected on every remaining loop.

---

## 3. The `z0` gradient (transient lever)

Here the leaf is `z0`, which is *only* the loop-0 state (`z_0 = z0`); `s_lm` (hence `u`) is
held fixed. `z0` therefore enters at a **single site**, and the gradient is a single chained
product through all `K` loops:

$$\frac{\partial L}{\partial z_0} = \frac{\partial L}{\partial z_K}\;\prod_{t=0}^{K-1}\frac{\partial z_{t+1}}{\partial z_t}, \qquad \frac{\partial z_{t+1}}{\partial z_t} = J_{\mathrm{FT},t}\,\mathrm{diag}(a)$$

(The `+u` term drops out of `∂z_{t+1}/∂z_t` because `∂u/∂z_t = 0`.) So

$$\frac{\partial L}{\partial z_0} \approx \frac{\partial L}{\partial z_K}\;\Big(J_{\mathrm{FT}}\,\mathrm{diag}(a)\Big)^{K}.$$

Every factor carries a `diag(a)` with `a < 1`, so the gradient is **attenuated geometrically
in `K`** — roughly `a^K` times the folding-trunk Jacobians. For small `K` it is informative;
for large `K` it **vanishes**. This is exactly what we observed empirically: at K=9 the z0
optimization never moved (the best loss equalled the starting loss — a dead gradient), while
K=3 worked. `z0` is a fading initial condition, so its leverage lives only in an early window.

---

## 4. The contrast in one line

| | enters the recurrence… | gradient structure | behavior in `K` |
|---|---|---|---|
| **`s_lm`** | as `u`, **re-injected every loop** | **sum** over `K` injection sites; last term un-attenuated | **persistent** — robust at all `K` |
| **`z0`** | as the **loop-0 state only** | **single product** of `K` Jacobians, each `× diag(a)` | **transient** — decays `~a^K`, vanishes at large `K` |

Same loss, same trunk, same autograd machinery — the entire difference is *where the leaf
enters the SSM recurrence*. Re-injection (sum, no decay) vs initial-condition (product,
geometric decay) is the mechanism behind every s_lm-vs-z0 result.

---

## 5. Implementation details that matter for correctness

- **Frozen model, single leaf.** `model.requires_grad_(False)` (line 52) means the only
  gradient buffer is the leaf's; backprop computes activations' grads but stores no
  parameter grads. Memory and compute scale with the trunk forward, not the 6B LM.
- **Gradient checkpointing.** `run_trunk` wraps each loop body in
  `torch.utils.checkpoint(..., use_reentrant=False)` (line 135), and `FoldingTrunk` checkpoints
  each of its 24 blocks. Only loop-boundary `z_t` are kept; the inner activations are
  recomputed during backward. Without this, `K` loops × 24 triangle-attention blocks at
  `L=609` would not fit in memory.
- **Precision.** The trunk runs under `torch.autocast(bf16)`; the leaf is kept in fp32 so the
  optimizer state and the accumulated gradient are full precision. ESMC fp8 is **off**
  (`_esmc_fp8=False`) — fp8 kernels are not cleanly differentiable.
- **Determinism / isolating one lever.** `lm_dropout = 0` (the per-loop LM noise is skipped),
  and the random initial pair state is drawn **once** and reused (`fixed_z0`, line 73) so the
  loss is a deterministic function of the leaf across optimization steps. When steering
  `s_lm`, `z0` is pinned to that fixed draw; when steering `z0`, the lifted `lm_z` is pinned to
  the baseline. Each run therefore moves exactly one lever. (Residual CUDA-kernel
  nondeterminism remains but does not change the RNG draw order.)

---

## 6. The loss gradient (`W1` over distance bins)

`P_K` is a softmax over distance bins, so it is smooth in `z_K`. The objective `w1` (line 163)
is the 1-Wasserstein distance over *ordered* bins, computed as the L1 distance of the CDFs:

$$W_1(P,Q) = \frac{1}{N_\text{pairs}}\sum_{\text{pairs}} \sum_{\text{bins}} \Big| \mathrm{cumsum}(P) - \mathrm{cumsum}(Q) \Big|.$$

`cumsum` and `sum` are linear (constant Jacobians); the only non-smooth piece is `|·|`, whose
subgradient is `sign(cumsum(P) − cumsum(Q))`. So `∂L/∂P_K` is just a signed, bin-cumulative
mask — cheap and well-defined — and the chain rule from §2 / §3 carries it back to the leaf.
Using `W1` rather than KL matters because the bins are ordered distances: `W1` penalizes
*how far* probability mass sits from the target, giving a smoother steering signal than a
bin-wise divergence.

---

## Pointers

- Forward/grad plumbing: `analysis/grad_steer/forward_grad.py`
  (`lift_to_lm_z`, `_loop_body`, `run_trunk`, `cross_distogram_from_z`, `distogram_from_s_lm`, `w1`).
- The SSM recurrence and the `a = exp(-delta·exp(log_a))` discretization live in the
  ESMFold2 modeling file (`_discretized_dynamics`, `_init_pair_state`) — see
  `docs/parcae_ssm_recurrence.md` for the recurrence itself.
- Steering experiments that use these gradients: `steer_horizon.py` (s_lm),
  `steer_z0.py` (z0), `steer_converge.py` (per-step convergence).
