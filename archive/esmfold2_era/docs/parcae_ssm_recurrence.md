# The Parcae Trunk: a State-Space Recurrence over Recycling

ESMFold2's folding trunk does **not** recycle in the AlphaFold sense (re-embedding the
predicted structure). Its recycling loop is a **linear time-invariant state-space model
(SSM)** — an S4-style *diagonal* SSM — wrapped around triangle-multiplicative folding
layers. This note writes out that recurrence and the meaning of its $A$, $B$, $\Delta$.

## Notation

- $L$ — number of tokens (residues); $L = 609$ for 7txw.
- $c$ — pair-rep channel dimension, $c = 256$. Every residue pair $(i,j)$ carries a $c$-vector.
- $z_t \in \mathbb{R}^{L\times L\times c}$ — the **recurrent state** (pair rep) at recycling loop $t$, $t=0,\dots,16$.
- $z_{\mathrm{init}} \in \mathbb{R}^{L\times L\times c}$ — the **fixed input** pair rep, built once from sequence + positional features.
- $\ell$ — the ESMC language-model representation (computed once per seed).
- $\odot$ — element-wise (channel-wise) product.

Every operation below acts **per pair** $(i,j)$ on its $c$-vector, **identically for all pairs**
(the SSM is position-invariant). So $A \in \mathbb{R}^{c}$ broadcasts over channels and
$B \in \mathbb{R}^{c\times c}$ is applied to each pair's $c$-vector.

## The per-loop update

Each recycling loop $t$ performs three steps:

**1. Build the injected input** (rebuilt fresh every loop):

$$ u_t = \mathrm{LN}\!\big(z_{\mathrm{init}} + \mathrm{LMenc}(\mathrm{dropout}(\ell))\big) $$

The LM dropout is re-sampled each loop; with `lm_dropout = 0` it is the identity. Note the LM is
added to the **fixed** $z_{\mathrm{init}}$, not accumulated.

**2. SSM state update** (the subject of this note):

$$ z_t \;=\; A \odot z_{t-1} \;+\; B\,u_t $$

**3. Refine with the folding layers** (4 triangle-multiplication blocks):

$$ z_t \;\leftarrow\; \mathrm{Fold}(z_t) $$

$z_t$ is carried to loop $t+1$. The state is initialized **random** (truncated normal), *not*
from $z_{\mathrm{init}}$.

## The SSM parameters $A$, $B$, $\Delta$

All three are **learned** and **fixed** — not input-dependent (this is what makes it
time-invariant / *non-selective*, unlike Mamba). They are built from three raw parameters via a
continuous→discrete (zero-order-hold) discretization with per-channel step size $\Delta$:

$$ \Delta \;=\; \mathrm{softplus}(\theta_\Delta) \;\in\; \mathbb{R}_{>0}^{\,c} $$

$$ A \;=\; \exp\!\big(-\,\Delta \odot \exp(\theta_a)\big) \;\in\; (0,1)^{c} $$

$$ B \;=\; \mathrm{diag}(\Delta)\,B_{\mathrm{cont}} \;\in\; \mathbb{R}^{c\times c} $$

| symbol | role | shape | learned params |
|---|---|---|---|
| $A$ | per-channel **decay / keep gate** (diagonal) | $c$ | $\theta_a,\theta_\Delta$: $2c = 512$ |
| $B$ | **input / write** matrix (mixes channels) | $c\times c$ | $B_{\mathrm{cont}}$ (identity-init): $c^2 = 65{,}536$ |
| $\Delta$ | per-channel **step size** (enters both $A$ and $B$) | $c$ | shared $\theta_\Delta$ |

Total $\approx 66\text{k}$ params for the whole recurrence — tiny next to the folding layers
(millions). $A$ is a **vector** (diagonal SSM): $256$ independent scalar gates, each in $(0,1)$.
$B$ is a full matrix.

## Intuition: a bank of leaky integrators

Unroll the recurrence (ignoring the folding refinement):

$$ z_t \;=\; \sum_{k=0}^{t} A^{\,t-k} \odot \big(B\,u_k\big) $$

An input injected $n = t-k$ loops ago is weighted by $A^{\,n}$. Because each channel of $A$ lies
in $(0,1)$,

$$ A^{\,n} \;=\; \exp\!\big(-\,n\,\Delta \odot \exp(\theta_a)\big) \;\longrightarrow\; 0 \quad (n \to \infty), $$

so each channel is an **exponential moving average** over the loop axis: old inputs decay
geometrically. *Repeated multiplication is exponentiation* — that is exactly the mechanism, and
the reason it is "exponential."

Each channel's **memory timescale** (loops until an input's influence drops to $1/e$) is

$$ \tau \;\approx\; \frac{1}{\Delta \cdot \exp(\theta_a)} \quad \text{loops}. $$

Equivalently, in continuous time a channel decays at rate $r = \exp(\theta_a)$, one loop is a step
of duration $\Delta$, and the surviving fraction per loop is $A = e^{-r\Delta}$.

- $A \to 1$ (small $\Delta\,e^{\theta_a}$): long memory, slow integrator.
- $A \to 0$ (large $\Delta\,e^{\theta_a}$): short memory, reacts to the current loop only.

The $256$ channels learn a spread of timescales — a bank of leaky integrators.

## Why $A$ is bounded but $B$ is not

- $A$ is applied **repeatedly** ($A^{\,n}$), so it **must** satisfy $0 < A < 1$ or the state
  diverges. The $\exp(-\mathrm{softplus}(\cdot)\exp(\cdot))$ form **guarantees** $A \in (0,1)$ for
  *any* learned values — stable by construction.
- $B$ is applied **once** to each loop's input; the bounded $A$ then decays that contribution. So
  $B$ needs no constraint and is a **free** learnable matrix ($u_t$ is already LayerNorm'd, keeping
  its scale controlled). Standard SSM design: *constrain the recurrence, free the input map.*
- $\Delta$ **couples** them: a larger $\Delta$ means a *stronger write* (via
  $B = \mathrm{diag}(\Delta)B_{\mathrm{cont}}$) **and** a *faster forget* (smaller $A$). So $\Delta$
  is a single per-channel **"responsiveness vs. memory"** dial.

## Why this matters for recycling

Because the state **integrates** the per-loop injected input across the 17 loops, the trunk's
basin commitment is a **gradual accumulation**, not a one-shot decision: a small per-loop push —
from the deterministic LM mask, or the per-loop dropout noise — builds up through the long-memory
channels until the state crosses into a basin. (Empirically, good 7txw seeds escape the default
B1 basin at loops $\approx 4\text{–}12$.)

## Code anchors

- recurrence: `transformers/.../modeling_esmfold2.py`, `_run_one_loop` (≈ L778–848) — SSM update at
  L845 `z = a * z + F.linear(injected_pair, b_mat)`, folding at L846.
- discretization: `_discretized_dynamics` (≈ L742–746).
- raw params: `parcae_log_a = zeros(c)`, `parcae_log_delta`, `parcae_b_cont = eye(c)` (≈ L552–560); $c = 256$.
- folding block: `PairUpdateBlock` in `modeling_esmfold2_common.py` (≈ L2598) —
  `tri_mul_out → tri_mul_in → pair_transition`, residual connections, inference dropout rate $= 0$.
