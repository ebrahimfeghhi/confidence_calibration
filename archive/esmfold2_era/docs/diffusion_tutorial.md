# ESMFold2 diffusion — notes

## 1. Adding noise to the data

**Notation.** $A$ = number of atoms · $x_{\text{clean}}\in\mathbb{R}^{A\times3}$ = clean atom coordinates · $x$ = noisy coordinates · $\varepsilon\sim\mathcal{N}(0,I)$ = standard Gaussian noise, shape $A\times3$ · $\sigma$ = noise level (a scalar) · $z\sim\mathcal{N}(0,1)$ = the standard-normal draw that sets $\sigma$ · $\sigma_{\text{data}}=16$, $\mu=-1.2$, $s=1.5$ = fixed schedule constants.

Start from the clean structure: atom coordinates $x_{\text{clean}} \in \mathbb{R}^{A \times 3}$.

Noise is added in **one shot**:

$$ x = x_{\text{clean}} + \sigma\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) $$

- $\varepsilon$ — standard Gaussian noise, same shape as $x$ (one independent draw per coordinate).
- $\sigma$ — the noise level, a single scalar that scales the noise.

How $\sigma$ is generated (one draw per training example), a **log-normal**:

$$ \sigma = \sigma_{\text{data}}\,\exp(\mu + s\,z), \qquad z \sim \mathcal{N}(0,1) $$

with $\sigma_{\text{data}} = 16$, $\mu = -1.2$, $s = 1.5$.

So: draw $z \sim \mathcal{N}(0,1)$ → compute $\sigma$ → add $\sigma\varepsilon$.

### Why $\mu, s$ (and $\sigma_{\text{data}}$) are hardcoded

They are not tuned per run or per protein. They are the fixed **EDM** noise schedule (Karras et al. 2022), with the specific values taken from **AlphaFold3** ($\sigma_{\text{data}}=16$, $\mu=-1.2$, $s=1.5$). All they do is decide *which noise levels the denoiser practices on during training* — the model is trained to invert $\sigma\varepsilon$ across the whole range of $\sigma$ this distribution produces, so the distribution has to cover that range with most weight where learning pays off.

**$\sigma_{\text{data}} = 16$ Å — the data scale.** It is roughly the standard deviation of atom coordinates in a real (centered) protein. EDM measures every noise level *relative* to this, so $\sigma \ll 16$ is "barely perturbed," $\sigma \approx 16$ is "noise comparable to the structure itself," and $\sigma \gg 16$ is "essentially pure noise." Hardcoding 16 just fixes the unit.

**$\mu = -1.2$ — where to center the noise.** Sampling $\sigma$ *log-normally* (Gaussian in $\ln\sigma$) spans orders of magnitude in one draw. The center sits at the median

$$\sigma_{\text{med}} = \sigma_{\text{data}}\,e^{\mu} = 16\,e^{-1.2} \approx 4.8\ \text{Å},$$

i.e. a few Å — the regime where backbone/side-chain detail is actually being resolved. A negative $\mu$ deliberately biases training toward these *low-but-not-tiny* noise levels: near-zero noise is trivial to denoise (no signal to learn) and pure noise carries no structure, so both extremes are uninformative and you don't want to waste samples there.

**$s = 1.5$ — how wide to spread.** In log space $\pm2s$ covers

$$z=-2:\ 16\,e^{-1.2-3.0}\approx 0.24\ \text{Å} \quad\text{(near-clean)} \qquad z=+2:\ 16\,e^{-1.2+3.0}\approx 97\ \text{Å} \quad\text{(near-pure-noise)},$$

so a single fixed distribution still exposes the model to the *entire* denoising trajectory it will walk at sampling time, from almost-clean to fully-destroyed. The long right tail also pulls the **mean** up to $\sigma_{\text{data}}\,e^{\mu+s^2/2}=16\,e^{-0.075}\approx 14.8$ Å $\approx \sigma_{\text{data}}$, so high-noise steps are still seen often enough to be learned.

Net: $\mu$ picks the *sweet spot* (a few Å, most informative), $s$ guarantees *coverage* (clean → pure noise), and $\sigma_{\text{data}}$ sets the *unit*. They're constants because they describe the training curriculum, not the input.

## 2. What the network is trained to predict

**Notation.** $x$ = noisy coordinates · $\sigma$ = its noise level · $x_{\text{clean}}$ = the clean coords we want back · $\theta$ = network weights · $F_\theta$ = the *raw* network · $D_\theta(x;\sigma)$ = the **denoiser** — $F_\theta$ wrapped so its output estimates $x_{\text{clean}}$ · $c_{\text{in}},c_{\text{out}},c_{\text{skip}},c_{\text{noise}}$ = EDM preconditioning coefficients (scalar functions of $\sigma$) · $\hat\varepsilon$ = the network's implied estimate of the noise · $p_\sigma$ = the data distribution blurred by noise level $\sigma$.

The network predicts the **denoised (clean) structure** — not the noise. The denoiser $D_\theta(x;\sigma)$ is trained so its output approximates $x_{\text{clean}}$. This is the **EDM "denoiser" (x0) parameterization**, the opposite of DDPM-style **ε-prediction** (predict the noise that was added).

### The raw network emits an update, not the coords

$D_\theta$ is a *wrapper*. The raw network $F_\theta$ outputs a per-atom update; EDM blends it with the noisy input via precomputed coefficients:

$$ D_\theta(x;\sigma) = \underbrace{c_{\text{skip}}(\sigma)}_{\sigma_{\text{data}}^2/(\sigma^2+\sigma_{\text{data}}^2)}\,x \;+\; \underbrace{c_{\text{out}}(\sigma)}_{\sigma_{\text{data}}\sigma/\sqrt{\sigma^2+\sigma_{\text{data}}^2}}\,F_\theta\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big). $$

- **High σ** ($c_{\text{skip}}\!\to\!0$, $c_{\text{out}}$ large): the input $x$ is mostly noise, so we **rely on the network output** — it must hallucinate global shape (e.g. the antibody–antigen orientation).
- **Low σ** ($c_{\text{skip}}\!\to\!1$): the input is nearly clean, so we **rely mostly on the noisy input** $x$ and $F_\theta$ only adds small refinements.

### Same as the paper

The paper writes this denoiser update as

$$ x_{\text{denoised}} = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + t^2}\,x_{\text{noisy}} \;+\; \frac{\sigma_{\text{data}}\cdot t}{\sqrt{\sigma_{\text{data}}^2 + t^2}}\,r_{\text{update}}, $$

which is **identical** to the box above — it just spells the coefficients out instead of naming them. The dictionary:

- their $t$ = our $\sigma$ (the noise level),
- their $x_{\text{noisy}}$ = our $x$,
- their $r_{\text{update}}$ = our $F_\theta(c_{\text{in}}x;\,c_{\text{noise}})$ (the raw network output),
- the coefficient on $x_{\text{noisy}}$, $\ \sigma_{\text{data}}^2/(\sigma_{\text{data}}^2+t^2)$, is exactly $c_{\text{skip}}$,
- the coefficient on $r_{\text{update}}$, $\ \sigma_{\text{data}}t/\sqrt{\sigma_{\text{data}}^2+t^2}$, is exactly $c_{\text{out}}$.

$c_{\text{in}}$ and $c_{\text{noise}}$ don't appear in the paper line only because they're folded *inside* $r_{\text{update}}$: the network already receives the preconditioned input $c_{\text{in}}x$ and the noise embedding $c_{\text{noise}}$, then emits $r_{\text{update}}$.

Two limits confirm it behaves sensibly:

- **$t \to 0$** (input already clean): $c_{\text{skip}}\to1$, $c_{\text{out}}\to0$, so $x_{\text{denoised}} \to x_{\text{noisy}}$ — pass the input straight through, nothing to fix.
- **$t \to \infty$** (input is pure noise): $c_{\text{skip}}\to0$, $c_{\text{out}}\to\sigma_{\text{data}}$, so the output is entirely network-driven.

### Why the paper calls it $t$ (not $\sigma$)

Same number, different letter: $t = \sigma$, the noise level. In EDM the noise level doubles as the **time variable of the sampling loop** — generation integrates from high noise down to ≈0, and $\sigma$ is exactly the quantity being decreased step by step, so it plays the role of "time." EDM sets time equal to the noise level outright ($\sigma(t)=t$), which is why the paper and code write $t$ where a from-scratch derivation would write $\sigma$.

Mind the powers: $t$ is the noise standard deviation $\sigma$ itself, **not** $\sigma^2$. In the formula, $t^2$ sits exactly where our $\sigma^2$ does and the lone $t$ where our $\sigma$ does — so $t^2 = \sigma^2$, with no hidden squaring. (The implementation also uses $\hat t$ / `t_hat` for a *temporarily bumped-up* noise level during the stochastic sampler — still a noise level, just nudged higher for one step.)

### "Predict clean" and "predict noise" are the same information

Once you have the clean estimate $D_\theta$, the added noise is recovered algebraically:

$$ \hat\varepsilon = \frac{x - D_\theta(x;\sigma)}{\sigma} \;=\; -\,\sigma\,\nabla_x \log p_\sigma(x) \quad(\text{the score}). $$

So x0-prediction, ε-prediction, and score-matching all carry the *same* information — EDM just picks the x0 form because it stays well-conditioned across the huge σ range. The **sampler still uses the noise form**: each generation step moves the coordinates along the direction $(x - D_\theta(x;\sigma))/\sigma$, which is exactly this $\hat\varepsilon$, scaled by how much $\sigma$ drops on that step. Each step → estimate $x_{\text{clean}}$ → take a small move along the implied $\hat\varepsilon$ toward lower noise → repeat.

## 3. The loss function

**Notation.** $\mathcal{L}$ = the training loss (a scalar to minimize over $\theta$) · $\mathbb{E}[\cdot]$ = average over the random draws of $\sigma$, $x_{\text{clean}}$ (a training structure), and $\varepsilon$ · $D_\theta(x;\sigma)$ = the denoiser from §2 · $\lambda(\sigma)$ = a **scalar weight that multiplies the squared error**, depending on the example's noise level · $\lVert\cdot\rVert^2$ = sum of squared coordinate errors over all atoms · $c_{\text{out}}$ = the EDM output coefficient from §2.

The denoiser is trained by simple **weighted least squares** against the clean coordinates: noise up a clean structure, denoise it, and penalize how far the result is from the original.

$$ \mathcal{L} = \mathbb{E}_{\sigma,\,x_{\text{clean}},\,\varepsilon}\Big[\,\lambda(\sigma)\,\big\lVert D_\theta(x_{\text{clean}} + \sigma\varepsilon;\ \sigma) - x_{\text{clean}}\big\rVert^2\,\Big]. $$

Read it inside-out: draw a structure $x_{\text{clean}}$, draw a noise level $\sigma$ (the §1 log-normal) and noise $\varepsilon$, form the noisy input $x_{\text{clean}}+\sigma\varepsilon$, run the denoiser, and measure the squared distance back to $x_{\text{clean}}$.

**The target is the rigid-aligned ground truth.** Coordinates are only defined up to a rigid motion (rotation + translation), so before computing the error the ground truth is Kabsch-aligned onto the prediction — otherwise a correct structure in a different pose would be wrongly penalized.

**What $\lambda(\sigma)$ does.** Without it, high-σ examples (errors of tens of Å) would dwarf low-σ examples (sub-Å) and hijack the gradient. EDM sets

$$ \lambda(\sigma) = \frac{1}{c_{\text{out}}(\sigma)^2} = \frac{\sigma^2+\sigma_{\text{data}}^2}{(\sigma\,\sigma_{\text{data}})^2}, $$

which exactly cancels the $c_{\text{out}}$ scaling in $D_\theta$, so the *effective* regression target seen by the raw network $F_\theta$ has unit variance at **every** σ. It is the loss-side partner of $c_{\text{in}}$ (which unit-scales the input): together they make every noise level contribute equally to learning.

## 4. Inference: sampling a structure

Training (§2–3) gave us a denoiser $D_\theta(x;\sigma)$ that, from a noisy structure, guesses the clean one. **Generation runs it in reverse**: start from pure noise and walk *down* a ladder of shrinking noise levels, denoising a little at each rung, until the noise reaches 0 and what's left is a structure. This whole section unpacks the loop the paper calls *Truncated Diffusion Sampling* (Algorithm 4).

**Notation.** $x$ = the working coordinates, updated in place (starts as pure noise, ends as the prediction) · $\{\sigma_k\}_{k=0}^{N}$ = the **noise schedule**, strictly decreasing, $\sigma_0$ = highest (start), $\sigma_N=0$ (clean) · $N$ = number of steps · $M$ = how many independent samples are drawn at once (the batch leading dim is $M\times B$ for $B$ proteins) · $\varepsilon\sim\mathcal{N}(0,I)$ = fresh Gaussian noise, redrawn each place it appears · $D_\theta$ = the trained denoiser from §2 · $x_{\text{denoised}}$ = its clean-structure guess at the current rung · $\hat t$ = the noise level actually fed to the denoiser (= $\sigma_{k-1}$, possibly bumped up by churn) · $d$ = the noise/score direction $(x-x_{\text{denoised}})/\hat t$ · $\eta$ = **step scale**, an Euler step-length multiplier (default 1) · $\sigma_{\text{data}}=16$ = data scale from §1 · $\sigma_{\max}$ = a cap on the starting noise level (the "truncated" part, §4.4).

> **Heads-up on $\lambda$ and $\gamma$.** The churn knobs $\gamma,\gamma_0,\gamma_{\min}$ and the temperature $\lambda$ (paper's *noise scale*) come in §4.2 — leave them at $\gamma=0,\lambda=0$ for now. ⚠️ This $\lambda$ is **not** §3's loss-weight $\lambda(\sigma)$; the paper reuses the letter for an unrelated knob.

### 4.1 The deterministic skeleton

Switch churn off ($\gamma=0,\lambda=0$) and ignore the frame bookkeeping for a moment. Then one step — going from rung $\sigma_{k-1}$ down to the next rung $\sigma_k$ — is just three lines:

1. **Denoise.** $x_{\text{denoised}} = D_\theta(x;\ \sigma_{k-1})$ — the network's best guess of the finished structure, seen from the current noise level.
2. **Find the noise direction.** $d = \dfrac{x - x_{\text{denoised}}}{\sigma_{k-1}}$ — exactly the $\hat\varepsilon$ of §2: "which way, and how far, is the noise."
3. **Step down.** $x \leftarrow x + \eta\,(\sigma_k - \sigma_{k-1})\,d$.

The sign is the whole trick. We go to a *lower* rung, so $\sigma_k - \sigma_{k-1} < 0$, which makes step 3 move $x$ **against** $d$ — i.e. *toward* $x_{\text{denoised}}$ — by an amount set by how much the noise level just dropped. $\eta$ (default 1) scales that move. (These three lines are Algorithm 4's lines 8, 10, 11 with churn removed.)

**Why not just output $x_{\text{denoised}}$ in one shot?** At a high noise level the denoiser can't resolve detail — its guess is a blurry average over all structures consistent with the noisy blob — so you trust it only a little, drop to a slightly lower noise level, and ask again. Many small, self-correcting steps beat one big leap. Concretely this is Euler integration of the **probability-flow ODE** $\frac{dx}{d\sigma}=d$: walk $\sigma$ from $\sigma_0$ down to $0$, nudging $x$ by $d\,\Delta\sigma$ at each step.

**Start and finish.** The loop is seeded at the top of the ladder, $x = \sigma_0\,\varepsilon$ (pure noise at the largest noise level), and after the last step ($\sigma_N=0$) the noise is gone and $x$ is the predicted structure. With $\lambda=0$ the run is fully **deterministic**: the *only* randomness is that initial draw, so one noise seed → one structure. (This is exactly the `noise_scale = 0` ODE behavior the 7txw experiments lean on.)

### 4.2 Churn — the stochastic ("temperature") version

**The problem with §4.1.** The deterministic loop is *greedy*. From its random starting point it rolls straight downhill into the nearest valley and settles. If that valley holds a mediocre structure, there's no escape — the loop can never climb back uphill to look elsewhere.

**The fix: shake before you settle.** Churn gives the structure a small random **kick** right before each denoise — a do-over. The kick can bounce it out of a bad valley; the denoiser is the gravity that immediately pulls it back down onto the landscape of realistic structures. Repeat over many steps and the sample settles into a *deeper* (more probable) valley than plain rolling would ever reach. This is just **simulated annealing**: random heat to escape bad spots, an energy that pulls back toward good ones.

How hard you kick is a single number — the **temperature** $\lambda$ (the `noise_scale` knob):

- $\lambda = 0$ — no kick → back to the deterministic §4.1 loop, one seed → one structure. *(The shipped default.)*
- larger $\lambda$ — harder kicks → more exploration; kick too hard and the structure never settles.

**What one step looks like now.** Each step grows a short prelude. Instead of *denoise → step down*, it becomes:

> nudge the noise level **up** a notch (the kick) → denoise → step **down**, past where you started.

So every step is a small **up-then-down**: the "up" is the do-over, the "down" is the real progress to the next-lower rung. Nothing else about the §4.1 step changes.

That's the whole idea. The rest is just *how big is the kick* — skip it on a first read if you like.

#### The kick, in detail

The kick has two independent settings:

- **How high to aim.** Raise the working noise level to $\hat t = \sigma_{k-1}(1+\gamma)$ — about $1.6\times$ with the default $\gamma_0 = 0.605$. This only switches on while the noise level is still high ($\sigma_k > \gamma_{\min}=1.107$); on the final low-noise steps it's off and you're back to §4.1 exactly.
- **How much noise that takes.** Moving a field from level $\sigma_{k-1}$ up to $\hat t$ means adding independent noise of size $\sqrt{\hat t^{\,2}-\sigma_{k-1}^2}$ — variances add, so this just "fills the gap." The temperature scales it: injected noise $=\lambda\sqrt{\hat t^{\,2}-\sigma_{k-1}^2}$. At $\lambda=1$ you land exactly on $\hat t$; at $\lambda=0$ you add nothing.

After the kick, denoise at $\hat t$ and take the same Euler step as §4.1 (with $\sigma_{k-1}$ replaced by $\hat t$).

**Why the kicks are only early.** They're gated to the high-noise steps because that's where the *global* decisions happen — overall fold, domain placement, antibody–antigen orientation. Those are the choices worth re-rolling; the low-noise tail is fine detail you don't want to disturb. This is exactly why the 7txw work raises `noise_scale`: a committed-bad seed is frozen into one orientation under $\lambda=0$, and kicking the high-noise steps is the only way to test whether a better orientation is still reachable.

> **Footnote — a wrinkle in the default.** The two settings above are truly independent: $\hat t$ is bumped based purely on the noise level, while $\lambda$ controls the injected noise. So at the default $\lambda=0$ the run is fully deterministic, yet $\hat t$ is *still* bumped on the early steps — the denoiser is asked about noise level $\hat t\approx1.6\,\sigma_{k-1}$ on a field that really holds only $\sigma_{k-1}$. That doesn't break determinism (no randomness is added); it just means the default isn't quite the bare §4.1 integrator. What flips a run deterministic↔stochastic is $\lambda$ alone. (Code: `gammas = where(schedule > gamma_min, gamma_0, 0)`, then `eps_std = λ·√(t̂²−σ²)`.)

### 4.3 Keeping the frame straight

**Why this is even a thing.** A protein structure has no "correct" position or orientation in space — slide the whole thing across the room, or spin it, and it's the same structure. The network and the loss are built to not care (they're *SE(3)-invariant*: blind to rigid motion). That freedom is convenient, but it has a price during sampling: if two copies of a structure are floating in different poses, comparing them atom-by-atom is meaningless until you line them up. Two lines in the loop handle this.

**RigidAlign — line up before you step.** The Euler step measures the gap between the noisy structure and the denoised prediction (the direction $d$ from §4.1) and moves along it. But the denoiser can hand its prediction back in *any* pose — nothing forces it to match the noisy input's orientation. If it doesn't, that atom-by-atom gap is dominated by a meaningless whole-body rotation/shift instead of the actual shape correction you want. So you first **rigid-align** the noisy coords onto the prediction — the classic *Kabsch* step: find the single rotation + translation that best overlays one on the other — and *only then* take the difference. Now $d$ is pure shape change ("how this structure needs to deform"), with the irrelevant pose mismatch subtracted out.

**CenterRandomAugmentation — re-center and re-roll the pose each step.** Before each step the structure is recentered at its midpoint and given a fresh random rotation/translation. Because geometry is pose-invariant this changes nothing physical, but it earns two things: it keeps the coordinates from drifting away from the origin (where the denoiser was trained to operate), and it reproduces the random-orientation augmentation the model saw in training — so any small pose-dependent quirks in the network get averaged out across steps rather than compounding.

Net: **RigidAlign makes the before/after comparison frame-consistent; CenterRandomAugmentation keeps the working structure in a training-like, drift-free frame.** Both exist purely because structures live in a world with no absolute coordinate grid.

### 4.4 The noise ladder (and what "truncated" means)

**The rungs aren't evenly spaced.** The schedule is the list of noise levels $\sigma_0 > \sigma_1 > \dots$ the loop walks down, ending at exactly 0. You'd expect even spacing, but no — the rungs are **bunched toward the low-noise end**. The reasoning: at high noise the structure is a coarse blob and a big jump costs nothing; at low noise you're resolving fine detail and need many small, careful steps. A single "curvature" exponent sets how lopsided the spacing is. You choose three things — a top level, a bottom level (the smallest *nonzero* one), and how many steps — plus the implied final 0.

> **In detail (skip on a first read).** This is the *Karras power-law schedule*. Lay $N$ points evenly, interpolate them between top and bottom **in "$p$-th-root space,"** then raise back to the power $p$ — that power is the warp that bunches the rungs:
> ```
> σ_i = σ_data · ( s_max^(1/p) + (i/(N−1))·(s_min^(1/p) − s_max^(1/p)) )^p ,   σ_N = 0
> ```
> Defaults $p=8$, $s_{\max}=160$, $s_{\min}=4\mathrm{e}{-4}$, $N=68$, $\sigma_{\text{data}}=16$ → raw top $16\cdot160=2560$, bottom $\approx0.0064$, last entry 0. The large $p$ is why most of the 68 steps land in the low-σ refinement zone.

**"Truncated" = don't start from the very top.** That raw top of 2560 is absurdly noisy — about 160× the data scale. The structure there is *already* indistinguishable from pure noise, so the first steps coming down from 2560 accomplish nothing: the denoiser sees noise and can only guess the average structure. So sampling **caps the start**: a `max_inference_sigma = 256` setting throws away every rung above 256 and begins the walk at 256. That's still ~16× the data scale — fully randomized, structurally blank — so you lose nothing, you just skip dead steps. (Side effect: the number of denoise steps actually run is a bit fewer than $N$, since the top of the ladder was dropped.)

That cap is the "Truncated" in *Truncated Diffusion Sampling*.

> **Aside — could we drop the cap for more diverse predictions?** You can (set `max_inference_sigma=None`), but it won't help. Both 256 and the raw 2560 are far above the data scale (16), so the input there is *informationally dead* — pure noise either way — and the extra high-σ steps inject no new "choice." In deterministic mode ($\lambda=0$) the high-σ flow is even mildly *contractive* (it funnels different starts toward the same basins), so you'd pay compute for the same, or slightly *less* diverse, outputs. Diversity is limited by the lack of injected randomness ($\lambda=0$) and the strongly-committing conditioning $z$ — not by the cap. The real levers are **churn** ($\lambda>0$) and **perturbing $z$**.

## 5. Inside one denoise call (`DiffusionModule`)

So far $D_\theta$ has been a black box: noisy structure + noise level in, clean guess out. This section opens it. It's the most architectural part of the model — more boxes-and-arrows than equations — so the little math there is parked in skip-able "in detail" blocks.

**Notation.** *atom rep* = one feature vector per atom (thousands of them) · *token rep* = one feature vector per residue (hundreds) · $z$ = the trunk's **pair** representation, a residue×residue table encoding how every pair of residues relates (this is the conditioning that carries the fold) · $s$ = a per-residue conditioning vector that also carries the current noise level · *skip connections* = atom-level detail the encoder stashes for the decoder.

### 5.1 Two resolutions — the key idea

A structure is ultimately a pile of **atoms** (3D points), thousands of them. But "how should this protein fold, where does the antigen sit" is reasoning best done per **residue** (amino acid) — far fewer, only hundreds. The denoiser works at *both* resolutions and bounces between them:

- **Atom view** — fine-grained and geometric, where the actual coordinates live. Good for *local* detail (which atoms sit near which).
- **Token (residue) view** — coarse, one summary vector per residue, where the *global* reasoning happens: overall fold, the antibody–antigen interface. This is where the expensive all-to-all attention runs and where the trunk's conditioning $z$ plugs in.

Why both? You can't do everything at atom level — all-to-all attention over thousands of atoms is far too expensive, and the real decisions live at the residue level anyway. But you can't do everything at token level either — the thing being denoised is *literally atom coordinates*, so you need atom resolution to read them in and write them out.

So one denoise call flows **atom → token → atom**:

1. **Atom encoder** — read the noisy atoms, do cheap *local* processing, summarize each residue's atoms into one token. *(zoom out)*
2. **Token transformer** — the heavy *global* reasoning across residues, conditioned on $z$ and the noise level. *(§5.3)*
3. **Atom decoder** — push the updated residue-level information back down to atoms and emit a coordinate update. *(§5.4, zoom back in)*

### 5.2 Atom encoder — zooming out (atoms → tokens)

The encoder turns raw noisy coordinates into per-residue summaries the transformer can reason over.

- **What goes in.** The noisy coordinates (rescaled to a sane range), plus fixed per-atom facts: which element, its charge, which residue it belongs to, a reference geometry, the atom's name. Only the coordinates change each step; the rest is constant and cached.
- **Local mixing.** Each atom looks only at *nearby* atoms — a small sliding window, not the whole protein — so it's cheap. The mixing is geometry-aware (it knows the 3D layout), so each atom builds a picture of its immediate neighborhood.
- **Pool into tokens.** All the atoms of one residue are summarized into a single per-residue vector. Thousands of atom vectors collapse to hundreds of token vectors — now small enough for global reasoning.
- **Keep a backup.** Some atom-level detail is stashed aside (skip connections) and handed straight to the decoder later, so the fine geometry isn't lost in the pooling.

In a phrase: the encoder is the **zoom-out** — from a cloud of atoms to a tidy per-residue summary, while pocketing the fine detail for the trip back.

> **In detail (skip-able).** The local mixing is *sliding-window attention* (window ≈128 atoms) with **3D rotary position embeddings** built from reference positions — that's what makes it geometry-aware. "Pool into tokens" is a scatter/aggregation from each atom to its residue. Working widths: atoms at 128, tokens at 768. The coordinate rescale is exactly the $c_{\text{in}}$ unit-variance scaling from §2. Note the encoder takes *no* trunk conditioning in the shipped path — it runs on geometry alone (see §5.3).

### 5.3 Token transformer — the global reasoning (where the conditioning lives)

This is the heart of the denoiser. We now hold a few hundred residue tokens, and the transformer lets **every residue attend to every other** — global, all-to-all reasoning, affordable only because we're at residue resolution, not atom. This is where the model decides the overall arrangement: which residues pack together, how domains sit, where the antigen docks.

**This stage is where essentially *all* the trunk conditioning acts.** Two things steer it:

- **The pair rep $z$ biases the attention.** $z$ is the trunk's residue×residue table of how each pair of residues relates. It's added as a *bias* to the attention scores — nudging "residue $i$ should attend to residue $j$" according to what the trunk learned about that pair. This is the channel through which the trunk's fold information actually shapes the structure, and **$z$ enters here and nowhere else** — not the encoder, not the decoder.
- **The noise level $s$ modulates each block.** The per-residue conditioning $s$ (which carries the current noise level) is mixed into the tokens just before the transformer and used to modulate every block. This is how the network knows *how noisy it currently is*, so it can reshape boldly at high noise and only nudge at low noise.

So if you want to know "where does the antibody–antigen orientation get decided," it's here: a committed orientation is one that $z$'s attention bias keeps reinforcing across all the token-transformer blocks.

> **In detail (skip-able).** 12 blocks, 16 heads, token width 768. The pair bias is *AttentionPairBias* ($z$ → additive per-head bias on the attention logits); the noise modulation is *adaptive LayerNorm* (AdaLN) driven by $s$. The atom encoder *can* accept a trunk single-rep, but it's `None` at inference, which is why the trunk only acts at this stage.

### 5.4 Atom decoder — zooming back in (+ the EDM combine)

After the transformer, each residue token holds an updated, globally-informed summary. The decoder pushes that back down to atoms:

- **Broadcast tokens to atoms.** Each residue's updated token is handed to all of its atoms, and combined with the atom-level detail the encoder stashed (skip connections), so the fine geometry comes back.
- **Local mixing again.** Another cheap round of local atom attention.
- **Emit a coordinate update.** Produce one 3D update vector per atom — the network's proposed correction to the structure.

The decoder never sees $z$ or $s$ directly; it inherits the conditioning **second-hand**, already baked into the token vectors the transformer produced.

Finally the **EDM combine** closes the loop — the §2 box: blend the per-atom update with the noisy input via $c_{\text{skip}}/c_{\text{out}}$. High noise → trust the update; low noise → mostly keep the input. The result is $x_{\text{denoised}}$, the clean-structure estimate that §4's sampling loop then steps toward.

**One denoise call, end to end:** zoom out (encoder) → reason globally with $z$ + noise level (transformer) → zoom back in (decoder) → EDM blend. The conditioning lives at the token transformer; the atom stages just translate between coordinates and tokens.

> **In detail (skip-able).** Decoder = gather token→atom (add to the encoder's skip), another sliding-window atom transformer (≈3 blocks), then a linear map to a per-atom 3-vector $r_{\text{update}}$. The combine is $x_{\text{denoised}} = c_{\text{skip}}\,x_{\text{noisy}} + c_{\text{out}}\,r_{\text{update}}$ from §2.
