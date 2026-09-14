# The summed kernel repulsion loss

*A from-fundamentals explanation of the loss that steers ESMFold2-Fast away from
already-visited docking basins. Self-contained; math is written in plain ASCII so
it reads correctly whether or not your viewer renders equations.*

---

## Notation

- `P` — the candidate's **cross-distogram**: the model's predicted pairwise-distance
  distribution over the antibody×antigen (interface) residue-pairs, taken from a short
  K=3 forward pass. For each residue pair it is a softmax over 64 distance bins.
- `anchor_a` — the cross-distogram of a **past attempt** (`a = 1 … N`). Every run we
  keep, so `N` grows over the campaign. These are the "history."
- `W1(P, Q)` — **Wasserstein-1** distance between two distograms. Here it is the L1
  distance between the two 64-bin distance **CDFs**, averaged over the interface pairs
  (`w1` in `steer/core/forward_grad.py`). Think of it as "how different are these two
  interface geometries," one nonnegative number.
- `d_a = W1(P, anchor_a)` — the candidate's distance to anchor `a`.
- `h` — the **bandwidth**: a distance that sets how far a single anchor's influence
  reaches (roughly "one basin wide").
- `bias(P)` — the loss we are about to build.

---

## 1. The problem the loss has to solve

We run the model over and over. After each run we have a list of distances from the new
candidate to every previous attempt: `d_1, d_2, …, d_N`. We want to steer the next
candidate into a **new** region of interface-geometry space — somewhere the anchors are
**not**.

So the entire job of the loss is: **turn that list of `N` distances into one number**
whose gradient points toward empty space. Everything below is about choosing that
number well.

What we want from it, stated as plain requirements:

1. **Small (good) in empty regions, large (bad) in crowded ones.** The loss should
   report how "occupied" the candidate's current location already is.
2. **Locality.** An anchor sitting in a completely different basin should have almost no
   say in where I go next. Only nearby anchors — the basin I am currently trying to
   leave — should push.
3. **Accumulation.** Ten anchors piled in one basin should push ten times harder than
   one. Redundant visits are how a basin gets *filled* so the search leaves it and does
   not drift back.
4. **Smoothness.** We optimize it by gradient descent, so it must be differentiable with
   well-behaved gradients — no hard edges, no cusps.

The next two sections build a function that satisfies all four. Section 2 builds the
contribution of **one** anchor; Section 3 sums them.

---

## 2. One anchor's "hill" — deriving the kernel

Take a single anchor and its distance `d` to the candidate. We need a function `f(d)`
that measures "how much does *this* anchor say I'm crowded." From requirements (1) and
(2):

- `f` should be **large when `d` is small** (I'm right on top of the anchor → very
  crowded),
- `f` should **fall to ~0 when `d` is large** (the anchor is far → irrelevant),
- `f` should be **smooth** with a **characteristic reach** — a distance beyond which the
  anchor stops caring.

That shape is a **bump**: a hill centered on the anchor that decays to zero. We now
motivate the three pieces of `exp(-(d/h)^2)` one at a time.

### 2a. Why exponentiate — `exp(-…)`

We need a smooth, positive function that starts at its maximum when `d = 0` and decays
toward `0` as `d` grows. `exp(-<something that grows with d>)` is the natural choice:

- **Always positive**, and it maps `0 → 1` (full crowding right at the anchor) and
  `∞ → 0` (no crowding far away). Exactly the range we want.
- **Infinitely smooth.** Its derivatives are all continuous, so the loss surface has no
  kinks and gradients are well-behaved everywhere.
- **No hard boundary.** Compare the alternatives:
  - A hard cutoff (`1 if d < r else 0`) is discontinuous → gradient is zero almost
    everywhere → useless for optimization.
  - A linear ramp (`max(0, 1 - d/r)`) has a kink at `d = r` and abruptly-zero gradient
    beyond it.
  `exp` gives a soft falloff that is differentiable at every distance, with no cliff.
- **Self-scaling gradient.** The derivative of `exp(-g(d))` is `-g'(d)·exp(-g(d))`, i.e.
  it carries a copy of itself, so the push it produces automatically shrinks as you get
  far away — locality falls out for free.

So `exp(-…)` converts "distance" into a smooth, decaying **influence weight**. What goes
inside the `exp` decides the *shape* and *reach* of that decay — that's the next two
pieces.

### 2b. Why divide by `h` — the bandwidth

`d` is a physical distance (in W1 units). You cannot exponentiate a raw distance and get
a meaningful shape, for two reasons, both fixed by dividing by a reference distance `h`:

- **Units.** `exp` needs a **dimensionless** argument. `d/h` is "how many bandwidths
  away am I" — a pure number. Without `h`, `exp(-d^2)` would silently declare the reach
  to be `d ≈ 1` in whatever arbitrary units W1 happens to use.
- **Reach control.** `h` is the knob that says **how far one anchor reaches**:
  - at `d = h`, the contribution is `exp(-1) ≈ 0.37`,
  - at `d = 2h`, it is `exp(-4) ≈ 0.018` (essentially gone).

  So an anchor meaningfully repels out to about one `h`, and is invisible past ~`2h`.
  Choosing `h ≈ the width of a docking basin` is what makes requirement (2), locality,
  concrete: anchors in *other* basins (many `h` away) contribute ~0. This is why `h` is
  later calibrated from the actual spacing between anchors rather than hand-set
  (Section 5).

### 2c. Why square — `(d/h)^2`

We need a power of `d/h` inside the exponent (a bare `exp(-d/h)` also decays, but has the
wrong shape). The square — giving the **Gaussian / RBF kernel** `exp(-(d/h)^2)` — is the
right one, for two connected reasons:

- **Shape near the center: a rounded top, not a spike.** With the square, the exponent
  `-(d/h)^2` is a parabola, so the hill is **flat at its peak** (`d = 0`) and curves
  down smoothly. With `|d|` instead (the "Laplacian" kernel `exp(-|d|/h)`), the exponent
  has a **kink at `d = 0`** → the hill comes to a sharp **cusp** where the gradient is
  discontinuous. Squaring makes the loss twice-differentiable *everywhere*, anchors
  included.
- **The push lands where you want it — a shell, not a plateau.** The gradient magnitude
  of one hill is proportional to `(d/h)·exp(-(d/h)^2)`. Trace it across distance:
  - at `d = 0` it is **zero** (you're at the flat top — smooth maximum, no direction
    singularity, and in practice you're never exactly on an anchor),
  - it **peaks at `d = h/√2 ≈ 0.707 h`** — intermediate range, i.e. *exactly the "just
    leaving the basin" zone* where you most want a strong shove,
  - it decays back to **~0 far away** — the locality property again.

  This "strong in a shell at mid-range, weak both at the center and far off" profile is
  precisely what basin-escape wants. The `|d|` (Laplacian) version instead has a roughly
  **constant-magnitude** push near the anchor — it keeps pushing just as hard whether
  you're deep inside or on the edge, which is closer to the failure mode we are trying to
  *fix* (see Section 6). The square localizes the force into a shell.

- **It's a real density kernel.** `exp(-(d/h)^2)` is the Gaussian used in kernel density
  estimation. Modeling each anchor as a little Gaussian blob of "already-occupied space"
  is what makes the *sum* in Section 3 an interpretable density — with standard theory
  for how to pick `h`.

### 2d. The one-anchor hill, in numbers

Putting 2a–2c together, one anchor contributes `k(d) = exp(-(d/h)^2)`. Its **crowding
value** and the **push it exerts** (∝ `(d/h)·k(d)`) as a function of how many bandwidths
away you are:

| `d/h` | crowding `exp(-(d/h)^2)` | relative push `(d/h)·exp(-(d/h)^2)` |
|------:|-------------------------:|------------------------------------:|
| 0.00  | 1.000                    | 0.000  (flat top — no direction)    |
| 0.25  | 0.939                    | 0.235                               |
| 0.50  | 0.779                    | 0.389                               |
| 0.707 | 0.607                    | **0.429  (strongest push)**         |
| 1.00  | 0.368                    | 0.368                               |
| 1.50  | 0.105                    | 0.158                               |
| 2.00  | 0.018                    | 0.037                               |
| 3.00  | 0.0001                   | ~0.000                              |

Read the right column top-to-bottom: no push at the dead center, hardest push as you
cross out through ~0.7 bandwidths, fading to nothing past ~2 bandwidths.

---

## 3. Summing the hills — the full loss

Now add up every anchor's hill:

```
bias(P) = sum over a of  exp(-(d_a / h)^2)        # minimize this
```

This is exactly a **kernel density estimate** (up to a constant) of the anchors,
evaluated at the candidate `P`. In words: *a smooth count of how densely you have already
sampled near here.* Minimizing `bias` moves the candidate to where that density is
lowest — the most unexplored interface geometry. That satisfies requirements (1) and (2)
directly.

### Sum vs. mean of the *same* kernel is a wash under Adam

It is tempting to say "the sum fills the basin, the mean can't." **That is wrong when you
optimize with Adam.** Given the same kernel, `bias = sum_a k(d_a)` and its average
`(1/N) sum_a k(d_a)` differ only by the constant `1/N`, and Adam is (to first order)
invariant to a global constant on the loss: scaling the loss by `N` scales the gradient by
`N`, the first moment `m` by `N`, the second moment `v` by `N^2`, and `N` cancels in the
`m/sqrt(v)` update. Summed-kernel and mean-kernel therefore take the **same steps** —
identical gradient *directions*, and Adam discards the only thing that differed (the
magnitude). (The `eps` in Adam's denominator breaks exact invariance, but `eps ≈ 1e-8` is
negligible against real gradients.)

So the property that actually fixes the current loss is **locality** — the kernel's
distance-decay from Section 2 — **not** the sum-vs-mean normalization. See Section 4.

Two reasons to still *write* it as a sum, neither about the gradient step:

- The un-normalized value is a genuine **density** — a "how full is this basin" reading
  that keeps growing as you revisit, whereas the mean saturates. That makes the sum the
  natural signal for the optional well-tempered stop criterion (using the loss *value*,
  not the step).
- It changes the *step* only in cases Adam does not cover: under plain SGD the `1/N`
  becomes a growing effective learning rate as `N` climbs, and if the kernel term is ever
  **added** to another fixed-weight loss term, the `N` shifts the balance between them.
  Neither applies to the single-term Adam setup today.

The metadynamics picture still holds for the *landscape* (each anchor deposits a repulsive
hill; the pile fills visited basins), but under Adam that filling is driven by the kernel's
locality reshaping the gradient *direction*, not by the un-normalized magnitude.

---

## 4. What the gradient actually does (the details)

Differentiating the loss with respect to the candidate:

```
grad_P bias = sum over a of  w_a * grad_P d_a ,   with   w_a = (2 d_a / h^2) * exp(-(d_a/h)^2)
```

Each anchor pushes the candidate along `grad_P d_a` (the direction that *increases*
distance from that anchor), weighted by `w_a`. And `w_a` is exactly the "bump" from
Section 2d: essentially **0 for far anchors**, **0 for the one you're sitting on**, and
**peaked for anchors at basin range (`d ≈ 0.707 h`)**.

So the net force is dominated by the anchors you are *currently escaping* and ignores
both the irrelevant far ones and the degenerate exactly-coincident one. Two consequences
worth stating explicitly:

- **It fixes the current loss's "self-weakening" — and the fix is locality, not the sum.**
  The current loss maximizes the mean of **raw W1** distances, and raw W1 has
  near-**constant-magnitude** gradient at every range, so a far anchor pushes about as hard
  as a near one. When anchors surround the candidate these equal pushes **partially cancel**
  and the net direction washes out — worse the more you have explored. (Not hypothetical: on
  8im0 the current loss was seen letting a run drift back to within W1 ≈ 0.024 of an
  already-visited basin core.) Replacing raw W1 with the localized kernel is what removes
  this: far anchors weigh ~0, so they cannot cancel the local push. Note this is the
  *kernel* doing the work — a mean-of-kernels fixes it just as well as a sum-of-kernels
  (Section 3).
- **The `d = 0` zero-gradient is harmless.** It is a smooth maximum, not a singular
  point, and the initialization nudge plus the many other nearby (nonzero-gradient)
  anchors keep the candidate moving. You never sit exactly on an anchor in practice.

---

## 5. The bandwidth `h` (the one real knob)

`h` is the KDE bandwidth, and standard density-estimation intuition applies:

- **Too small** → each hill is a narrow spike. Anchors become isolated; between them the
  summed `bias` is nearly flat, so the gradient is ~0 and the search barely moves. Escape
  is slow.
- **Too large** → the hills oversmooth into one broad blob; the loss stops distinguishing
  basins and starts behaving like the mean again.
- **Just right** → neighboring anchors' hills overlap by ~`exp(-1) ≈ 0.37`, so a
  well-visited basin merges into one filled region while distinct basins stay separate.

So `h` should be about **the typical spacing between anchors** — the natural width of a
basin. Two rules make this work in practice:

- **Calibrate `h` from anchor-to-anchor spread, NOT candidate-to-anchor.** Use the
  anchors' own nearest-neighbor W1 (the median NN distance is a good statistic), tracked
  incrementally as each new anchor is appended — `O(N)` per run, since you only compare
  the new anchor to the stored ones, never the full `O(N^2)` matrix. Do **not** derive
  `h` from the current candidate's distances (recomputed each step): that makes `h` chase
  wherever the candidate currently sits, a feedback loop that destroys the fixed "far
  anchors → ~0" length scale that locality depends on. `h` should describe the *explored
  landscape*, not the momentary probe.
- **Run 0 / single anchor needs no special case.** With one anchor there is no
  nearest-neighbor spread to measure — but with a single anchor `bias = exp(-(d/h)^2)` is
  monotonic in `d`, so its descent **direction** is identical to the mean loss's
  (move away from the anchor) *regardless of* `h`. Here `h` only scales the step size, not
  the direction, so any value is safe until a second anchor exists to calibrate against.

If a single global `h` ever proves too blunt (dense and sparse regions wanting different
widths), the natural upgrade is a **variable bandwidth** — give each anchor its own
`h_a = its nearest-neighbor distance` — before reaching for anything fancier.

---

## 6. Contrast with the alternatives

The axis that matters is **locality** (is a far anchor down-weighted?), and it is set by the
per-anchor *function*, not by whether you sum or average:

| loss | per-anchor term | localized? | note |
|---|---|---|---|
| **current** | raw `d_a`, then `mean`, maximized | **no** — raw W1 pushes at ~constant magnitude at every range | far anchors cancel the near push → direction washes out |
| **softmin** | ~`min_a d_a` | partially — nearest only | discards every anchor but one |
| **kernel** (sum *or* mean) | `exp(-(d_a/h)^2)` | **yes** — reach set by `h`, far anchors ~0 | the fix; sum vs mean is equivalent under Adam (Section 3) |

So the kernel's advantage over the current loss is the **kernel itself** (locality with a
tunable reach `h`). The "summed" part is a labeling / stop-signal convenience, not an
optimization difference under Adam.

---

## 7. How it plugs into the pipeline

Concretely, per run of `repel_proven.py`:

1. Forward-pass the current `s_lm` (+ optionally `z0`) for K=3 loops → candidate
   cross-distogram `P`.
2. Compute `d_a = W1(P, anchor_a)` for all stored anchors.
3. `bias = sum_a exp(-(d_a/h)^2)`; take `M` gradient steps on `s_lm` (+`z0`) to
   **minimize** `bias`.
4. Append the run's final `P` to the anchor list; update `h` from the anchors' NN spread.
5. Every 16 runs, decode the optimized state through the full 17-loop trunk + diffusion
   head and score (monitoring only — not part of the loss).

Two implementation notes that keep the change isolated to the loss branch:

- **Sign convention is free.** The file already routes everything through one `md` with
  `loss = -md`, and treats *higher `md` = more repelled = better* for the console log, the
  CSV `repel` column, and the plateau-warn check. Set `md = -bias` in the kernel branch
  (so `loss = -md = bias` is minimized, and low `bias` = high `md` = "more repelled").
  Every downstream consumer then works unchanged. (Only caveat: the numeric *scale* of
  `md` differs from mean-mode, so don't compare the two modes' logged values, and make
  sure the plateau check is slope-based, not an absolute threshold.)
- **The loss operates on whatever distogram blocks you feed it** — the antibody×antigen
  interface block always, plus the antigen×antigen block when a target refolds. Nothing
  about the kernel assumes "interface only," so it stays general.

---

## 8. Where this sits in the literature

- **Kernel density estimation / novelty search / potential fields.** `bias` is a KDE of
  visited states; minimizing a local density estimate to move toward the least-crowded
  region is the classic novelty-search / RBF potential-field mechanism. Not new — which
  is reassuring; it's a known-good primitive.
- **Particle Guidance** (repulsion as a sum of pairwise kernels during sampling) uses the
  same kernel, but over particles sampled **in parallel with no memory**.
- **Metadynamics / well-tempered metadynamics** deposits accumulating repulsive hills
  along a trajectory — the **sequential** picture.

The delta here is the *combination*: Particle-Guidance's kernel run **sequentially /
accumulating** (the metadynamics variant), in a folding trunk's **latent space**, to a
short K=3 horizon, **label-free**, to escape antibody-antigen **docking** basins.
