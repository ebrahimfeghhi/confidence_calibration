# Latent structure in the pairformer rep: does 7txw encode both orientation basins?

## Question

For 7txw, the antibody–antigen complex either docks in the correct antigen orientation
("good", A|C DockQ ≈ 0.9) or a wrong one ("bad", ≈ 0.03). Does the pairformer trunk
representation `z`:

- **(A) encode both basins simultaneously** (a superposition), with the diffusion head just
  realizing the dominant one — i.e. the good basin is *latent* in bad seeds and the diffusion
  head "can't find it"? or
- **(B) fully commit to one basin and delete the other**?

## Key evidence: diffusion samples SPLIT at the transition loop

From the diffuse-from-each-loop sweep (`run/esm/run_7txw_loop_diffusion.py`, scored to
`results/csv_files/dockq_7txw_loopLL.csv`), the 5 diffusion samples drawn from a *single*
trunk rep behave like this across recycling loops for the 4 good seeds:

| seed | behavior at / around the snap |
|------|-------------------------------|
| 10 | loop03 `[0.65, 0.25, 0.04, 0.03, 0.02]` → loop04 all ~0.92 |
| 14 | loop05 `[0.88, 0.85, 0.85, 0.72, 0.03]` → loop06 all good |
| 33 | loops 5–8 linger at ~0.30 (half-committed) → loop09 snaps to ~0.9 |
| 39 | loop06 `[0.62, 0.59, 0.57, 0.52, 0.49]` (all intermediate); still one `0.03` straggler at loops 10–11 → concordant by loop12 |

The decisive observation: **at the transition loop, the same `z` produces both a correctly
docked (~0.9) and a wrong (~0.03) structure.** A 0.9 DockQ requires the correct *global*
antigen orientation — diffusion noise alone cannot stumble into that from a single-basin
conditioning. So the split means **`z` genuinely encodes both basins at that loop, and the
diffusion head samples both.**

## Answer: a dynamical collapse, not a static choice

Neither (A) nor (B) is right as a static description — it is **time-dependent within recycling**:

- **Early / mid recycling: both basins are encoded** (the splits prove it) and both are
  diffusion-accessible.
- **Late recycling: the trunk commits to one basin and the other becomes inaccessible.** By
  ~loop 12 the splits stop, samples are concordant, no 0.03 stragglers remain. The loser is not
  merely down-weighted — it drops out of the diffusion-accessible landscape.
- The commitment is **progressive and winner-take-all** (seed 33's 4-loop plateau; seed 39's
  slow sharpening), and for "separatrix" seeds the winner is decided by tiny run-to-run
  nondeterminism — exactly the ~6% **metastable** seeds seen across runs
  (see `docs/mask_pattern_plan.md` reproducibility note).

### This refutes the "diffusion head can't find it" hypothesis

Whenever `z` encodes both basins, the diffusion head finds both (the splits). So the diffusion
head is **not** the bottleneck. A committed-bad seed yields only bad structures because, by the
final loop, **`z` no longer encodes the good basin** — the trunk deleted it — not because
diffusion failed to recover an encoded-but-minor alternative. The commitment happens upstream,
in the recurrent trunk.

## What is still open

"Deleted" vs "retained below the detection floor." Five samples cannot see a <~5%-weight basin,
so the splits stopping by loop 12 is strong suppression but not proof of *total* deletion. The
final `z` could still carry a sub-threshold latent trace of the loser.

## Two experiments to settle it

### Idea 1 — many-sample / high-temperature diffusion from the *final* committed `z`
Take a committed-bad seed's final-loop trunk rep and sample diffusion ~100× with elevated
`noise_scale` (the sampler exposes `noise_scale` / `step_scale` / `max_inference_sigma`). 
- If the good basin **never** reappears → the loser is effectively **deleted**.
- If it reappears at low rate → it is **latent but suppressed** (and diffusion *can* find it with
  enough sampling pressure).
This directly probes whether a competing basin survives in the committed `z`. Reuses the
`structure_head.sample` path already wired in `run_7txw_loop_diffusion.py`.

### Idea 2 — distogram bimodality across loops (nearly free, no folding)
The trunk emits `distogram_logits` — an explicit probability distribution over pairwise
distances — *before* diffusion (`modeling_esmfold2.py` forward, just after `parcae_coda`).
Inspect the **A×C block** distogram across loops for marginal / metastable seeds:
- **bimodal** (two distance modes = two orientations) collapsing to **unimodal** would show the
  deletion happening in the model's own probabilistic readout, with no diffusion sampling at all.
- Most informative on metastable / transition seeds, where (A) predicts a visible second mode.
- Because every trunk layer is the *identical* `trimul → trimul → MLP` (no attention confound),
  reading the distogram per layer × recycle isolates pure triangle-multiplication iteration —
  you can watch the bimodal→unimodal collapse and attribute it to specific layers.
Capture is cheap (one `[L,L,bins]` distogram per loop, same hook pattern as the rep capture).

## Mechanistic hunch — and why it fits ESMFold2's trunk specifically

ESMFold2's trunk has **no attention on the pair representation at all**. Each trunk layer
(`PairUpdateBlock` in `modeling_esmfold2_common.py`) is exactly:

    tri_mul_out      (TriangleMultiplicativeUpdate, outgoing)
    tri_mul_in       (TriangleMultiplicativeUpdate, incoming)
    pair_transition  (per-edge MLP)

There is **no triangle attention** — AF2/AF3 pairformer has it, ESMFold2 dropped it (it's the
expensive part). The model's only attention (`AttentionPairBias`, `SWA3DRoPEAttention`) runs on
the *single*/atom rep, which is pooled out of the pair rep *after* the trunk
(`RowAttentionPooling`) to feed the diffusion/confidence heads — i.e. attention *reads out* the
geometry, it never *refines* it.

So the entire antigen-orientation commitment happens in a recurrence whose only pair-mixing
operation is the bilinear, consistency-propagating triangle multiplication
(`z_ij ← Σ_k z_ik · z_kj`) plus a per-edge MLP. Triangle multiplication couples every pair
through all third residues, so a local "maybe here, maybe there" is ironed into one globally
coherent geometry over iterations; two incompatible *global* orientations cannot both be a
consistent distance set, so a persistent 50/50 superposition is geometrically frustrated.

Crucially, there is **no multi-head / attention machinery in the pair path that could carry two
structural hypotheses in parallel subspaces** — unlike a general transformer trunk. This is a
**dynamics / attractor** property, not a capacity wall: the 256-dim pair rep *can* hold a
transient mixture (that's the splits above), but iterated trimul has single-geometry attractors
that collapse it. Multiplicity therefore lives in the distogram *distribution* and in diffusion
sampling — never as two stable geometries in the pair tensor.

## Pointers
- Per-loop per-sample DockQ: `results/csv_files/dockq_7txw_loopLL.csv`;
  plot `analysis/plot_dockq_loops.py` → `analysis/figs/dockq_iptm_across_loops_7txw.png`.
- Diffuse-from-each-loop harness: `run/esm/run_7txw_loop_diffusion.py`.
- Related: antigen-block localization & the per-seed "snap"
  (`pairformer-z-separates-good-seeds`); orientation framing (`tts-orientation-mask-null`).
