# Plan: forward-only diverse-basin beam/population search (`confornet/beam_search.py`)

> **Status: PROPOSED — not yet implemented (as of 2026-06-24).** Design only; no code written.
> Reflects the refinements from the design discussion (noise lever moved off the pair state to a
> per-particle `s_lm`/affine conditioning offset; distinct-z0 batch self-test; prune protects rare
> clusters; relative cluster threshold; width auto-sized to ~80–130/round). An earlier Ultraplan
> cloud version exists but lacks these last deltas. Mirror of `~/.claude/plans/radiant-knitting-matsumoto.md`.

## Context

The 7txw antibody-antigen docking problem has a coverage bottleneck, not a selection
bottleneck: best-pair ipTM already ranks good seeds at the top at scale (AUC ~0.94), but the
good antigen-orientation basin **G** is reached only by an undirected mask+dropout lottery, so
you need thousands of seeds to cover it. We want to make diversity **directed and cheap** so a
handful of runs cover the distinct basins (B1 default decoy, B2 confident decoy, G good).

From the gradient-steering work we established why this is hard and what to exploit:
- Basins are **discrete attractors**; the trunk's recurrence "rounds" each trajectory into one
  basin over the 16 loops, turning a smooth early landscape into plateaus separated by cliffs.
- Escape B1->G is a **late, discrete snap (loops ~3-12)**; **early loops are uninformative**, and
  the future-G particle is indistinguishable from the B1 crowd early on.
- Per-loop **noise (dropout) is load-bearing** for escape — undirected noise already finds G
  sometimes; that is the lever we amplify.

User constraints (firm): **no training / gradient-free**, must **generalize to any ab-ag pair**
(no G-centroid targets, no force-masking specific residues, discover basins as you go), run on a
**96 GB GPU** (maximize it), and **validate on 7txw first** (only pair with basin labels +
baseline + ground truth in the bundle). Pruning: **whatever is most likely to work given the VRAM**
-> no early pruning; hold a large population, never discard the still-invisible escaper, spread
particles every loop, cluster only at the end.

## Approach — prune-and-fan beam (repulsion = selection, not a computed push)

Run the ESMFold2-Fast recycling trunk **forward-only** (no autograd, no gradient checkpointing —
this is what frees the VRAM), with particles batched as one tensor `z [B,L,L,256]`. Repulsion is
done by **selection, not a geometric push**: to literally push particles apart you'd need a
direction in basin/distogram space while you can only perturb the input/state — translating between
them is a Jacobian (the gradient we're avoiding), so a z-space shove is only a weak proxy. Instead:

Each **round** (a short segment of loops) — **fan out** every survivor into `F` children by adding
*undirected* noise to its **conditioning** (a per-particle `s_lm` offset / channel affine — never the
pair state `z`; see detail 3), recycle the wide batch one segment, read each child's
**cross-block distogram**, **cluster**, and **keep one representative per cluster** (prune back to
`K`). Selection-by-clustering IS the repulsion: you generate blindly and retain only the spread,
measuring *actual* output basins (no Jacobian needed) and dropping off-manifold junk at the same
step. Repeat. Anneal the fan-out noise **down** in the late loops so survivors settle cleanly into
basins (recycling needs the full 16 loops to sharpen). At the end, decode the final representatives
with the diffusion head and select by pair ipTM.

**Generality (firm constraint).** The algorithm uses ONLY pair-agnostic signals: cross-distogram
**W1** for clustering/pruning, **noise scale relative to running `||z||`**, and **pair ipTM** for
final selection. It must carry no 7txw-specific priors — no G/B1/B2 centroids, no pairformer-z PCA,
no "keep the B1 cluster", no fixed residue masks, no G-radius targets — inside any decision the
algorithm makes. The 7txw centroids (`basins.load_centroids`), wg/wb1/wb2 labels, and any PCA view
are **diagnostics only**: computed for logging/validation, never fed into clustering, pruning, or
selection. Strip them and the algorithm is unchanged.

**Why prune-and-fan, not a fixed population (user's point):** with pruning the *momentary* fan-out
width can be as wide as VRAM allows, then collapses to `K` at near-zero memory — so the next burst
can be just as wide. A fixed population ties up VRAM carrying all `B` particles every loop and only
ever explores those `B` trajectories; prune-and-fan reseeds *fresh* noise every round and explores
far more total trajectories, **reseeding at loops 3-12 where the escape to G actually happens.**
Same VRAM, more coverage.

**The one safeguard against culling the invisible escaper:** prune by **density-aware
cluster-keep-one-per-cluster**, NOT by pure farthest-point spread. The *dense majority cluster*
(on 7txw that's the B1 decoy, but the rule is stated without naming it) always survives, so its
representative preserves the seedbed the escaper grows from, and the fan-out re-rolls the escape
lottery from it each round. Pure farthest-point pruning chases oddball outliers and could drop the
whole majority — that is the failure mode to avoid. This is a generic density property, not a
7txw prior.

## New files

- **`confornet/beam_search.py`** — the experiment (forward-only population search + clustering +
  optional decode). Pair name is a flag (`--name 7txw`), code stays pair-agnostic.
- **`confornet/beam_one.sh`** — launcher mirroring `confornet/confornet_one.sh`: pick a free GPU,
  set `CUDA_VISIBLE_DEVICES`, run with `$ESM_PYTHON`, `--tag g$idx --seed-base $((idx*1000000))`.
  Run two independent instances (one per free GPU) for double coverage; results merge by CSV.

## Reuse (do not re-implement)

From `confornet/forward_grad.py`: `load`, `compute_z_init`, `fixed_z0`, `single_rep_from_hidden`,
`load_hidden`, `lift_to_lm_z`, `_loop_body`, `w1`, `BASELINE`, `N_LOOPS`, `AB`, `C`,
`model._discretized_dynamics()` (a/b_mat setup is `forward_grad.py:123-125`).
From `confornet/basins.py`: `load_centroids` (7txw G/B1/B2 from the bundle; diagnostics only).
From `run/esm/run_7txw_loop_diffusion.py`: `diffuse_from_z`, `save_samples` (survivor decode; each
sample exposes `.iptm/.ptm/.pair_chains_iptm`). Minimal-save + peak-mem-print patterns from
`steer_confornet.py`; survivor-decode pattern from `decode_confornet.py`.

## Key implementation details

**1. Forward-only loop driver.** Replicate `run_trunk` without `checkpoint`:
```python
@torch.no_grad()
def trunk_loops(model, z, z_init, lm_z, pair_mask, a, b_mat, n_loops, hook=None):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n_loops):
            z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
            if hook is not None: z = hook(z, i)   # separating force / metrics
    return z
```
Only `z` carries across loops; `z_init`, `lm_z` (baseline conditioning, fixed), `pair_mask`, `a`,
`b_mat` are loop-invariant. `FoldingTrunk` only calls `checkpoint` when grad is enabled, so
`no_grad` gives the plain forward for free.

**2. Batching B>1.** Stack particles into `z [B,L,L,256]`. `z_init [1,...]` and `pair_mask [1,L,L]`
**broadcast** — do not expand them. `lm_z` is now **per-particle `[B,...]`** (each particle has its
own conditioning, see step 3) — recompute it per loop rather than holding it resident. Needs:
- **Mandatory batch-equivalence self-test (`--selftest`, run once before anything else):** build
  **distinct** z0 for 3 particles, run each singly, then batched as `[3,...]`, and assert each
  batched slice matches its single run (bf16 tol). Use *distinct* z0 — identical copies can't catch a
  batch-collapse `[0]`-broadcast bug (slice 0 would equal all slices anyway). If it fails, stop and
  fall back to a per-particle Python loop (slower, still forward-only).
- batched `z0`: stack `fixed_z0(model, z_init, seed=i)` for i in 0..B-1 (distinct draws = initial
  diversity), or call `_init_pair_state` on a `[B,...]` ref.
- new `cross_distogram_batched(model, z, pair_mask)`: same as `forward_grad.cross_distogram_from_z`
  but keep the batch axis — `...distogram_head(z + z.transpose(-2,-3))`, softmax, then
  `[:, AB, C, :].reshape(B, -1, nbins)`. Write a new function; don't edit the imported one.

**3. Diversity lever — per-particle CONDITIONING offset, NOT the pair state.** Noise is never added
to `z`: raw noise on the pair state violates the trunk's triangle consistency (off-manifold) AND is
re-absorbed within a segment. Instead perturb the low-dim conditioning, which enters through learned
maps (on-manifold) and is re-injected every loop (persistent). Two interchangeable forms
(`--lever {slm,affine}`, default `slm`):
- **slm (recommended):** baseline `s_lm [1,L,256]` + a fixed per-particle random per-residue offset
  `delta_i`; lift `lm_z_i = lift_to_lm_z(model, s_lm + delta_i)` and re-inject every loop. Low-dim,
  on-manifold via the learned SingleToPair lift, per-residue — the **synthetic analog of the per-seed
  mask** that drives real diversity. Hold the tiny `s_lm+delta [B,L,256]`, **recompute `lm_z_i` in
  each loop** (cheap; avoids a 190 MB/particle resident `lm_z`).
- **affine:** a random per-particle channel-wise affine on `lm_z` (reuse `steer_confornet.ConforNet`
  with random, un-optimized `W,b`), broadcast over all `L×L` pairs — even lower-dim, geometry-preserving.
Sample offsets in fp32 (bf16 `randn` has coarse mantissa). **Anneal** the offset scale down across
rounds (hot early to cross watersheds, cool late to settle); an optional small *per-loop* jitter is
the "synthetic dropout". Initial spread also uses distinct `z0` draws (the transient seed).
Start scale ~0.05–0.2 of the baseline `s_lm`/`lm_z` norm; probe via `--rel/--anneal`.

**4. Fan-out / cluster-prune per round (the beam).** Per round: take the `K` survivors, fan each to
`F` children with fresh noise (step 3) → wide batch of up to `K*F` = `W`, sized to fill VRAM →
`trunk_loops` one segment → `cross_distogram_batched` → **cluster the `W` children and keep one
representative per cluster, pruning back to `K`**. Cluster **unsupervised** in cross-distogram W1
space (W1 threshold / greedy density-aware assignment) so the dense majority cluster always yields a
representative (NOT pure farthest-point, which would drop the majority). No centroids/labels enter
this step. **τ is relative**, set from the round's pairwise-W1 distribution (e.g. a low quantile) —
a hardcoded 7txw τ would not generalize. **When clusters > K, do NOT trim by dropping the smallest:**
a freshly-escaped rare basin (e.g. G) is a *small* cluster, so size-based trimming discards exactly
the prize — cap by keeping cluster *diversity* (dense seedbed + the most distinct of the rest) or let
`K` float up to the cluster count. Drop NaN/degenerate distograms here (validity gate). Log per-round
metrics. Segment schedule sums to
`N_LOOPS+1 = 17`; the last segment runs to the full horizon with noise annealed to 0 so survivors
are clean basins. Validity for v1 = bounded noise + the trunk's re-fold each loop + this cluster
prune + the final ipTM filter; no off-manifold leash beyond bounding `sigma`.

**5. End: decode -> select.** For each final representative (`z` still GPU-resident), decode with
`diffuse_from_z(model, builder, feat, chain_infos, cond, z_i_np, n_samples, n_steps, name, seed)`;
record best `.iptm` (and `.pair_chains_iptm`). Final selection across representatives = best-pair
ipTM — pair-agnostic. (7txw only, **diagnostic**: also log wg/wb1/wb2 vs centroids to see whether a
G representative was found and whether ipTM ranked it; this label never changes which representative
is selected.)

**6. Memory / auto-size the fan-out width `W`.** `W = K*F` is the wide batch processed per round and
is what fills VRAM (survivors `K` carried between rounds is tiny). After `load()` (ESMC offloaded to
CPU): `free = cuda.mem_get_info()[0]`. Probe peak transient with `reset_peak_memory_stats()` + one
`trunk_loops(n_loops=1)` at batch 1 and 2; marginal per-particle cost `~M2-M1` (state ~190 MB +
batch transient). `W_max = floor(0.85*free / (M2-M1))`, clamp to `--max-W`. Choose `F = W_max // K`
so each round fans as wide as VRAM allows. `--W 0` = auto.
**Estimate (L=609):** ~190 MB/particle persistent (`z`) + ~0.6-1 GB transient (a few coexisting
pair-sized tensors; `chunk_size=32` keeps triangle internals small) → with ~75 GB usable after
weights, expect `W ~ 80-130`; with `K~6`, `F~16`. ~5 rounds ≈ ~500 trajectories/run; two GPUs ≈
~1000 (vs thousands of naive seeds). To grow `W`, lower `chunk_size` (16/8) — smaller triangle
transient, some throughput cost.

## Saving (1 TB quota — NEVER persist the 190 MB per-particle z)

- per-round metrics CSV (append+flush): `round, survivor, parent, rms_z, nearest_w1,
  wg, wb1, wb2, basin` (`CSV_DIR/beam_<tag>.csv`).
- final representative cross-distograms as small `.npy` (`npairs×nbins`).
- decode representatives only: cif + confidence JSON via `save_samples`; a survivor CSV with
  `iptm/ptm/pair_chains_iptm` + basin (`CSV_DIR/beam_survivors_<tag>.csv`).
- Decode while z is GPU-resident; do not write z to disk.

## Run (esm venv, GPU, in the user's per-GPU tmux sessions — do not launch without go-ahead)

```
CUDA_VISIBLE_DEVICES=N $ESM_PYTHON confornet/beam_search.py \
    --name 7txw --tag gN --seed-base N000000 --K 6 --W 0 --rounds 5 --decode
# second free GPU = second independent beam for double coverage:
CUDA_VISIBLE_DEVICES=M $ESM_PYTHON confornet/beam_search.py \
    --name 7txw --tag gM --seed-base M000000 --K 6 --W 0 --rounds 5 --decode
```
(`--K` survivors kept per round; `--W 0` auto-sizes the fan-out width to VRAM; `--rounds` segments
the 17 loops; `--max-W`, `--rel`/`--anneal` tune noise.)

## Validation / success criteria (7txw)

1. **Coverage**: how many distinct basins (G/B1/B2 by nearest centroid) appear among the final
   representatives, and did **any** reach G (`wg < 0.08`)?
2. **Selection sanity**: does best-pair ipTM over representatives pick a G survivor (guard the B2
   confident-decoy trap by checking wg, not ipTM alone)?
3. **Decode confirm**: decode the G representative -> DockQ vs `confornet/data/ground_truths/
   7txw-assembly1.cif` (existing `eval/eval_ost_foldbench.py`); expect A|C DockQ ~0.85-0.92 for a
   true-G hit.
4. **Control ablation (must run)**: naive sampling at equal forward-pass budget — `--K 1 --rounds 1`
   with the same total particle count and noise off (independent z0 draws, no fan-out/prune). Show
   prune-and-fan raises basin coverage / G-hit-rate over naive sampling at equal compute — otherwise
   the beam machinery adds nothing.

## Open knobs / future (not v1)

Clustering stays in raw cross-distogram **W1** (pair-agnostic) — do NOT swap in a 7txw pairformer-z
PCA space as the clustering metric; PCA is allowed only as an offline **diagnostic** view of the
runs, never inside the algorithm. Other future knobs: weight-noise (thickets) as an alternative
noise lever; a fixed-population (`--K = W`, no prune) mode for comparison; generalizing data-prep
to arbitrary pairs once 7txw validates (only the input JSON is required; centroids are diagnostic
and optional).
