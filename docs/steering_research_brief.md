# Research brief: steering 7txw antibody–antigen docking into the good basin

> ## ⚠️ SUPERSEDED IN PART — 2026-07-14
>
> Everything below was measured on **ESMFold2-Fast run SINGLE-SEQUENCE**
> (`msa_encoder.enabled = false`; the driver also filters MSA-bearing JSONs out).
> **Running without an MSA is what CAUSES the confident-decoy inversion** — confidence
> discriminates across the pool and then inverts inside its own top decile, exactly where
> `argmax` reads. Both checkpoints invert without an MSA; both are healthy with one.
>
> The trap class is **real but ~40% smaller** than the numbers here say (trap rate 53% → 33%
> once the model has an MSA). **Every headline number below is inflated.**
>
> → **[`docs/overnight_2026_07_14_summary.md`](overnight_2026_07_14_summary.md)** — read first.


Self-contained brief for an agent with no prior context. Goal: get fresh, better ideas for
a **generalizable, label-free, compute-efficient** method to reliably produce the correct
antibody–antigen complex.

## 1. The problem
- Target: **`7txw`** (a Fab antibody, chains A=heavy + B=light; chain C=antigen). Model:
  **ESMFold2-Fast** (single-sequence; an ESMC language model → a recycled "pairformer/parcae"
  pair-folding trunk (17 recycling loops) → a diffusion structure head).
- It is an **orientation problem, not a folding problem.** Both monomers fold correctly in
  essentially every run (~1 Å); the only thing that varies is the **antigen's docking pose**.
- Outcome falls into discrete **basins**: **G** = native pose (DockQ ~0.9), **B1** = the dominant
  default decoy, **B2** = a *confident* wrong-register decoy (docks the right CDR patch in the wrong
  rotation; DockQ ~0.03 but its confidence/ipTM ≈ G's, so confidence can't distinguish it).
- Geometry (in trunk cross-block distogram W1 distance from the unperturbed baseline): **B1 ≈ 0,
  B2 ≈ 0.66, G ≈ 1.04** — i.e. basins are ordered by distance-from-baseline, B1→B2→G.
- **Goal:** reliably and cheaply generate the **G** complex, ideally so every attempt finds it.

## 2. Baseline method (what production currently does): native mask+dropout TTS
- ESMFold2-Fast is run with **per-seed LM input masking** (`lm_mask_pct=0.1`: ~10% of ESMC input
  tokens replaced by the MASK token, a deterministic per-seed draw) and **per-loop LM dropout**
  (`lm_dropout=0.3`). Each **seed** = one mask + dropout schedule → 5 diffusion samples.
- This stochasticity is what produces basin diversity: across seeds you get a mix of B1/B2/G.
  **Native good rate ≈ 4% per seed** (measured over n=2060 seeds).
- Selection at scale works: **best `pair_iptm` ranks the good seeds at the top (AUC 0.945)** despite
  B2 being a confident decoy — so *selection is ~solved; coverage/generation is the bottleneck.*
- The **mask is the causal basin lever**: which residues get masked predicts the good basin
  (L1-logistic mask→good, 5-fold CV **AUC 0.62**, p=0.01); good seeds over-mask antigen residues
  **C48–C50** (near the native epitope). The *passive* ESMC/`s_lm` representation does **not**
  separate basins (a "default-well" displacement null).
- Mechanistically: every seed starts in **B1** (the universal default); the **B1→G escape is a
  late, discrete recycling event (loops ~3–12)**, seed-specific. Bad seeds branch B1→B2 or never
  leave B1. The unperturbed (no-mask/no-dropout) baseline stays in B1 all 17 loops and sinks deeper.
- The basin signal lives in the **pairformer-z trunk** representation (localized to antigen
  cross-blocks), not the ESMC LM rep.

## 3. The steering method that produced the figure (`centroid_dist_slm_nudge.png`)
Idea: instead of waiting for a lucky seed, **gradient-steer** the LM single representation `s_lm`
(the ESMC-derived rep that is lifted to `lm_z` and injected into the trunk every loop) to *drive*
the baseline run into G, using gradients of a **cheap short-horizon distogram** (no diffusion decode).

**Label-free repulsion, reset-each-run (the figure's config):**
- Work entirely from **one cached, no-mask baseline `s_lm`** (`s0`). No ESMC re-run during steering.
- Keep a growing list of **anchors** = distograms already visited, seeded with the baseline's
  K=3 distogram (the "B1 floor").
- **Each run** (10 per trial, anchors accumulate within a trial; **reset between trials**):
  1. Reset the leaf to `s0 + d`, where `d` is a **random direction of fixed length 5% of ‖s0‖**
     (the "nudge"; breaks a zero-gradient cusp + supplies directional diversity).
  2. **M=5 Adam steps (lr=0.3)** that **maximize the mean W1 distance** of the **K=3** cross-block
     antibody×antigen distogram from all accumulated anchors (`loss = −mean_a W1(P_K(s_lm), a)`).
     "K=3" = read the distogram after only 3 of the 17 recycling loops (cheap, and the trajectory is
     still responsive to `s_lm`; full-horizon gradients are a flat B1 attractor and don't steer).
  3. Append the converged K=3 distogram as a new anchor.
- **Classification/readout (free, no decode):** run the steered `s_lm` the full 17 loops, take the
  cross-block distogram, and compute its W1 distance to the **G/B1/B2 centroids** (centroids = mean
  loop-16 distogram over the known good / B1 / B2 seeds). A run is **"real G"** if its distance to
  the G centroid `wg < 0.08` (within-G band of real good seeds is 0.033–0.079).
- Hyperparameters were inherited from a proven earlier single-chain run that decoded to DockQ 0.85
  (M=5, lr=0.3, K=3, nudge=5%); 15 steps over-optimizes and goes off-manifold (unfoldable).
- Code: `analysis/grad_steer/steer_repel.py`.

## 4. What the figure shows
Per-trial small multiples (10 trials × 10 runs). x = run index (run 0 = the unperturbed baseline
seed). Three lines per trial = W1 distance from that run's 17-loop distogram to **B1 (gray),
B2 (orange), G (green)** centroids; shaded band = within-G band; **★ = real-G run (wg<0.08)**.
You can see runs bounce between basins (one line near ~0, the others ~0.7–1.1), with green dipping
into the band only occasionally. Title: "12 real-G runs over 10 trials."

## 5. Results
- **Per-trial: 8/10 trials reach ≥1 real-G run. Per-run: 12% real-G (21% looser G-label).**
- Among the 8 successful trials, first real-G appears at run ~3.9 on average; **2/10 trials whiff
  entirely** in 10 runs (their 10 random nudge directions all funnel into B1/B2 — the diversity is a
  *ball around one baseline point*, so it's correlated, and when it drains to B2 all 10 go).
- **DockQ-validated (decoded the 12 starred runs, 17-loop z → diffusion → OST DockQ):**
  A|C mean ≈ **0.90** (0.80–0.92), B|C ≈ **0.80** (0.77–0.83). **Zero false positives** — `wg<0.08`
  is a faithful proxy for a real ~0.9-DockQ complex; no confident-decoy B2 slips through.

## 6. The K=3 commitment / compute question (live — refreshed every ~5 min by a loop)
Re-ran the nudge config saving the **K=3** trunk `z` as well as the 17-loop `z`, then decode each
steered-G run at **both** horizons. Question: does a steered run commit to G by the K=3 steering
horizon (→ classify/decode at K=3, skip 14 loops, ~half the per-run trunk cost) or only by loop 17?

**Result so far is MIXED / run-dependent (n=2):**

| star | K=3 A\|C | K=17 A\|C | K=3 B\|C | K=17 B\|C | commits by K=3? |
|---|---|---|---|---|---|
| t00 r07 (wg 0.068) | 0.912 | 0.914 | 0.791 | 0.820 | **yes** (~identical) |
| t01 r04 (wg 0.049) | 0.424 | 0.769 | 0.479 | 0.736 | **no** (K=3 much worse) |

Some steered-G runs are already good at 3 loops; **others are not** — t01 r04 is a genuine 17-loop
real-G run whose K=3 structure is only ~0.42 and matures to ~0.77 only by loop 17.

**Distogram-level diagnostic (cheap, no decode — `wg_k3_check.py`):** it's the *distogram*, not just
the decode, that's late. wg (W1 to G centroid) at K=3 vs K=17: t00 r07 = **0.079 → 0.068** (already
at the G band by loop 3), but **t01 r04 = 0.895 → 0.048** — its K=3 distogram is sitting in a *decoy*
and only flows to G by loop 17. So the steering finds an `s_lm` that *ends* at G over 17 loops, but
whose 3-loop intermediate is still mid-escape (the late B1→G dynamics happen after loop 3 for this
run). **Consequence: K=3 is NOT a safe shortcut — even cheap K=3 *classification* would mis-discard
the late committers.** At best you save the extra loops on the *early*-committing subset, not all
runs.

**Status (final, n=2):** the K=3 run was cut short by a disk-quota crash at trial 2 (3 trials,
2 stars), so the K=3-vs-K=17 comparison stands at n=2 — *mixed*: t00 r07 commits by loop 3, t01 r04
is a genuine late committer (decoy distogram at K=3, wg@K3 0.895). Takeaway holds: **K=3 is not a
reliable universal shortcut**; even cheap K=3 classification mis-discards the late committers.

## 7. Variants tried (and outcomes)
- **Reset + s_lm nudge** (the figure): works, 8/10 trials, but the random ball-around-baseline
  diversity whiffs ~20% of trials.
- **z0-variation** (vary the trunk's initial recurrent state `z0` per run instead of the nudge):
  *worse*, 4–5/10 trials. `z0` is a *transient* lever (decays through the recurrence, gone by the
  late commit window), and it contaminates the repulsion signal.
- **z0 as the optimized lever** (steer `z0`, fix `s_lm`): never reaches G (transient).
- **Carry-forward** (continuous walk, do NOT reset `s_lm` between runs): drifts off-manifold into
  unfoldable distograms (DockQ collapses); a leash is needed.
- **Carry + "window-1"** (repel from only the most-recent anchor): **freezes** — after a run the
  leaf sits *exactly* on its own just-appended anchor → W1=0 → zero gradient (the abs'(0) cusp).
- **Carry + re-nudge / + keep-baseline-anchor + leash** (running now): testing whether a *connected*
  B1→B2→G walk works if you (a) kick off the cusp each run or (b) always keep the baseline anchor,
  with a leash to stay on-manifold.
- **Leash** = after each step, project `s_lm` back into a ball of radius `0.15·‖s0‖` around baseline
  (keeps it foldable; real-G hits live at displacement ≈0.13). Caveat: ball is centered on the
  baseline *point*, not the real-seed `s_lm` cloud.
- **Considered, not yet run — mask-init steering:** replace the synthetic nudge with a *different
  ESMC-masked `s_lm` per run* (10% mask, re-run ESMC each run) as the per-run init, then repel.
  Rationale: the mask is the *causal* basin lever and a masked `s_lm` is on-manifold/native, so its
  inits are spread across real basins (more independent → should fix the trial-4 whiffs). Cost: it
  re-adds an ESMC forward per run (efficiency vs reliability tradeoff).

## 8. Mechanistic facts that constrain ideas
- Basins are **discrete attractors** — distances to centroids are bimodal; a run is decisively *in*
  one. The trajectory through basin-space is discrete jumps, not a continuous slide.
- "Away from the explored anchors" ≠ "toward G." Repelling from the B1/B2 anchor cloud points to the
  cloud periphery, which the 17-loop dynamics often re-capture into B1/B2 (B2 is a strong attractor).
- The **mean-over-anchors repulsion self-weakens** as anchors accumulate (you can't be far from a
  spreading cloud), so late runs barely move.
- **Reset discards partial progress** — a run can get partway toward G (G-distance drops to ~0.85)
  then resets; the reset design can't accumulate the journey.
- The **baseline ≠ the B1 centroid** — it's the deep-well default point, a specific anchor, not the
  basin as a region.
- With the **W1 (L1) loss + Adam**, gradient magnitude is *not* small near the baseline (L1 gradient
  is ~constant; Adam normalizes magnitude). The only true zero is the exact cusp. So escaping the
  baseline is easy; the bottleneck is *direction* (aim) and the attractor, not gradient size.

## 9. Constraints / desiderata for a good method
- **Generalizable across antibody–antigen pairs** — NO complex-specific priors: no force-masking
  C48–C50, no explicit B2-centroid repulsion, no G-radius target. Must be **label-free** (discover
  basins as it goes; you only know B2/G *after* finding them).
- **Compute-efficient** — the **dominant cost is the pair-folding trunk** (O(L²) pair ops recycled
  17×), NOT ESMC. Gradient steering pays a backward-pass tax (5 backward passes through the trunk per
  run). The right metric is **compute-to-first-good**, and the baseline to beat is native TTS
  sampling + ipTM selection. (Whether directed steering actually beats sampling on compute is
  unproven; the K=3 finding helps steering's case.)

## 10. Open question — where we want better ideas
The current method optimizes a **continuous** `s_lm`/distogram objective and *hopes* the discrete
17-loop dynamics drop it into a *new* basin. There is no explicit notion of "basin" in it and no
guarantee a run lands somewhere new — so it's a dressed-up lottery (12%/run, 2/10 whiffs).

We want a **principled basin-discovery method**: ideally each run discovers a *new* basin in order
**B1 → B2 → G**. Two honest framings:
1. **Mode-coverage / novelty search:** maintain the set of discovered basins; accept a run only if
   its outcome is bimodally distinct from all discovered ones. Guarantees the discovery sequence —
   but doesn't make the rare G cheaper to reach (no map from input-perturbation → basin; the mask
   prior is only AUC 0.62).
2. **Work with the mechanism that creates the basins:** the B1→G escape is decided in the recycling
   loops (~3–12). Intervening *there* (mid-trunk, in the branch window) engages the basin-generating
   dynamics directly instead of nudging upstream `s_lm` and praying — the most principled
   *unexplored* handle, and where a *directed* B1→B2→G could live.

**What we'd love ideas on:** a label-free, generalizable, trunk-cheap procedure where each
perturbation reliably discovers a *new* docking basin (culminating in G) — whether via a smarter
diversity/novelty mechanism, a mid-recycling intervention, or a reframing we're missing.

---
Key code/artifacts: `analysis/grad_steer/steer_repel.py` (steering), `forward_grad.py` (cheap
trunk/distogram forward + grad), `decode_starred.py` + `eval/eval_ost_foldbench.py` (decode→DockQ),
`analysis/figs/centroid_dist_slm_nudge.png` (the figure). Result CSVs in `results/csv_files/`.

---

## 11. Current direction & handoff (adopting ConforNets)

We are pivoting the steering method to **ConforNets** (arXiv:2604.18559; code
github.com/aqlaboratory/confornets) — a more principled, generalizable version of §8 that fixes our
three weaknesses (generalizability, on-manifold, efficiency).

**The method:** instead of steering the raw full `s_lm`, optimize a tiny **channel-wise affine** on
the pre-Pairformer pair latent — `φ(z) = z·Wᵀ + b`, `W` (256×256, identity-init), `b` (256,
zero-init), shared across all L×L positions. k particles are pushed apart by an unsupervised
**pairwise distogram-CDF repulsion** (their `dist_cdf_mse` = MSE between distogram CDFs ≈ our `w1`;
alt `coord_mse` on decoded coords). Adam lr 1e-3, ~21 steps, halve lr every 5, grad-clip 10, init
identity + small noise to break the symmetry cusp. Channel-wise ⇒ protein-agnostic and on-manifold
by construction. **Caveat: ConforNets is validated ONLY on monomer conformations — never multimer
docking. So our open question is: can a global affine drive a discrete antigen docking-pose change
(B1→B2→G)?**

**Built this session (ready, NOT yet GPU-tested — cluster was fully contended):**
- `analysis/grad_steer/steer_confornet.py` — k=2 ConforNet affine on `lm_z` (c_z=256), cross-block
  K=3 distogram repulsion (`w1`), eval each particle at 17 loops → basin; loops CONTINUOUSLY.
  **Minimal saving:** small CSV (`results/csv_files/steer_confornet_<tag>.csv`: run, particle, basin,
  wg/wb1/wb2, repel) + the ~256 KB affine `(W,b)` ONLY for a real-G hit (wg<0.08) under
  `7txw_confornet/<tag>/` — NEVER the 144 MB z. Verified to parse; affine identity-init is a no-op.
- `analysis/grad_steer/confornet_one.sh` — waits for ONE fully-idle GPU and runs it (single GPU,
  fair use). Running in tmux session `confornet`, idling until a GPU frees. (A 4-GPU auto-grab
  version was correctly blocked as monopolizing shared GPUs — do NOT rebuild that.)

**Next steps:**
1. When a GPU frees and it first runs, check `/tmp/confornet_g<idx>.log`: `run 0: basins=…` = working;
   a traceback = fix the untested affine→trunk gradient/dtype path.
2. Read the headline: do the k=2 particles **separate into different docking basins**
   (`separated=True`) and does either reach **real G** (wg<0.08, `realG>0`)?
3. If a global affine can't separate the poses → docking-transfer risk is real; try `coord_mse`
   (coordinate repulsion) or a block-wise (non-global) affine. The supervised `aligned_mse_loss`
   transfer mode is unused (later idea if we have a reference good complex).

**Compute reality:** the **pair-folding trunk dominates** compute (NOT ESMC), so gradient steering
pays a backward-pass tax, and it's **unproven** that directed steering beats native mask+dropout TTS
(~4%/seed) + ipTM selection (AUC 0.945) on compute-to-first-good.

**Disk gotcha (caused a mid-session crash):** there is a **1 TB per-user quota** on `/home`
(`ebrahim`); the old steering runs saved a **144 MB pairformer z PER RUN** and filled it → all 3 runs
crashed (`Errno 122`). We freed 182 G by deleting `7txw_nodrop_bad` (DockQ already in a CSV). Keep
new runs saving minimally (steer_confornet does). Check `quota -s` before big runs.
