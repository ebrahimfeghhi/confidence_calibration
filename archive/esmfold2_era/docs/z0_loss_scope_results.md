# Joint-z0, loss-scope, and z0-nudge ablation — results

_Last updated: 2026-07-02 ~23:00. Follow-on to [`repel_campaign_6targets_handoff.md`](repel_campaign_6targets_handoff.md)
covering the 6-target label-free repulsion campaign. Base recipe unless noted: `M=5, K=3, lr=0.3,
init_frac=0.05`, `cycle-decode=16`, stop at DockQ ≥ 0.23, 12h budget per campaign. "z0" = jointly
optimizing the initial pair state `z0` alongside `s_lm` (`--joint-z0 --lr-z0 0.01`)._

## Interfaces, per target

Every target here is an antibody–antigen complex; four are conventional Fabs (2 physical antibody
chains, heavy + light) and two are single-domain VHH nanobodies (1 antibody chain). A Fab therefore
has **two** real ab–ag chain-pairs (heavy–antigen, light–antigen); a VHH has one.

| Target | Antigen chain | Antibody chain(s) | Ab type | Ab–ag interfaces |
|---|:-:|:-:|:-:|---|
| 8jyr | A | B, C | Fab | A–B, A–C |
| 8urf | C | A, B | Fab | A–C, B–C |
| 8vzo | A | B, C | Fab | A–B, A–C |
| 8vvl | A | B, C | Fab | A–B, A–C |
| 8im0 | A | B (copy 2) | VHH | A–B‑2 |
| 8z8v | A | B | VHH | A–B |

**Caveat on every steering-campaign number below.** `repel_proven.py`'s own scoring (`_abag()`)
reports the **max over whichever ab–ag interfaces exist** for that target — for the four Fab
targets this collapses two real chain-pair DockQ values into one number and discards which
interface won. Per-interface breakdown + `pair_iptm` are **not available** for any steering
campaign below (see [Data gaps](#data-gaps--open-questions)). The **baseline (unsteered) sweep**
used the full multi-interface OST path and *does* have true per-interface numbers.

## Stage 1 — does joint-z0 help? (mean loss, interface-only scope, nudged z0)

All six targets now have a completed s_lm-only vs. +joint-z0 (nudged) comparison:

| Target | Config | Stop | Runs / cycles | Best DockQ | Best ipTM | Elapsed | GPU-h | Peak VRAM |
|---|---|---|---|---:|---:|---:|---:|---:|
| 8urf | s_lm-only | HIT | 80 / 5 | 0.378 | 0.869 | 16207s | 4.50 | 25.6 GB |
| 8urf | + joint-z0 | HIT | 80 / 5 | **0.592** | 0.868 | 16243s | 4.51 | 27.8 GB |
| 8jyr | s_lm-only | HIT | 32 / 2 | 0.952 | 0.895 | 7802s | 2.17 | 29.1 GB |
| 8jyr | + joint-z0 | HIT | 32 / 2 | 0.951 | 0.889 | 7809s | 2.17 | 31.7 GB |
| 8vzo | s_lm-only | HIT | 48 / 3 | 0.544 | 0.714 | 38963s | 10.82 | 82.1 GB |
| 8vzo | + joint-z0 | **TIMEOUT** | 54 / 3 | 0.181 | 0.751 | 43517s | 12.09 | ~86 GB |
| 8vvl | s_lm-only | TIMEOUT | 137 / 8 | 0.167 | 0.703 | 43313s | 12.03 | 37.8 GB |
| 8vvl | + joint-z0 | TIMEOUT | 137 / 8 | 0.178 | 0.677 | 43412s | 12.06 | 40.8 GB |
| 8im0 | s_lm-only | EXHAUSTED | 512 / 32 | 0.047 | 0.897 | 28582s | 7.94 | 10.5 GB |
| 8im0 | + joint-z0 | EXHAUSTED | 512 / 32 | 0.033 | ~0.90 | 28936s | 8.04 | ~10 GB |
| 8z8v | s_lm-only | TIMEOUT | 139 / 8 | 0.070 | 0.877 | 43311s | 12.03 | 37.1 GB |
| 8z8v | + joint-z0 | *running* | 64+ / 4 so far | 0.041 (declining: .062→.050→.044→.041) | 0.848–0.874 | in progress | — | — |

**Reading it:** one clean win (8urf, +0.21 DockQ, not a ceiling effect — baseline-alone only
reaches 0.404 there). 8jyr is a wash. 8vzo's *nudged* z0 run looks worse than s_lm-only (0.181 vs
0.544) — **but this triggered the whole nudge investigation below, and it turned out to be an
artifact of the nudge, not of z0 itself** (see Stage 4 — removing the nudge on the exact same
target+config hit 0.655, beating the s_lm-only baseline). 8vvl-interface-only+z0 finished flat
(0.178, barely above the 0.167 no-z0 result) — z0 alone doesn't rescue 8vvl; it needs the
intra-antigen term too (Stage 2). 8im0's z0 run finished at 0.033, same dead-end as s_lm-only —
8im0 has no known reachable basin under any method tried so far. 8z8v+z0 (nudged) is still running
and trending down, currently below its own s_lm-only comparison point.

## Stage 2 — loss scope: interface-only vs. +intra-antigen vs. full-distogram (on joint-z0)

8vvl was diagnosed as an antigen-*misfolding* case (antigen-only lDDT 0.208 vs 0.79–0.92 for the
other 5 targets), so it's the target used to test whether broadening the loss beyond the interface
matters:

| Target | Loss scope | z0 | Stop | Runs / cycles | Best DockQ | Best ipTM | Elapsed | GPU-h |
|---|---|:-:|---|---|---:|---:|---:|---:|
| 8vvl | interface-only | no | TIMEOUT | 137 / 8 | 0.167 | 0.703 | 43313s | 12.03 |
| 8vvl | interface-only | yes | TIMEOUT | 137 / 8 | 0.178 | 0.677 | 43412s | 12.06 |
| 8vvl | + intra-antigen | no | TIMEOUT | 137 / 8 | 0.157 | 0.742 | 43379s | 12.05 |
| 8vvl | + intra-antigen | **yes** | **HIT** | 16 / 1 | **0.305** | 0.580 | 5099s (1.4h) | 1.42 |

**Neither lever alone got 8vvl past 0.18; both together hit 0.305 in a single cycle** — the
cheapest hit of any mean-loss campaign here. z0 alone is not a substitute for the intra-antigen
term on this target; the term is doing real work, and z0 adds something on top of it specifically
(not generally — see the flat interface-only+z0 row above).

**Full-distogram** (untargeted: loss/anchors over the entire L×L distogram instead of targeted
blocks), run sequentially 8jyr → 8urf → (8vvl planned) with joint-z0 on GPU3:

| Target | Stop | Runs / cycles | Best DockQ | Best ipTM | Elapsed |
|---|---|---|---:|---:|---:|
| 8jyr (full-disto) | HIT | 48 / 3 | 0.337 | 0.765 | 11743s (3.3h) |
| 8urf (full-disto) | *interrupted* | 112 / 6 | 0.203 (peak @ cyc 1, declined to 0.064–0.101 since) | 0.822–0.898 | ~10620s before interrupt |
| 8vvl (full-disto) | *never ran* | — | — | — | killed ~2 min after auto-start |

Compare to targeted interface-only+z0: 8jyr hit 0.951 (vs. full-disto's weaker 0.337, and in fewer
cycles); 8urf hit 0.592 (vs. full-disto's declining 0.203-peak, which was also below the unsteered
baseline of 0.404 — see below). **Full-disto consistently underperforms the targeted loss on both
targets tested**, and shows a distinct failure signature: it peaks early (cycle 1) then declines,
unlike targeted-loss runs which either hit within a few cycles or stay roughly flat. Likely
mechanism: the full L×L distogram is dominated by intra-chain (non-interface) entries, so the
repulsion gradient is mostly driven by things unrelated to the docked interface, causing runs to
wander away from a good region once found by chance rather than reinforcing it. **8urf's run and
the queued 8vvl run were cut short** (GPU3 was repurposed for the no-nudge investigation, then
handed off entirely — GPU3 is now running an unrelated FoldBench sweep for a separate purpose) —
this ablation is not fully concluded, but the pattern on the 2 targets tested is consistent and
large enough (3–9x gap) to be load-bearing for the "use targeted loss" recommendation regardless.

## Stage 3 — loss function: mean vs. kernel repulsion

Implemented `--loss kernel`: summed Gaussian-kernel repulsion (`bias = Σ_a exp(-(d_a/h)²)`,
minimized directly), bandwidth `h` self-calibrated per-run from the anchors' own nearest-neighbor
W1 spread. See [`summed_kernel_loss.md`](summed_kernel_loss.md) for the derivation.

| Target | Loss | z0 | Extra | Stop | Runs / cyc | Best DockQ | Elapsed |
|---|---|:-:|---|---|---|---:|---:|
| 8jyr | kernel | yes | — | *killed* (weak) | 48 / 3 | 0.144 | ~4h (interrupted) |
| 8vvl | kernel | yes | intra-c | *killed* (weak) | 32 / 2 | 0.150 | ~2h57m (interrupted) |

**Kernel loss underperformed mean loss on both targets tested** (compare: 8jyr mean+z0 hit 0.951 at
cycle 2; 8vvl mean+intrac+z0 hit 0.305 at cycle 1) — both kernel runs were killed mid-campaign
(not timed out) once the z0-nudge investigation below took priority, so treat "killed (weak)" as
"trending clearly behind, not concluded." **This turned out to be entangled with the nudge
question** — see Stage 4: removing the nudge fixed 8jyr's kernel-loss result completely (0.144 →
0.952) but did *not* fix 8vvl+intrac's (0.150 → 0.126–0.168, still no clear improvement), so kernel
loss's weakness has at least two distinct causes depending on target/config, only one of which the
nudge explains.

## Stage 4 — the z0-nudge investigation

**Motivation:** every run (joint-z0 or not) gives its optimized variable a small random "nudge"
before the M=5 gradient steps (5% of the variable's own norm, in a random direction) so runs don't
all start from the literal same point. For `s_lm` this makes sense — it's the only source of
per-run diversity in the non-joint-z0 case. But once z0 is *also* jointly optimized, its gradient at
step 0 already depends on the current `lm_z` (which differs every run because of s_lm's own nudge)
— so z0 arguably gets a distinct, non-degenerate gradient direction for free, and the extra random
kick may just be adding uninformative noise directly to the trunk's actual starting state.

**Mechanism check (why might this matter, or not):**
1. z0 only enters the recycling trunk once, before loop 1. Every loop after does
   `z = a*z + b*inject(lm_z)`, and `a` is a **learned, per-channel decay coefficient strictly < 1**
   (confirmed in the model source: `a = exp(-softplus(delta)*exp(log_a))`,
   `modeling_esmfold2.py:742-746`). So z0's own identity decays geometrically across the 17
   recycling loops, while `lm_z` (from s_lm) is freshly re-injected every loop and never decays.
   The K=3 loss actually optimized sees z0 strongly (only 3 loops of decay); DockQ depends on K=17,
   where z0's effect has decayed much further — a possible train/eval horizon mismatch.
2. **Ruled out:** a "shared Adam budget" theory (does z0 steal s_lm's optimization step?). Checked
   per-step `drift` (relative displacement from start) for s_lm with vs. without joint-z0 on 8urf,
   n=80 runs each: mean final-step drift 0.1186 (no z0) vs. 0.1201 (joint-z0) — statistically
   identical. Adam keeps independent per-parameter-group moment estimates; z0 isn't diluting s_lm's
   own movement.
3. **Checked and found no relationship:** does the *size* of z0's own movement (`drift_z0`) predict
   run quality? Correlation(drift_z0, best decoded DockQ) = **-0.087** across the same 80 runs —
   essentially zero. The actual hit run (8urf+z0, DockQ 0.592) had *below-median* z0 drift. So "z0
   moved a lot" isn't what makes a run good or bad — the *direction* the nudge+gradient pushed z0
   in matters, not the magnitude.

**Implemented `--no-z0-nudge`** (`repel_proven.py`): `z0_leaf` can now start directly from the fixed
base `z0` (no random kick), shaped purely by the coupled gradient. Results so far, same target+config
as the nudged version above, only the nudge changed:

| Target | Loss | Extra | Nudged best DockQ | No-nudge best DockQ | Read |
|---|---|---|---:|---:|---|
| 8jyr | kernel | — | 0.144 (killed, 3 cyc) | **0.952 (HIT, cyc 1, 16 runs, 65 min)** | Dramatic fix |
| 8vzo | mean | — | 0.181 (TIMEOUT, 3 cyc) | **0.655 (HIT, cyc 1, 16 runs, 3.6h)** — beats the 0.544 s_lm-only baseline | Dramatic fix, and better than baseline |
| 8vvl | kernel | intra-c | 0.150 (killed, 2 cyc) | 0.126–0.168, still climbing/bouncing (3 cyc so far) | No clear improvement yet |
| 8urf | kernel | — | *(no nudged-kernel result — this target wasn't reached before the switch)* | 0.075–0.090, flat (4 cyc so far) | Weak; doesn't repeat the 8jyr win |
| 8im0 | kernel | — | 0.033 (mean+nudged z0, EXHAUSTED, for reference) | 0.017–0.049, flat (5 cyc so far, run by Fable) | Expected null — 8im0 has no known reachable basin under any method |

**Reading it:** two clean, dramatic wins (8jyr, 8vzo) under **two different loss functions** —
suggesting the nudge problem isn't specific to one loss function. But it does **not** generalize
uniformly: 8vvl+intrac and 8urf show no comparable fix so far. Current best guess: the nudge was
specifically harmful in cases where the trajectory was already close to a good escape and a blind
random kick to the trunk's actual starting state was knocking it off course before the (only 5)
correction steps could recover; it's not a universal problem with joint-z0 itself. **Not enough
data yet to recommend "always use no-nudge" as a blanket default** — 8vzo is the only case with a
true no-nudge vs. established-baseline (not just vs.-nudged) comparison, and it's n=1.

### Does mean-loss's own reported metric even show when a trial is stuck?

Checked as a sanity check for a separate proposed feature (auto-ending a trial once it stops finding
new basins, using min-W1-to-nearest-anchor as the signal — not yet implemented). The exact proposed
signal was never logged for any existing run, but the closest available proxy — `repel`, i.e.
mean-loss's own mean-distance-to-*all*-anchors — **was** checked against the 8vvl z0-only trajectory,
which we know from its DockQ was stuck (137 runs, 8 cycles, DockQ never escaped ~0.09–0.18):

- `repel` ranged 0.41–0.76 across the whole run, with a **new peak (0.76) at run 75 of 137** — more
  than halfway through a confirmed-stuck trajectory. It never shows a plateau.

This is direct evidence that mean-to-all-anchors is a poor stuck-detector (diluted by distance to
everything else, even when repeatedly landing near one already-visited basin) — supports building
the proposed min-to-nearest-anchor signal instead, but that requires new instrumentation + a fresh
calibration run (planned: 7txw first, then the 6 real targets), not a retrofit onto existing data.

## Baseline (unsteered) control

8jyr and 8urf both have a completed baseline sweep (1024 seeds × 5 diffusion samples each, no
steering at all) — this tests whether scale alone reaches the same place steering does.

| Target | Interface | Best DockQ | Seed / sample | pair_iptm | global ipTM | ptm | Hit rate (≥0.23) |
|---|:-:|---:|---|---:|---:|---:|---:|
| 8jyr | A–B | **0.961** | seed 979 / s1 | 0.647 | 0.907 | 0.913 | 22/5120 (0.4%) |
| 8jyr | A–C | 0.936 | seed 388 / s0 | 0.597 | 0.891 | 0.902 | 32/5120 (0.6%) |
| 8jyr | *collapsed (max)* | 0.961 | seed 979 / s1 | 0.647 | 0.907 | — | 38/5120 (0.7%) |
| 8urf | A–C | 0.404 | seed 611 / s4 | 0.125 | 0.560 | 0.658 | 1/3310 (0.03%) |
| 8urf | B–C | 0.162 | seed 611 / s4 | 0.123 | 0.560 | 0.658 | 0/3310 |
| 8urf | *collapsed (max)* | 0.404 | seed 611 / s4 | 0.125 | 0.560 | — | 1/3310 (0.03%) |

**Reading it:** 8jyr's baseline-alone reaches 0.961 unsteered — better than every steered result for
this target (0.951–0.952 mean/z0, 0.337 full-disto) — steering isn't necessary here, the basin is
easy to find by chance with enough samples. 8urf's baseline-alone tops out at 0.404, well below the
best steered result (0.592, mean+z0) — this is the clean case where steering does real work. No
baseline sweep exists for 8vzo, 8vvl, 8im0, or 8z8v.

## Compute & memory summary (all completed / concluded campaigns)

| Tag | Target | z0 | no-nudge | intra-c | full-disto | kernel | Stop | Runs | Cyc | Best DockQ | Elapsed | GPU-h | Peak VRAM |
|---|---|:-:|:-:|:-:|:-:|:-:|---|---:|---:|---:|---:|---:|---:|
| 8jyr_k3m5 | 8jyr | | | | | | hit | 32 | 2 | 0.952 | 7802s | 2.17 | 29.1 GB |
| 8jyr_k3m5_jz0 | 8jyr | ✓ | | | | | hit | 32 | 2 | 0.951 | 7809s | 2.17 | 31.7 GB |
| 8jyr_k3m5_fulldisto | 8jyr | ✓ | | | ✓ | | hit | 48 | 3 | 0.337 | 11743s | 3.26 | 34.0 GB |
| 8jyr_k3m5_kernel | 8jyr | ✓ | | | | ✓ | killed | 48 | 3 | 0.144 | ~14400s | ~4.0 | ~31 GB |
| 8jyr_k3m5_kernel_nonudge | 8jyr | ✓ | ✓ | | | ✓ | **hit** | 16 | 1 | **0.952** | 3902s | 1.08 | 31.1 GB |
| 8urf_k3m5 | 8urf | | | | | | hit | 80 | 5 | 0.378 | 16207s | 4.50 | 25.6 GB |
| 8urf_k3m5_jz0 | 8urf | ✓ | | | | | hit | 80 | 5 | 0.592 | 16243s | 4.51 | 27.8 GB |
| 8urf_k3m5_fulldisto | 8urf | ✓ | | | ✓ | | interrupted | 112 | 6 | 0.203 | ~10620s | ~2.95 | n/a |
| 8urf_k3m5_kernel_nonudge | 8urf | ✓ | ✓ | | | ✓ | *running* | 64+ | 4 | 0.090 | in progress | — | — |
| 8vzo_k3m5 | 8vzo | | | | | | hit | 48 | 3 | 0.544 | 38963s | 10.82 | 82.1 GB |
| 8vzo_k3m5_jz0 | 8vzo | ✓ | | | | | timeout | 54 | 3 | 0.181 | 43517s | 12.09 | ~86 GB |
| 8vzo_k3m5_jz0_nonudge | 8vzo | ✓ | ✓ | | | | **hit** | 16 | 1 | **0.655** | 12991s | 3.61 | 86.2 GB |
| 8vvl_k3m5 | 8vvl | | | | | | timeout | 137 | 8 | 0.167 | 43313s | 12.03 | 37.8 GB |
| 8vvl_k3m5_jz0 | 8vvl | ✓ | | | | | timeout | 137 | 8 | 0.178 | 43412s | 12.06 | 40.8 GB |
| 8vvl_intrac_jz0 | 8vvl | ✓ | | ✓ | | | hit | 16 | 1 | 0.305 | 5099s | 1.42 | 39.2 GB |
| 8vvl_intrac_slmonly | 8vvl | | | ✓ | | | timeout | 137 | 8 | 0.157 | 43379s | 12.05 | n/a* |
| 8vvl_k3m5_kernel_intrac | 8vvl | ✓ | | ✓ | | ✓ | killed | 32 | 2 | 0.150 | ~10620s | ~2.95 | ~38 GB |
| 8vvl_k3m5_kernel_intrac_nonudge | 8vvl | ✓ | ✓ | ✓ | | ✓ | *running* | 48+ | 3 | 0.168 | in progress | — | — |
| 8im0_k3m5 | 8im0 | | | | | | exhausted | 512 | 32 | 0.047 | 28582s | 7.94 | 10.5 GB |
| 8im0_k3m5_jz0 | 8im0 | ✓ | | | | | exhausted | 512 | 32 | 0.033 | 28936s | 8.04 | ~10 GB |
| 8im0_k3m5_kernel_nonudge | 8im0 | ✓ | ✓ | | | ✓ | *running (Fable)* | 80+ | 5 | 0.049 | in progress | — | — |
| 8z8v_k3m5 | 8z8v | | | | | | timeout | 139 | 8 | 0.070 | 43311s | 12.03 | 37.1 GB |
| 8z8v_k3m5_jz0 | 8z8v | ✓ | | | | | *running* | 64+ | 4 | 0.041 | in progress | — | — |

_*Peak VRAM not recorded for `8vvl_intrac_slmonly` (pre-instrumentation log) and `8urf_k3m5_fulldisto`
(interrupted before its summary-writer ran). Several rows above (`8jyr_k3m5_kernel`,
`8vvl_k3m5_kernel_intrac`, `8urf_k3m5_fulldisto`, `8vvl_k3m5_fulldisto`, `8im0_k3m5_jz0`) are
missing from `campaign_summary.csv` — either killed via SIGTERM before the summary-writer ran, or
(for `8im0_k3m5_jz0`) an apparent lost write like the earlier `8vvl_intrac_slmonly` case; numbers
above are recovered directly from each run's log file. 8vzo's ~82-86 GB peak is the largest in the
set by far — it's a bigger complex than the other 5 targets, runs ~4x slower per-run (~750s vs.
~175-590s elsewhere)._

## Current live campaign status (as of 2026-07-02 ~23:00)

| GPU | Owner | Target | Config | Status |
|---|---|---|---|---|
| 0 | us | 8urf | kernel, joint-z0, no-nudge | running, cyc 4, best 0.090 |
| 1 | **Fable** | 8im0 | kernel, joint-z0, no-nudge | running, cyc 5, best 0.049 |
| 3 | **Fable** | — | unrelated FoldBench sweep (30 complexes × 10 seeds) | running |
| 6 | us | 8vvl | kernel, joint-z0, intra-c, no-nudge | running, cyc 3, best 0.168; queued next: 8vzo (kernel, no-nudge, no intra-c) |
| 7 | us | 8z8v | mean, joint-z0, nudged | running, cyc 4, best 0.041, declining |

## Data gaps / open questions

1. **`pair_iptm` / per-interface DockQ is not available for any steering campaign.** The code to
   save it landed 2026-07-02 ~08:32, but every campaign that started before ~09:36 that day ran the
   older in-memory code. Backfilling would need a re-decode + re-OST pass per campaign (~10-20 min
   GPU time total for the 7 recoverable s_lm-only campaigns; the joint-z0 hits can't be backfilled
   this way since their winning z0 was never saved to disk before the `z0_hit_<tag>.npy` fix) —
   **still an open question, not yet answered.**
2. **Several `campaign_summary.csv` rows are missing** (`8vvl_intrac_slmonly`, `8im0_k3m5_jz0`, and
   everything killed via SIGTERM before its summary-writer ran) — root cause for the two
   naturally-completed-but-missing rows not confirmed (best guess: an unlocked
   check-header-then-append race with a concurrently-finishing campaign).
3. **Full-disto ablation is incomplete** — 8urf's run was interrupted mid-campaign and 8vvl's never
   started, both when GPU3 was repurposed. The 2-target pattern (full-disto underperforms, and
   degrades over cycles rather than plateauing) is consistent and large, but isn't a fully concluded
   ablation.
4. **No-nudge is not yet a settled recommendation.** Two dramatic wins (8jyr, 8vzo — the latter also
   beating its established baseline) but no effect so far on 8vvl+intrac and no repeat on 8urf.
   Mechanism for why it helps in some cases and not others is not understood yet.
5. **Auto-trial-end idea (min-W1-to-nearest-anchor stopping signal)** — proposed, not implemented.
   Requires new instrumentation (the signal was never logged); recommended calibration path is to
   add the signal and re-check it against the already-known-stuck `8vvl_k3m5_jz0` trajectory before
   picking patience/threshold numbers, then validate on 7txw before rolling out further.
