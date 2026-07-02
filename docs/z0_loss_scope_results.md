# Joint-z0 & loss-scope ablation — results

_Last updated: 2026-07-02. Follow-on to [`repel_campaign_6targets_handoff.md`](repel_campaign_6targets_handoff.md)
covering the 6-target label-free repulsion campaign. Recipe unless noted: `M=5, K=3, lr=0.3,
init_frac=0.05`, loss = mean-over-anchors, `cycle-decode=16`, stop at DockQ ≥ 0.23, 12h budget per
campaign. "z0" = jointly optimizing the initial pair state `z0` alongside `s_lm` (`--joint-z0
--lr-z0 0.01`); see [`joint-z0-intra-antigen-loss` memory] for the mechanism._

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

**Important caveat on the steering-campaign numbers below.** The "best DockQ" reported by
`repel_proven.py`'s own scoring (`_abag()`) is the **max over whichever ab–ag interfaces exist**
for that target — for the four Fab targets this collapses two real chain-pair DockQ values into
one number and **discards which interface won and what the other one scored**. Per-interface
breakdown and `pair_iptm` were **not saved** by any steering campaign to date (see
[Data gaps](#data-gaps--open-questions) — the code to save them landed today, after every campaign
below except the just-launched one had already started). The **baseline (unsteered) sweep**, by
contrast, used the full multi-interface OST path and *does* have true per-interface DockQ +
`pair_iptm` — see [Baseline control](#baseline-unsteered-control).

## Stage 1 — does joint-z0 help? (mean loss, interface-only scope)

| Target | Config | Stop | Runs / cycles | Best DockQ | Best global ipTM | Time to result | GPU-hours | Peak VRAM |
|---|---|---|---|---:|---:|---:|---:|---:|
| 8urf | s_lm-only | HIT | 80 / 5 | 0.378 | 0.869 | 16207s (4.5h) | 4.50 | 25.6 GB |
| 8urf | **+ joint-z0** | HIT | 80 / 5 | **0.592** | 0.868 | 16243s (4.5h) | 4.51 | 27.8 GB |
| 8jyr | s_lm-only | HIT | 32 / 2 | 0.952 | 0.895 | 7802s (2.2h) | 2.17 | 29.1 GB |
| 8jyr | + joint-z0 | HIT | 32 / 2 | 0.951 | 0.889 | 7809s (2.2h) | 2.17 | 31.7 GB |
| 8vzo | s_lm-only | HIT | 48 / 3 | 0.544 | 0.714 | 38963s (10.8h) | 10.82 | 82.1 GB |
| 8vzo | + joint-z0 | *running* | 39+ / 2 so far | 0.181 (cyc 2, climbing from 0.123) | 0.725 | in progress | — | — |
| 8vvl | s_lm-only | TIMEOUT | 137 / 8 | 0.167 (peak @ cyc 7) | 0.703 | 43313s (12.0h, exhausted) | 12.03 | 37.8 GB |
| 8vvl | + joint-z0 | *running* | 15+ / 0 so far | no cycle yet | — | in progress | — | — |
| 8im0 | s_lm-only | EXHAUSTED | 512 / 32 | 0.047 (peak @ cyc 29) | 0.897 | 28582s (7.9h, exhausted) | 7.94 | 10.5 GB |
| 8im0 | + joint-z0 | *running* | 185+ / 7 so far | 0.018–0.033, flat | 0.887–0.897 | in progress | — | — |
| 8z8v | s_lm-only | TIMEOUT | 139 / 8 | 0.070 (peak @ cyc 7) | 0.877 | 43311s (12.0h, exhausted) | 12.03 | 37.1 GB |
| 8z8v | + joint-z0 | *queued* (after 8im0, GPU7) | — | — | — | — | — | — |

**Reading it so far:** one clean win (8urf: +0.21 DockQ, and baseline-alone only reaches 0.404 at
this interface — see below — so this isn't a ceiling effect). 8jyr is a wash (~0, both saturated).
8vzo and 8vvl-interface-only are still running and are the two comparisons that will actually tell
us whether the win generalizes. 8im0's z0 run is trending flat/no better than its own s_lm-only
run, but 8im0 has no known reachable basin under *any* method so far, so a null result there isn't
informative either way. z0 costs ~2–6 GB extra peak VRAM (well within headroom) and has not yet
made anything worse — reasonable to default it on going forward regardless, pending 8vzo/8vvl
confirming or not confirming a DockQ benefit.

## Stage 2 — loss scope: interface-only vs. +intra-antigen vs. full-distogram (on joint-z0)

8vvl was diagnosed as an antigen-*misfolding* case (antigen-only lDDT 0.208 vs 0.79–0.92 for the
other 5 targets — see [`joint-z0-intra-antigen-loss`] memory), so it's the target used to test
whether broadening the loss beyond the interface actually matters:

| Target | Loss scope | z0 | Stop | Runs / cycles | Best DockQ | Best ipTM | Time | GPU-hours |
|---|---|:-:|---|---|---:|---:|---:|---:|
| 8vvl | interface-only | no | TIMEOUT | 137 / 8 | 0.167 | 0.703 | 43313s | 12.03 |
| 8vvl | interface-only | **yes** | *running* | 15+ / 0 | no cycle yet | — | in progress | — |
| 8vvl | **+ intra-antigen** | no | TIMEOUT | 137 / 8 | 0.157 (peak @ cyc 2) | 0.742 | 43379s | 12.05 (peak VRAM not recorded — pre-instrumentation log) |
| 8vvl | **+ intra-antigen** | **yes** | **HIT** | 16 / 1 | **0.305** | 0.580 | 5099s (1.4h) | 1.42 |

Neither lever alone got 8vvl past 0.17; both together hit 0.305 in a single cycle (16 runs, 1.4h —
by far the cheapest hit of any campaign here). The one open cell (interface-only + z0, running on
GPU1 now) will show whether z0 alone can substitute for the intra-antigen term, or whether the
term is genuinely necessary for a misfolding target.

**Full-distogram** (untargeted: loss/anchors over the entire L×L distogram instead of targeted
blocks) is running on GPU3, sequence 8jyr → 8urf → 8vvl, all with joint-z0:

| Target | Cycle | Best DockQ | Best ipTM | Elapsed |
|---|:-:|---:|---:|---:|
| 8jyr (full-disto) | 1 | 0.128 | 0.880 | 500s |
| 8jyr (full-disto) | 2 | 0.123 | 0.738 | 1000s |

Compare to 8jyr's targeted interface-only+z0 run, which had *already hit* 0.951 by cycle 2 at a
similar elapsed time — early evidence for the predicted dilution effect (more, less-relevant pairs
under a plain mean loss blur the gradient signal), but only one target's worth of data, and 8urf/
8vvl haven't started under this arm yet.

## Stage 3 — loss function: mean vs. kernel (planned)

Not yet implemented — `--loss` currently supports `mean` and `softmin` only. The plan (see
[`summed_kernel_loss.md`](summed_kernel_loss.md) for the full derivation) is a `kernel` mode:
`bias = Σ_a exp(-(d_a/h)²)`, minimized directly, with bandwidth `h` self-calibrated from the
anchors' own nearest-neighbor spread. Once implemented, evaluate on targets where the mean loss
demonstrably has room to improve — **8vvl, 8im0, 8z8v** — rather than 8jyr, where baseline-alone
already saturates and wouldn't distinguish the loss functions.

## Baseline (unsteered) control

Only 8jyr and 8urf have a baseline sweep (1024 seeds × 5 diffusion samples each, no steering at
all) — this tests whether scale alone reaches the same place steering does.

| Target | Interface | Best DockQ | Seed / sample | pair_iptm | global ipTM | ptm | Hit rate (≥0.23) | Sweep status |
|---|:-:|---:|---|---:|---:|---:|---:|---|
| 8jyr | A–B | **0.961** | seed 979 / s1 | 0.647 | 0.907 | 0.913 | 22/5120 (0.4%) | complete (15.3h, 1024 seeds) |
| 8jyr | A–C | 0.936 | seed 388 / s0 | 0.597 | 0.891 | 0.902 | 32/5120 (0.6%) | complete |
| 8jyr | *collapsed (max)* | 0.961 | seed 979 / s1 | 0.647 | 0.907 | — | 38/5120 (0.7%) | complete |
| 8urf | A–C | 0.404 | seed 611 / s4 | 0.125 | 0.560 | 0.658 | 1/3310 (0.03%) | **~65% done, running** |
| 8urf | B–C | 0.162 | seed 611 / s4 | 0.123 | 0.560 | 0.658 | 0/3310 | running |
| 8urf | *collapsed (max)* | 0.404 | seed 611 / s4 | 0.125 | 0.560 | — | 1/3310 (0.03%) | running |

**Reading it:** 8jyr's baseline-alone reaches 0.961 unsteered (better than either steered run,
0.951–0.952) — this target does not demonstrate that steering was necessary. 8urf's baseline-alone
tops out at 0.404 so far (and is unlikely to move much more the way it's trending), well below both
steered results (0.378 s_lm-only, 0.592 +z0) — this is the clean case where steering is doing real
work, not just getting lucky faster. No baseline sweep exists for 8vzo, 8vvl, 8im0, or 8z8v.

## Compute & memory summary (all completed campaigns)

| Tag | Target | z0 | intra-c | full-disto | Stop | Runs | Cycles | Best DockQ | Elapsed | GPU-h | Peak VRAM |
|---|---|:-:|:-:|:-:|---|---:|---:|---:|---:|---:|---:|
| 8jyr_k3m5 | 8jyr | | | | hit | 32 | 2 | 0.952 | 7802s | 2.17 | 29.1 GB |
| 8jyr_k3m5_jz0 | 8jyr | ✓ | | | hit | 32 | 2 | 0.951 | 7809s | 2.17 | 31.7 GB |
| 8urf_k3m5 | 8urf | | | | hit | 80 | 5 | 0.378 | 16207s | 4.50 | 25.6 GB |
| 8urf_k3m5_jz0 | 8urf | ✓ | | | hit | 80 | 5 | 0.592 | 16243s | 4.51 | 27.8 GB |
| 8vzo_k3m5 | 8vzo | | | | hit | 48 | 3 | 0.544 | 38963s | 10.82 | 82.1 GB |
| 8vvl_k3m5 | 8vvl | | | | timeout | 137 | 8 | 0.167 | 43313s | 12.03 | 37.8 GB |
| 8vvl_intrac_jz0 | 8vvl | ✓ | ✓ | | hit | 16 | 1 | 0.305 | 5099s | 1.42 | 39.2 GB |
| 8vvl_intrac_slmonly | 8vvl | | ✓ | | timeout | 137 | 8 | 0.157 | 43379s | 12.05 | not recorded* |
| 8im0_k3m5 | 8im0 | | | | exhausted | 512 | 32 | 0.047 | 28582s | 7.94 | 10.5 GB |
| 8z8v_k3m5 | 8z8v | | | | timeout | 139 | 8 | 0.070 | 43311s | 12.03 | 37.1 GB |

_8vzo's 82 GB peak is the largest in the set — worth watching if more targets of similar size get
queued alongside other jobs on the same card. *This row's entry is missing from
`campaign_summary.csv` (looks like a lost write, not a run failure — the log itself completed
cleanly); numbers above are recovered directly from `log_8vvl_intrac_slmonly.txt`._

## Data gaps / open questions

1. **`pair_iptm` is not in any dockq CSV yet.** The code to save it (`pair_iptm`, `ptm` columns)
   landed in `repel_proven.py` today (2026-07-02, ~08:32), and every campaign above except the
   just-launched `8vvl_k3m5_jz0` (started 09:36) had already started with the older in-memory code
   before that — Python doesn't hot-reload a running process, so they'll finish without it too.
   Going forward this is solved automatically. For everything already completed:
   **do you want a light re-score/backfill pass** (re-decode the single best-known run per
   campaign and re-run OST keeping the full per-interface JSON, instead of the collapsed max) to
   recover true per-interface DockQ + `pair_iptm` retroactively? This is cheap — one decode +
   one OST call per campaign, on the order of 10–20 minutes total GPU time on a free GPU — for the
   **7 s_lm-only campaigns** (leaves + fixed `z0_base` fully determine the decode, so it's
   reproducible). The **3 joint-z0 HITs** (`8jyr_k3m5_jz0`, `8urf_k3m5_jz0`, `8vvl_intrac_jz0`)
   can't be backfilled this way — their winning per-run `z0` was only buffered cycle-locally and
   was never written to disk (this predates the `z0_hit_<tag>.npy` save-on-hit fix), so the exact
   winning state is gone.
2. **No per-interface split for any steering-campaign number above** — same root cause as #1
   (the scoring path takes `max()` over interfaces before saving). The backfill in #1 would also
   fix this for the 7 recoverable campaigns.
3. **The `campaign_summary.csv` row for `8vvl_intrac_slmonly` is missing** even though its log
   completed normally — I recovered the numbers from the log directly, but haven't root-caused why
   the row-write silently didn't happen (my best guess is a write racing with another campaign
   finishing at the same time — the header-check-then-append isn't lock-protected — but I haven't
   confirmed this). Not urgent, but flagging in case you'd like it patched.

Let me know how you'd like to handle #1/#2 (skip it, or spend the ~10-20 min GPU time on the
backfill) and I'll fold the results back into this doc.
