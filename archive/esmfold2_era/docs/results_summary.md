# Repulsion vs. baseline — results summary

_Last updated: 2026-06-30. Baselines from the fixed OST pipeline
(`dockq_ost_esmfold2fast_tts_v2_all.csv`, 51-seed `esmfold2fast_tts`)._

**DockQ is on the antibody–antigen interface.** Most targets here are 3-chain Fabs with **two**
ab–ag interfaces — heavy–antigen and light–antigen — so the single "best DockQ" is the **max over
both** (per-interface values are shown in the second column). 2-chain VHH/Nb have one interface.
`best DockQ` = best sample over all seeds. `seeds ≥t` = number of seeds whose best sample reaches
DockQ ≥ t. **The threshold that matters is ≥0.23**; ≥0.80 (high-quality) is shown too.
"Our method" = label-free / GT-guided distogram repulsion.

| Target | ab–ag interface(s) | Best DockQ | seeds ≥0.23 | seeds ≥0.80 | Our best | Rescue? |
|--------|--------------------|-----------:|:-----------:|:-----------:|----------|---------|
| 7txw | A–C 0.95 / B–C 0.92 | 0.948 | 5/51 | 4/51 | 0.91 | n/a — baseline solves |
| 8ts0 | A–C 0.95 / B–C 0.93 | 0.947 | 11/51 | 9/51 | 0.94 | no — baseline already solves |
| **8f60** | A–C **0.94** / B–C none | **0.944** | **51/51** | **41/51** | not attempted | n/a — baseline solves (was a scoring artifact, see below) |
| 8b7h | A–C 0.88 / B–C 0.78 | 0.880 | 51/51 | 32/51 | not attempted | n/a — baseline solves |
| **8c5h** | A–B (2-chain) | 0.658 | 11/51 | 0/51 | **0.82–0.92** | **✅ yes — only confirmed ≥0.80 rescue** |
| 8dtn | A–B (2-chain) | 0.405 | 6/51 | 0/51 | not attempted | — |
| 8fxv | A–B (2-chain) | 0.371 | 38/51 | 0/51 | not attempted | — (often ≥0.23, never ≥0.80) |
| 8vvl | A–B 0.16 / A–C 0.13 | 0.163 | 0/51 | 0/51 | queued | — |
| 8urf | A–C 0.16 / B–C 0.08 | 0.156 | 0/51 | 0/51 | 0.13 | ❌ no (earlier 0.94 was H–L artifact) |
| 8jyr | A–B 0.10 / A–C 0.14 | 0.136 | 0/51 | 0/51 | queued | — |
| 8vzo | A–B 0.11 / A–C 0.13 | 0.131 | 0/51 | 0/51 | queued | — |
| 8z8v | A–B (2-chain) | 0.047 | 0/51 | 0/51 | ~0.01 | ❌ no (both fail) |
| 8im0 | A–B-2 (2-chain) | 0.020 | 0/849 | 0/849 | 0.03 | ❌ no (no reachable basin; separate 849-seed run) |

## Key findings
- **At the ≥0.23 threshold (the one that matters), baseline already reaches an acceptable dock for
  7 of 13 targets** (≥1 seed): 7txw, 8ts0, 8f60, 8b7h, 8c5h, 8dtn, 8fxv. The targets baseline never
  reaches ≥0.23 are **8urf, 8z8v, 8vzo, 8jyr, 8vvl, 8im0** (best DockQ 0.02–0.16).
- **Exactly one confirmed steering rescue at ≥0.80: 8c5h** (baseline 0.658 → steering 0.82–0.92).
  8c5h is 2-chain, so its A–B interface *is* the ab–ag interface. (Note: at ≥0.23, 8c5h baseline
  already reaches 11/51, so the rescue is specifically about high-quality ≥0.80 docks.)
- **No other target is a high-quality rescue.** 8ts0/8b7h/8f60 baseline already solve them;
  8urf/8z8v/8im0 fail under both baseline and steering on the ab–ag interface.
- **8f60 was a scoring artifact, not a failure.** OST's chain mapper groups chains by sequence and
  lumped the Ig-superfamily antigen (BTLA) with an antibody chain, **dropping the antigen** and
  reporting `chain_unmapped`/0.0 — when the real heavy–antigen DockQ is **0.94** (41/51 seeds ≥0.80,
  median 0.92). Now scored via an isolated-pair fallback (`dockq_status = ok_isolated`). The light
  chain genuinely doesn't contact the antigen, so B–C has no interface — the dock is heavy-chain
  mediated.
- **Three scoring bugs inflated/hid earlier results, all now fixed:**
  1. OST rigid-scores / chain-mapper crash → silent nans logged as DockQ 0 (mislabeled 8ts0, 8b7h as
     failures; they're not).
  2. Inline `max()` over interfaces → reported the heavy–light interface on 3-chain Fabs. This is
     what made 8urf look like a 0.94 rescue; the real ab–ag DockQ is 0.16.
  3. OST chem-group mis-mapping → dropped an Ig-like antigen entirely (8f60 reported 0.0 instead of
     its real 0.94). Fixed by scoring the antibody-chain + antigen pair in isolation.

## Caveats / open items
- **Per-interface, not just the max.** For Fabs the table's single number is `max(heavy–ag,
  light–ag)`. Where they diverge it matters: 8b7h is 0.88 (heavy) vs 0.78 (light, 0/51 ≥0.80), and
  8f60's antigen only contacts the heavy chain. If a per-chain metric is wanted, use the second
  column / the CSV directly.
- **Steered decodes are not persisted to CSV on the ab-ag interface.** 8urf's 0.13 and 8z8v's ~0.01
  come from one-off rescores; the repel script only saves `wg`. Persist per-run ab-ag DockQ so no
  claim depends on the tmux buffer.
- **8vvl / 8jyr / 8vzo are antigen-first** (ab-ag interface = A:B, A:C). The inline scorer assumes
  antigen=chain C, so for these it is monitoring-only — re-score decodes with the CSV interfaces,
  don't trust `dockq_<t>_k*m6.csv`. (The batch numbers above use the correct CSV interfaces.)
- 8vzo / 8jyr / 8vvl still have zero steering runs (queued behind the first three jobs).
- Baseline seed counts uneven: 51 for most targets vs 849 for 8im0 (and 800–2060 for 7txw in its
  larger runs; the value here is the 51-seed `esmfold2fast_tts` subset).
