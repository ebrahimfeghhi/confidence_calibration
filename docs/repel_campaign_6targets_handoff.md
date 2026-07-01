# 6-target repulsion campaign — handoff (launched 2026-06-30 ~19:22)

Self-contained notes for the next session. Six baseline-failing ab–ag targets are being
run through label-free distogram-repulsion campaigns to look for a DockQ ≥ 0.23 rescue.

## What's running

Three GPUs, two targets each, **sequential** (each target gets ≤12 h or until a hit, then
the GPU's second target starts). One long-lived process per target (model loaded once);
anchors accumulate **in memory** for the whole 12 h.

| tmux | GPU | order (≤12 h each) | L (residues) | antigen chain | baseline best DockQ (all <0.23) |
|------|-----|--------------------|:------------:|:-------------:|:-------------------------------:|
| `camp1` | 1 | **8urf → 8z8v** | 605 / 708 | C / A(2-chain) | 0.156 / 0.047 |
| `camp6` | 6 | **8jyr → 8vvl** | 642 / 707 | A / A          | 0.136 / 0.163 |
| `camp0` | 0 | **8im0 → 8vzo** | 235 / 1006| A(2-chain)/A   | 0.020 / 0.131 |

`8im0` is the known **uncrackable** one (no reachable good basin in all prior work) —
expect it to time out; it's included for completeness. `8vzo/8jyr/8vvl` are **antigen-first**
(antigen = chain A).

## The recipe (per target)

```
steer/experiments/campaign_gpu.sh <gpu> <struct1> <struct2>
# which runs, per structure:
repel_proven.py --target <t> --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 \
    --loss mean --trials 1 --runs 512 --no-score --fused --chunk-size 32 \
    --cycle-decode 16 --stop-dockq 0.23 --time-budget 43200 \
    --n-samples 5 --n-steps 200 --launch-batch 2 --tag <t>_k3m5
```

This is the **proven** sequential-greedy self-repulsion recipe (K=3, M=5, lr=0.3,
init_frac=0.05, mean-over-anchors loss, s_lm lever) plus new **campaign mode**:

- **Accumulate 16 anchors → batch-decode those 16 → score ab–ag DockQ + ipTM → repeat.**
  "Maintain the 16 anchors and do another 16" is automatic — the anchor list just keeps
  growing in the one process (no restart, no disk anchor persistence).
- **Stop** the target when any decoded sample reaches DockQ ≥ 0.23 (`--stop-dockq 0.23`)
  OR after 12 h (`--time-budget 43200`), then advance to the GPU's second target.
- `--no-score` skips the K=17 diagnostic for speed (leaves/DockQ unaffected).
- Decode: `n_samples=5, n_steps=200` (matches the validated steer→decode pipeline;
  efficiency comes from `--no-score`/`--fused`/batched launch, not from cutting steps).
- ab–ag DockQ uses the **FoldBench metadata antigen_chain** (not a hardcoded "C"), so the
  antigen-first targets score the correct interface. The repulsion loss cross-distogram
  uses `configure_target`'s per-target AB×C slices (antibody × antigen) — correct per target.

## Outputs (in `steer/experiments/`)

| file | contents |
|------|----------|
| `dockq_<t>_k3m5.csv` | **cumulative** per-(run,sample) `iptm,dockq` — the thing to watch |
| `repel_<t>_k3m5.csv` | per-run summary (basin/wg are nan under `--no-score`) |
| `repel_<t>_k3m5_steps.csv` | per-step repel loss + off-manifold drift |
| `plateau_<t>_k3m5.csv` | **per-run plateau watch**: `last_gain`, `plateaued` (0/1) |
| `leaves_<t>_k3m5.npy` | `[n_runs,1,L,256]` optimized s_lm, re-saved each cycle (crash-safety / post-hoc decode) |
| `decoded_<t>_k3m5/` | decoded CIFs `run{r}_s{k}.cif` |
| `log_<t>_k3m5.txt` | full campaign log (also streamed to the tmux pane) |

Quick check: `grep "cycle .* decode" steer/experiments/log_<t>_k3m5.txt` for per-cycle best
DockQ/ipTM, and `grep "CAMPAIGN HIT\|CAMPAIGN TIMEOUT" log_*.txt` for stop reasons.

## Plateau watch (requested)

Each run flags whether the repel loss is still climbing at step M=5. Inline
`[plateau-WARN] run N: repel still climbing at step 5 (last_gain=… > 0.01); M=5 may be too few`
and a row per run in `plateau_<t>_k3m5.csv`. If many runs are non-plateaued, M=5 is too few
for that target → raise `--M` (or use `--plateau` to run to convergence, capped `--max-steps`).

## Code changes made this session (`repel_proven.py`)

- **Fixed a real `--no-score` bug**: `del …P17` raised `NameError` on run 1 because `P17`
  is only assigned when scoring. Now `P17` is only deleted when it exists.
- New campaign flags: `--cycle-decode N`, `--stop-dockq THRESH`, `--time-budget SECS`,
  `--launch-batch B`, `--plateau-warn TOL`. All default off → the proven path is unchanged.
- `_cycle_decode()`: batched 0→17 launch (`final_trunk_z_batched`, chunked by `--launch-batch`,
  per-leaf fallback on OOM) + per-leaf diffusion decode + ab–ag DockQ/ipTM.
- Anchors stored **fp16** in campaign mode (bounds VRAM as they accumulate for 12 h; w1's
  cumsum promotes to fp32). Proven (non-campaign) path keeps fp32.
- Also **recaptured 8vzo's baseline rep** (it had only a `configs/` dir — capture never
  finished). `run/esm/capture_reps.py --complex 8vzo --run_name 8vzo_reps_baseline --seeds 0
  --lm_mask_pct 0.1 --lm_dropout 0.3` (matching the other 5). Log: `log_capture_8vzo.txt`.

Smoke test (8im0, 2 runs, cycle=2, 30 steps) passed: `cycle 1 decode … best DockQ=0.013
best ipTM=0.879` — full steer→anchors→batched-decode→DockQ→ipTM→CSV path works.

## How to monitor / resume

- Watch: `tmux attach -t camp1` (or camp6/camp0), or `tail -f steer/experiments/log_<t>_k3m5.txt`.
- Live DockQ: `column -s, -t steer/experiments/dockq_<t>_k3m5.csv | sort -t… ` or just eyeball
  the `cycle … best DockQ` log lines.
- **If a process dies** (OOM/crash): anchors are in-memory only, so relaunch starts fresh
  (`campaign_gpu.sh` re-run, or a single `repel_proven.py …` with the recipe above). The
  `leaves_<t>_k3m5.npy` written so far can be post-hoc decoded with `decode_batch.py --tag
  <t>_k3m5 --target <t>` if you want the structures without re-steering.
- **VRAM lever** if a big target (8vzo L=1006) OOMs: lower `--launch-batch` (env `LB=1`) or
  raise `--chunk-size` (env `CS=48`); campaign_gpu.sh reads `NS/STEPS/LB/RUNS/CYC/CS` env vars.

## Gotchas

- `--no-score` is incompatible with `--until-gt`/`--pairwise` (both need the K=17 score).
- Disk: ~148 G free on `/home` at launch (1 TB quota). CIFs are small but `leaves_*.npy`
  grow; fine for this run. Check `quota -s` if things get tight.
- These are the only 6 baseline-<0.23 ab–ag targets (per `docs/results_summary.md`). A rescue
  here (esp. 8vvl/8jyr/8vzo/8urf) would be the first label-free steering win beyond 8c5h.
