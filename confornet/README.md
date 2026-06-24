# ConforNet 7txw steering — runbook for the next session

**This is the single, self-contained document for running the ConforNet docking-steering
experiment on a fresh GPU server.** It assumes no prior context. Read top to bottom; every
command is exact. Deeper science background (optional) is in
[`../docs/steering_research_brief.md`](../docs/steering_research_brief.md).

---

## 0. Status — read first

| Piece | State |
|-------|-------|
| Code (`confornet/` core, decode, eval wiring) | ✅ ready; imports + syntax verified |
| `confornet/data/` bundle (~300 MB) | ⛔ **NOT built yet** — must run `prepare_data.py` on the **main** server first (§2) |
| First GPU run of `steer_confornet.py` | ⛔ **never executed** — the smoke test in §4 is also its first real test of the affine→trunk gradient path |
| Decode → DockQ path | ✅ code ready; unrun |

So before this runbook works end-to-end, **two GPU steps must happen on the main server**
(build the bundle, smoke-test), then the bundle + repo are copied here. §1–§2 cover that.

---

## 1. The experiment in one paragraph

`7txw` is a Fab antibody (chains A=heavy, B=light) + antigen (chain C). ESMFold2-Fast folds
both monomers correctly (~1 Å) every run; only the **antigen docking pose** varies, landing in
discrete **basins**: **G** = native (DockQ ~0.9), **B1** = default decoy (the universal start),
**B2** = a *confident* wrong-register decoy (DockQ ~0.03 but ipTM ≈ G's). Goal: reliably and
cheaply generate **G**. The method here (ConforNet) optimizes a tiny **channel-wise affine**
`φ(z) = z·Wᵀ + b` (W: 256×256 identity-init, b: 256 zero-init, shared over all L×L positions) on
the pair latent `lm_z`, with **k particles pushed apart** by maximizing the pairwise **W1 distance
between their K=3 cross-block (antibody×antigen) distograms** — a cheap, label-free repulsion. We
then run each steered particle the full 17 recycling loops and classify its basin by W1 distance
to the G/B1/B2 centroids. **Open question this experiment answers:** can a *global* affine move a
*discrete* docking pose (B1→B2→G)? ConforNets was validated only on monomer conformations, never
multimer docking — so this is genuinely unknown.

**Why it's cheap to ship:** the steering only needs 3 small centroid distograms + one baseline
representation, not the 195 GB of cached reps. `prepare_data.py` bakes those into a ~300 MB bundle.

---

## 2. Build the bundle (ONE time, on the MAIN server)

The main server is the one holding `~/esm/foldbench_predictions/7txw_reps` (195 GB). Run:

```bash
cd ~/tts_abag
~/esm/.venv/bin/python confornet/prepare_data.py     # needs 1 GPU; ~minutes
# -> writes confornet/data/{centroids/{G,B1,B2}.npy, baseline/..., jsons/7txw.json,
#    ground_truths/7txw-assembly1.cif, targets/interface_antibody_antigen.csv}  (~300 MB)
```

Then copy the repo + bundle to the new server (the bundle is git-ignored, so rsync it explicitly):

```bash
# new server needs the whole repo (paths.py, run/esm/, eval/) AND the bundle:
git clone <tts_abag remote> ~/tts_abag                 # or rsync the repo tree
rsync -a ~/tts_abag/confornet/  newserver:~/tts_abag/confornet/   # carries data/
```

---

## 3. Set up the NEW server

**Environment** — the experiment runs under the **`esm` fork**'s venv, which brings a **patched
`transformers`** containing `models/esmfold2` (vanilla PyPI transformers will NOT work). Python
3.12, an NVIDIA GPU, HuggingFace access to `biohub/ESMFold2-Fast` (weights auto-download).

```bash
git clone https://github.com/Biohub/esm.git ~/esm
cd ~/esm && uv sync            # uv.lock pins the patched transformers + torch 2.12+cu130
# (alt: pip install esm@git+https://github.com/Biohub/esm.git@main)
```

`esm`'s `pyproject.toml` pulls `transformers @ git+https://github.com/Biohub/transformers.git@main`.
Venv lands at `~/esm/.venv`; if it differs, export `ESM_PYTHON=/path/to/python`.

**For DockQ only (§6):** a `foldbench` conda env providing the `ost` binary. Set `OST_BIN`/`OST_LIB`
in `confornet/run_newserver.sh` to this host's env. Steering itself (§5) needs no OST.

**Verify setup (no GPU needed):**

```bash
cd ~/tts_abag && python paths.py | grep -E 'CONFORNET_DATA|ESM_PREDICTIONS'   # paths resolve
cd ~/tts_abag/confornet && CUDA_VISIBLE_DEVICES="" ~/esm/.venv/bin/python -c \
  "import basins,forward_grad,steer_confornet,decode_confornet; print('imports OK')"
ls ~/tts_abag/confornet/data/centroids   # must show G.npy B1.npy B2.npy (else bundle missing)
```

---

## 4. Smoke test (FIRST — this is the first-ever GPU run of the steering code)

```bash
cd ~/tts_abag
CUDA_VISIBLE_DEVICES=0 ~/esm/.venv/bin/python confornet/steer_confornet.py --max-runs 1 --tag smoke
```

**Expected:** lines like `loading precomputed centroids from .../data/centroids` then
`run 0: basins=['B1', 'B2'] separated=True repel=0.83 | totals G=… realG=…` and a clean exit.

**If it tracebacks:** the affine→trunk gradient/dtype path has a bug (it has never run). Likely
spots: autocast dtype mismatch where `net(lm_z0)` (float) feeds `run_trunk` (bf16 autocast), or a
grad flowing into a frozen buffer. Fix in `confornet/forward_grad.py` (`run_trunk`/`_loop_body`) or
`steer_confornet.py` (`distogram`/`ConforNet.forward`). This is independent of the reorg.

---

## 5. Experiment 1 — PRIMARY run (ready)

```bash
tmux new -s confornet
cd ~/tts_abag
bash confornet/run_newserver.sh        # waits for ONE fully-idle GPU, then runs continuously
```

`run_newserver.sh` exports the bundle paths and execs `confornet_one.sh` (fair single-GPU use).
To run directly on a chosen GPU instead:

```bash
CUDA_VISIBLE_DEVICES=N ~/esm/.venv/bin/python confornet/steer_confornet.py --tag gN --seed-base N000000
```

**CLI knobs** (default): `--k` particles (2) · `--steps` (21) · `--lr` (1e-3, the ConforNet-paper
value for the affine) · `--K` repulsion horizon (3) · `--noise` identity-break (0.01) ·
`--tag` (label) · `--seed-base` (0) · `--max-runs` (0 = infinite).

**Outputs** (minimal by design — never the 144 MB z):
- `results/csv_files/steer_confornet_<tag>.csv` — one row per particle: `run, particle, basin,
  wg, wb1, wb2, repel`.
- For each real-G hit (`basin==G & wg<0.08`): the ~256 KB affine at
  `$ESM_PREDICTIONS/7txw_confornet/<tag>/run<NNNNN>_p<i>/confornet.pt`. (With `run_newserver.sh`,
  `$ESM_PREDICTIONS` = `confornet/data/baseline`, so hits land under `data/baseline/7txw_confornet/`.)

### How to read the result (the headline)
- **`separated=True`** → the k particles landed in *different* basins (the affine CAN diversify the
  pose). `separated=False` every run → a global affine cannot split the docking pose → go to §7.
- **`realG>0`** → a particle reached true G (`wg<0.08`). This is the win condition.
- Track the rate of `separated` and `realG` over ~50–100 runs. Compare against the native baseline:
  mask+dropout TTS finds G ~4%/seed. Beating that on compute-to-first-G is the bar.

---

## 6. Validate a G hit with DockQ

```bash
cd ~/tts_abag
~/esm/.venv/bin/python confornet/decode_confornet.py --tag gN          # regenerate z from affines -> cif
~/esm/.venv/bin/python eval/eval_ost_foldbench.py \
    --pred_dir $ESM_PREDICTIONS/7txw_confornet_decode_gN \
    --out_csv results/csv_files/dockq_confornet_gN.csv
```

`decode_confornet.py` reloads each saved affine, rebuilds the trunk z exactly as the run evaluated
it (`lm_z = affine(lm_z0)` → 17 loops), runs the diffusion head, writes 5 samples/hit. A true-G hit
should score **DockQ ≈ 0.85–0.9**. (Caveat: `wg<0.08` is a *label-based detector* using the G
centroid — fine for evaluating this target, but it is NOT a generalizable selector. For a new
antibody you have no G centroid; selection there must fall back to ipTM, which ranks good seeds at
AUC 0.945 but cannot distinguish the confident B2 decoy.)

---

## 7. Follow-on experiments — decision tree

Run these **only if** Experiment 1's result calls for it. The first is ready; the rest need code
(precise specs given so the next session can build them).

**7a. Tune the primary (ready, same script).** If results are borderline (occasional `separated`,
no `realG`), sweep: larger `--k` (3–4) for more pose diversity; `--lr` 3e-3 / 1e-2 (affine may need
a bigger step than the paper's 1e-3 to move a discrete pose); more `--steps` (31). Watch `repel` —
if it saturates early, lr is too low; if particles go off-manifold (decode DockQ collapses), too high.

**7b. Coordinate repulsion `coord_mse` (NOT BUILT).** If distogram repulsion separates basins but
none reach G, the distogram metric may be too coarse. Spec: in `steer_confornet.py`, add a repulsion
mode that decodes each particle's z through the diffusion head (`diffuse_from_z`, 1 sample, few
steps) and maximizes pairwise MSE between *aligned decoded coordinates* instead of `w1` on
distograms. Costlier (adds a diffusion forward per step) — gate behind a `--repel coord` flag.

**7c. Block-wise / per-region affine (NOT BUILT).** If a *global* affine can't separate the pose
(`separated=False` always, §5), the channel-wise-global constraint is too rigid. Spec: change
`ConforNet` so `W`/`b` are NOT shared across all L×L positions — e.g. a separate affine for the
antibody×antigen cross-block (`AB,C` from `forward_grad`) vs. the diagonal blocks. This trades some
protein-agnosticism for the ability to act on the docking interface specifically.

**7d. Supervised `aligned_mse` (later, only with a reference good complex).** Steer toward a known-G
structure instead of repelling. Out of scope unless a reference G is available; defeats the
label-free goal, so it's a fallback only.

---

## 8. Gotchas & etiquette

- **Untested gradient path** — §4 is the first GPU execution; expect to possibly debug the
  affine→trunk backward once.
- **Disk: 1 TB `/home` quota.** Steering saves only CSV + ~256 KB affines (decode adds ~MB of cif).
  Never save the 144 MB pairformer z per run (that crashed earlier runs). `quota -s` before big runs.
- **GPU etiquette: shared cluster.** Use `confornet_one.sh` / `run_newserver.sh` (ONE fully-idle
  GPU, fair use). **Do NOT** use `archive/steering/confornet_runner.sh` — it auto-grabs multiple
  shared GPUs (the banned version).
- **Generalizability guardrails** (the whole point): the *steering objective* must stay label-free —
  no hardcoded G-centroid target, no forcing the known good mask residues (C48–C50), no B2-centroid
  repulsion. The G/B1/B2 centroids are used ONLY for the offline `wg` readout/classification on this
  target, not to drive optimization.

---

## 9. File map

```
prepare_data.py     build the bundle (MAIN server, once)            [§2]
steer_confornet.py  the experiment: k-affine + K=3 distogram repel  [§5]
forward_grad.py     differentiable trunk -> cross-block distogram (engine)
basins.py           B1/B2/GOOD + centroid() + load_centroids()
decode_confornet.py affine -> regenerate z -> structures (for DockQ) [§6]
confornet_one.sh    wait for ONE idle GPU, run steer_confornet
run_newserver.sh    export bundle paths, then confornet_one.sh      [§5]
data/               the ~300 MB bundle (git-ignored)
```
