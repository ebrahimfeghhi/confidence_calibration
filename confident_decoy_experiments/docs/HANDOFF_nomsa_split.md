# HANDOFF — run 6 no-MSA template arms (split of the 25-interface sweep)

**Written 2026-08-09.** A 28 h sweep is being split across machines. **You take the 6 slowest runs**;
the origin box is running the other 17. Nothing here depends on the other machine — the runs are
fully independent.

---

## What this experiment is

Protenix inference where **each chain is templated from a single donor pose** (an injected 4-character
mmCIF) and the **MSA is replaced by a 1-sequence a3m** containing only the query. 50 seeds (101–150)
× 5 diffusion samples per run.

The question: on interfaces where templating already works, does *removing the MSA* make a good
template produce a good pose more consistently? Early data is strongly bimodal — `8fas_BC` went from
27/50 seeds acceptable and an ipTM-selected DockQ of 0.169 to **50/50 seeds and 0.982** (an interface
nothing had ever fixed), while `8aok_AB` got worse (14/50 → 0/50 seeds above 0.6). We need the rest of
the interfaces to know which way it goes on average.

Templates carry **intra-chain geometry only** — Protenix zeroes every cross-chain template feature via
`multichain_mask` (`pairformer.py:1013`), so a donor never transmits a binding arrangement.

---

## 1. Get the code

```bash
cd <tts_abag>                      # github.com/ebrahimfeghhi/tts_abag
git fetch && git checkout reorg/declutter && git pull
```

## 2. Prerequisites

All are referenced by **absolute path**, so they must exist locally:

| what | path |
|---|---|
| Protenix repo + venv | `/home/ebrahim/Protenix`, `/home/ebrahim/Protenix/.venv/bin/python` |
| protenix-v2 checkpoint | `/home/ebrahim/checkpoint` |
| arm metadata | `confident_decoy_experiments/artifacts/` — `dockq_arm_jobs.json`, `dockq_arm_donors.json`, `dockqdonor_scores.json`, `sweep_tmpl/dockqdonor/*.json` |
| injected donor templates | `$HOME/mmcif/*.cif` |
| template release dates | `$HOME/common/release_date_cache.json` (must contain entries for those mmCIFs) |

`PROTENIX_ROOT_DIR` defaults to `$HOME`, which is why the last two live there.

**If the donor mmCIFs or their release-date entries are missing**, don't try to regenerate them — say
so. There is a 6.6 MB self-contained package at `/home/ebrahim/nomsa_handoff/` on the origin box
(6 template mmCIFs + a3ms + inputs + a standalone `setup_and_run.sh` that installs them and runs
everything). Use that instead.

## 3. Run

```bash
python confident_decoy_experiments/scripts/nomsa_donors.py \
  --subset all --n-seeds 50 --gpu <GPU> \
  --only 8pn0_nomsa_126_4,8bf4_nomsa_120_2,8be1_nomsa_142_4,8q3j_nomsa_102_4,8q3j_nomsa_137_2,8the_nomsa_113_4
```

| run | interface | est. time (1 GPU) |
|---|---|---|
| `8pn0_nomsa_126_4` | 8pn0_BE | 5.0 h |
| `8bf4_nomsa_120_2` | 8bf4_AB | 3.8 h |
| `8be1_nomsa_142_4` | 8be1_AC | 2.3 h |
| `8q3j_nomsa_137_2` | 8q3j_AB | 2.1 h |
| `8q3j_nomsa_102_4` | 8q3j_AC | 2.1 h |
| `8the_nomsa_113_4` | 8the_AB | 2.1 h |
| | | **~17 h serial** |

For N GPUs, split `--only` into N invocations and run them concurrently — one job per GPU (a Protenix
process peaks near 50–90 GB depending on complex size; do not put two on one card).

**Resumable**: seeds that already have output are skipped, so a kill and relaunch loses only the
in-flight job. `8bf4_nomsa_120_2` may already have 10 of its 50 seeds on the origin box; if you start
clean you'll simply produce all 50, which is fine.

## 4. Verify — this is not optional

A condition failure is **invisible in the output**: a run with no template, or with the MSA still
attached, still emits 250 plausible poses. Three arms on this project were scored and believed for a
week while silently untemplated. The script prints `VERIFIED` / `FAILED` per run; confirm in
`confident_decoy_experiments/artifacts/nomsa_donors_logs/<name>.log`:

- `Found 1 templates` on every chain load — **`Found 0 templates` means the run was untemplated**
  (usually a template-id that isn't exactly 4 characters, which Protenix rejects)
- `N_msa 1` — the MSA really is single-sequence
- **no** `UNMASKED TEMPLATES ACTIVE` line — this arm must use the stock mask
- 50/50 seeds produced `*_summary_confidence_sample_4.json`

**Report a FAILED run as failed. Do not score it, and do not report its numbers.**

## 5. Send back

```bash
tar czf nomsa_results.tgz -C confident_decoy_experiments/artifacts/sweep_out nomsa_donors
```

~90 MB per run (CIFs + confidence JSONs). **Do not score on your machine** — scoring needs OST and the
FoldBench ground-truth assemblies, and is done on the origin box so both halves of the sweep go through
one identical protocol (`--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq`, interface
matched by chain name).

Along with the tarball, report per run: the four verification checks above, wall-clock, and any run
that failed or was left incomplete.
