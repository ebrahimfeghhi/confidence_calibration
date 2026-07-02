# DockQ / OST scoring pipeline — audit handoff

**Purpose.** Hand a second reviewer a self-contained map of how we score decoded
antibody–antigen structures with OpenStructure (OST), the bugs found so far, and the
specific things to scrutinize. Written 2026-06-29 after three scoring bugs were found to
have contaminated our FoldBench "failure" list and inflated one apparent steering rescue.

> Plain-text doc on purpose (no LaTeX). All paths are absolute as used in the repo.

---

## 1. What the pipeline does

For each decoded structure we compute **interface DockQ** against the FoldBench ground-truth
assembly, and read off the **antibody–antigen** interface score (the metric we care about).

Flow:

1. Model decodes a structure → mmCIF.
2. `postprocess_cif()` patches the CIF into an OST-compatible mmCIF (runs the Protenix
   postprocess, in-process or via a `PROTENIX_PYTHON` subprocess).
3. `run_ost()` shells out to OST `compare-structures`.
4. OST: cleans structures → **computes a chain mapping** (model chains ↔ reference chains)
   → lDDT → **DockQ per interface** → writes JSON with parallel arrays `dockq[]` and
   `dockq_interfaces[]` (each interface = `[ref_c1, ref_c2, mdl_c1, mdl_c2]` — **positions 0,1 are
   the REFERENCE chains**, they equal `dockq_reference_interfaces[i]`; 2,3 are the model chains.
   Verified against OST source `mol/alg/scoring.py:1112-1122`, the action help, and a real
   `8tuz` JSON. So `extract_dockq`/`dockq_cif` matching `iface[0:2]` against the targets-CSV
   reference labels is **correct** — do NOT "fix" it to use `iface[2:4]`).
5. Caller selects the DockQ for the antibody–antigen interface.

---

## 2. File map (exact locations)

| Thing | Path |
|---|---|
| Core scorer (batch path) | `eval/eval_ost_foldbench.py` — `run_ost()`, `postprocess_cif()`, `extract_dockq()`, `process()` |
| Path/config constants | `eval/paths.py` |
| **Inline** scorer (steering runs) | `steer/experiments/repel_proven.py::dockq_cif` (decodes + scores each run on the fly) — **patched this session** |
| Output CSVs | `results/csv_files/dockq_ost_*.csv` |
| Decoded steering CIFs | `steer/experiments/decoded_<tag>/run<r>_s<k>.cif` |
| Decoded baseline/TTS CIFs | `~/esm/foldbench_predictions/<run>/<target>/seed_*/predictions/<t>_sample_*.cif` |

Resolved constants (from `eval/paths.py`):

```
OST_BIN       = /home/ebrahim/miniconda3/envs/foldbench/bin/ost
OST_LIB       = /home/ebrahim/miniconda3/envs/foldbench/lib   (set as LD_LIBRARY_PATH)
GROUND_TRUTHS = /home/ebrahim/FoldBench/examples/ground_truths   (<t>-assembly1.cif)
CSV_DIR       = /home/ebrahim/tts_abag/results/csv_files
```

The OST command we run:

```
OST_BIN compare-structures \
  -m <patched_model.cif> -r <native-assembly1.cif> -o out.json \
  --fault-tolerant --min-pep-length 4 --min-nuc-length 4 \
  --lddt --dockq
# run with env LD_LIBRARY_PATH=OST_LIB
# success = returncode 0 AND out.json exists; then read out.json["dockq"] / ["dockq_interfaces"]
```

---

## 3. Bugs found (chronological)

### Bug #1 — `--rigid-scores` crash discards a valid DockQ *(fixed)*
OST computed DockQ fine, then crashed on the rigid-superposition step
(`Exception: at least 3 points are required for superposition`), exited nonzero, and the
caller threw away the **entire** output including the already-computed DockQ → logged as 0.
**Fix:** dropped `--rigid-scores`/`--tm-score`, keep `--lddt --dockq` (see
`eval/eval_ost_foldbench.py::run_ost`, line ~137). This had hidden 8c5h's real score.

### Bug #2 — OST auto chain-mapper crashes on multi-chain natives *(fixed in inline path)*
Even without rigid-scores, OST's **automatic chain mapping** raises
`IndexError: arrays used as indices must be of integer (or boolean) type` during
"Computing chain mapping" on natives with many chains. Example: 8ts0's native has **15
chains** (A/B/C + 9 single-residue ligands D–L + a second copy M/N/O). No JSON is written →
nan. Structure-specific: 8urf has the *same* 15-chain layout but does not crash.
**Fix:** on failure, retry with an explicit identity map `-c A:A B:B C:C`.

### Bug #3 — `max()` over interfaces reports the heavy–light interface *(fixed in inline path)*
The inline scorer took `max()` over **all** interface DockQs. On a Fab that returns the
**heavy–light (A–B)** interface (~0.90, trivially correct because the antibody folds against
itself) instead of the **antibody–antigen** interface. This produced a **false "8urf rescued
0.90"** — 8urf's real ab–ag DockQ is 0.04–0.13.
**Fix:** select only interfaces involving the antigen chain (see §5 caveat on the
"antigen = C" assumption); fall back to all interfaces for 2-chain (VHH / single-interface)
complexes.

### Bug #4 — RESOLVED: 8f60 and 8b7h nan because OST exits 0 with an *unusable* mapping
The premise "even after the `-c A:A B:B C:C` fallback" was the clue: that fallback was gated on
OST's **exit code**, but 8f60/8b7h make OST **exit 0 and write a JSON** — so `_ost(...) or
_ost(..., -c)` short-circuits on the first success and the `-c` retry never runs. (Only 8ts0, which
exits non-zero, was ever caught.) Two distinct native-side causes, both confirmed from the
`chain_mapping` column and the natives (see `~/Protenix/docs/nan_dockq_interfaces.md`):
- **8f60** — `chain_mapping = {"A":"A","B":"B"}` in 510/510: the auto-mapper **drops the antigen
  chain C** (the native has a real 858-atom C plus a branched N-glycan that appears to confuse the
  mapper), so no A–C/B–C interface exists → nan.
- **8b7h** — `chain_mapping = {"A-2":"A","B-2":"B","C-2":"C"}`: the native is a **2-copy assembly**
  (A/B/C + A-2/B-2/C-2); OST maps the single-copy model onto **copy-2**, so the interface is
  reported as `(A-2,C-2)` while the CSV asks for `(A,C)` → exact-label miss → nan.

**Fix (this session).** The retry now fires on an **unusable result** (requested interface absent),
not just a non-zero exit — in both the batch path (`evaluate_group`) and the inline scorers — and
`extract_dockq` does a unique copy-suffix-stripped match that recovers 8b7h. Cases that are still
genuinely unscorable are **classified** by the new `dockq_status` column rather than silently nan'd
(see Bug #5).

The patched inline `dockq_cif` (this session) — auto map, retry on an *unusable* result, antigen
label from metadata (`ANTIGEN`), forced map from the target's CSV chains (`FORCE_MAP`), robust to
OST exceptions:

```python
def _ost(patched, outj, chain_map=None):
    cmd = [OST_BIN, "compare-structures", "-m", str(patched), "-r", str(NATIVE),
           "-o", str(outj), "--fault-tolerant", "--min-pep-length", "4",
           "--min-nuc-length", "4", "--lddt", "--dockq"]
    if chain_map:
        cmd += ["-c", *chain_map]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=OST_ENV)
    except Exception:                       # TimeoutExpired/OSError -> caller may retry
        return False
    return r.returncode == 0 and outj.exists() and outj.stat().st_size > 0

def _abag(outj):                            # ab–ag DockQ (antigen label from FoldBench metadata)
    d = json.load(open(outj))
    ifaces = list(zip(d.get("dockq_interfaces", []), d.get("dockq", [])))
    abag = [dq for iface, dq in ifaces if ANTIGEN in {c.split("-")[0] for c in iface[:2]}]
    pool = abag if abag else [dq for _, dq in ifaces]
    return max(pool) if pool else None

def dockq_cif(cif):
    try:
        with tempfile.TemporaryDirectory() as td:
            patched, outj = Path(td)/"m.cif", Path(td)/"o.json"
            if not postprocess_cif(Path(cif), patched):
                return float("nan")
            dq = _abag(outj) if _ost(patched, outj) else None
            if dq is None:                  # unusable -> force identity map & re-parse
                dq = _abag(outj) if _ost(patched, outj, FORCE_MAP) else None
            return dq if dq is not None else float("nan")
    except Exception:
        return float("nan")
```

### Bug #5 — `nan` conflated "OST didn't run" with "model didn't dock" *(fixed)*
`extract_dockq` returned `nan` both when OST produced no usable output AND when OST ran fine but the
requested interface was simply **absent** (the antigen didn't dock). The second case is biologically
**DockQ ≈ 0** (a model failure), but recording it as `nan` and excluding it from aggregates
**overstates "fraction of seeds ≥ 0.80."** **Fix:** `extract_dockq` now returns `(dockq, status)` and
the batch CSV gains a `dockq_status` column — `ok` (real score) · `interface_absent`/`chain_unmapped`
(mapped but no dock → **0.0**) · `copy_mismatch`/`ambiguous_copy`/`ost_fail` (genuinely unscorable →
stays **nan**) · `ok_isolated` (recovered by Bug #7's per-pair rescue). Aggregates should count `0.0`
as a failure and exclude only the nan statuses. (8f60 first looked like a genuine never-docks 0.0 —
it was actually an OST mapping artifact; see Bug #7.)

### Bug #6 — doc said the `dockq_interfaces` tuple was model-first *(fixed: see §1)*
§1 originally listed each interface as `[model, model, ref, ref]`. It is `[ref, ref, mdl, mdl]`
(positions 0,1 = reference). The code was already correct; only the doc was wrong. Flagged here so
nobody "fixes" `extract_dockq` to index `iface[2:4]`.

### Bug #7 — OST chem-group mis-mapping silently zeroed a near-perfect dock (8f60) *(fixed)*
8f60's antigen is BTLA, an Ig-superfamily protein. OST's chain mapper groups reference chains by
sequence/chem similarity and lumped the Ig-like antigen (ref C) with an antibody chain (`chem_groups
= [['A'],['B','C']]`), then mapped only `A→A, B→B` and **dropped the antigen** — scoring just the
trivial heavy–light A–B interface (0.918) and reporting antibody–antigen as `chain_unmapped`/0.0.
Forcing `-c …C:C` doesn't help (OST raises `Chem group mismatch in custom mapping`). But the model is
correct — model A/B/C match native A/B/C at **identity 1.00** each, and the true A–C DockQ is
**~0.92**. So this was a pure scoring artifact — the most severe kind in this audit: a top dock
reported as a failure. (Note: this *also* refutes the earlier branched-N-glycan hypothesis; cleaning
the glycan changes nothing.)
**Fix:** when an interface can't be scored by the joint mapper (`status != ok`), `evaluate_group`
retries that pair **in isolation** — subset both structures to just those two chains
(`_subset_cif` / `_isolated_pair_dockq`) so the mapper can't mis-group, then score. Validated
score-preserving on a working target (8urf: isolated == full-complex to 3 dp); fires only on non-`ok`
interfaces → zero regression. Recovered rows get `dockq_status = ok_isolated`.

---

## 4. Corrected scorecard (all on the antibody–antigen interface)

Re-scores used each target's **intended interface from the original CSV's
`chain1`/`chain2` columns** (so the interface choice is grounded in existing data, not
guessed). "baseline" = 50 TTS seeds × 5 samples, best-per-seed; "steered" = repel k6m6 runs.

| target | type / ab–ag iface | baseline (ab–ag) | steered (ab–ag) | verdict |
|---|---|---|---|---|
| **8c5h** | 2-chain, A–B | diverse-seed TTS: max 0.658, 0/51 ≥0.80, median 0.11 | **0.82–0.92** | ✅ steering reliably reaches the good basin; diverse TTS does not (see seed_86 note §6) |
| **8ts0** | Fab, A–C/B–C | **9/50 ≥0.80**, max 0.95 | 5/13 ≥0.88, max 0.94 | solvable, but **baseline already covers it** (was mislabeled a failure by bug #2) |
| **8urf** | Fab, A–C/B–C | 0/50, max 0.16 | **0/15, max 0.13** | ❌ not rescued — earlier 0.90 was bug #3 |
| **8z8v** | VHH, A–B | 0/50, max 0.05 | ~0.01 | ❌ not cracked (inline scores were genuine — 2-chain) |
| **8dtn** | A–B | max 0.405, 0/51 ≥0.80, med 0.13 | — | baseline does not solve |
| **8fxv** | A–B | max 0.371, 0/51 ≥0.80, med 0.26 | — | baseline does not solve |
| **8f60** | A–C/B–C | **0/51, max 0.000** (all `chain_unmapped`) | — | ❌ genuine — antigen C is never mapped by OST (native N-glycan); now a real 0.0, was nan (bug #4). A native pre-clean would confirm model-fail vs OST artifact |
| **8b7h** | A–C/B–C | **32/51 ≥0.80**, max 0.880, med 0.81 | — | ✅ **NOT a failure** — baseline solves it; the nan was the copy-2 label bug (#4), now fixed via the suffix-tolerant match |

**Headline:** 8c5h is currently the only confirmed genuine rescue (baseline-fails →
steering-succeeds). 8ts0 (9/51 ≥0.80) and **8b7h (32/51 ≥0.80)** were never failures — both were
mislabeled by scoring bugs (#2, #4). The genuine baseline-failure set is now **8c5h, 8dtn, 8fxv,
8f60** (and 8f60 is a special "OST can't even score the antigen" case, not a normal low DockQ).

Which FoldBench targets ever had nan/contaminated DockQ (from
`dockq_ost_esmfold2fast_tts_all_samples.csv`): **8b7h, 8c5h, 8dtn, 8f60, 8fxv, 8ts0**
(all-nan or partial-nan). **All re-scored** into `dockq_ost_esmfold2fast_tts_v2_all.csv` (new
`dockq_status` column); across the whole 37,995-row benchmark only 8f60 (510 `chain_unmapped` =
0.0) and one 8b7h row (`copy_mismatch` = nan) are now non-`ok`.

---

## 5. What's verified vs. assumed — scrutinize these

- **Verified by reproducing OST stderr:** bugs #1, #2. **Verified empirically on the same
  structures:** bug #3 (8urf ab–ag 0.04–0.13 while H–L 0.88–0.90).
- **~~Fragile assumption — "antigen = chain C".~~ FIXED.** The inline scorers now read the antigen
  reference label from the FoldBench targets CSV `antigen_chain` column (Fab→`C`, VHH/Nb→`A`) instead
  of hardcoding `C`, and strip a copy suffix when matching it (so a duplicate-copy native reporting
  `C-2` still matches). Decoded-chain ordering = JSON `proteinChain` order (first→`A`…); the antigen
  is `C` only for 3-chain Fabs and `A` for 2-chain VHH/Nb — the metadata lookup handles both.
- **~~Patch scope (inline only).~~ FIXED.** The batch path (`eval_ost_foldbench.py`) now has the same
  protections: `run_ost` retries with an identity map and `evaluate_group` re-runs with a forced map
  on an *unusable* result; `extract_dockq` does the suffix-tolerant match and returns a status (no
  H–L `max()` exposure — it matched on exact CSV labels all along, which §1 now documents correctly).

---

## 6. seed_86 is a knife-edge nondeterministic seed (NOT a contradiction)

8c5h's good basin (DockQ ~0.90) is reachable from baseline folding **only through the
specific seed_86**, and only intermittently: dropout + CUDA nondeterminism tips that fold
into the good basin in **~20–24% of runs** (the rest land bad, DockQ ~0.01). The canonical
0.899 came from the broader many-fold sweep. The 5 CIFs in
`steer/experiments/decoded_8c5h_seed86/` all landed bad (max 0.128, 0/5 ≥0.80) — fully
consistent with a ~24% rate (0-of-5 occurs ~30% of the time), NOT a contradiction.

Implication for the "rescue" framing: **diverse-seed TTS** (50 *different* seeds) tops out
at DockQ 0.658 and never reaches ≥0.80, so it does not find the basin. A single knife-edge
seed (seed_86) finds it ~1-in-5. **Steering reaches 0.82–0.92 reliably.** So steering's value
is reliability — hitting a basin that diverse seeding misses and only a lucky seed finds
intermittently — rather than reaching something otherwise totally unreachable.

---

## 7. Suggested audit checklist

1. ✅ **Resolved (Bug #4).** 8f60 = antigen C dropped by the auto-mapper (branched N-glycan); 8b7h =
   2-copy native mapped to copy-2. Both: OST exits 0 with an unusable mapping, so the exit-code-gated
   `-c` retry never fired. Retry now triggers on an unusable result.
2. ✅ **Done.** Antigen chain is read from the targets-CSV `antigen_chain` column in both inline scorers.
3. ✅ **Done.** Batch path patched: identity-map retry in `run_ost`, forced-map-on-unusable in
   `evaluate_group`, suffix-tolerant match + `dockq_status` classification in `extract_dockq`.
4. ⏳ **Pending recompute.** Re-score the contaminated targets into a fresh `*_v2.csv` (the scorer now
   refuses to append onto the old 12-column schema), then recount genuine model-fails (`0.0`,
   `dockq_status=interface_absent`/`chain_unmapped`) vs scoring-fails (nan: `ost_fail`/`copy_mismatch`).
5. ✅ Resolved — see §6: seed_86 is a ~24% knife-edge seed, not a contradiction.

> **Tuple order (Bug #6):** OST's `dockq_interfaces` is `[ref, ref, mdl, mdl]` — positions 0,1 are the
> reference chains. The code is correct; do not switch `extract_dockq` to `iface[2:4]`. See §1.
