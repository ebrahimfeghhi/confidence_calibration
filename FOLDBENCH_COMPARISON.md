# FoldBench vs. tts_abag: how the evaluation lines up

This note explains how `tts_abag`'s evaluation relates to the upstream
[FoldBench](../FoldBench) framework, why `eval/eval_ost_foldbench.py` matches chains
differently from FoldBench's `eval_by_ost.py`, and — the question that prompted this —
**whether a post-processing step was skipped.**

**TL;DR.** No post-processing step was skipped, and FoldBench's published results are not
corrupted. FoldBench's original matcher is *fragile* rather than buggy: it leans on an
unstated invariant (model chains live in a separate namespace, e.g. `A0,B0,C0`) that our
models break (they emit `A,B,C`). Re-implementing the matcher was the right call. The only
real defect was in our *own* hardening — suffix-stripping — which was harmless for the
antibody–antigen set but wrong for same-letter homo-oligomers; that is now fixed.

---

## 1. The FoldBench pipeline and its (only) two post-processing steps

For each registered model, `FoldBench/run.sh` runs four model-supplied files
(`algorithms/<model>/`): `preprocess.py` → `make_predictions.sh` → `postprocess.py`, then
the shared `evaluate.py` and `task_score_summary.py`.

`postprocess.py` (see `FoldBench/algorithms/Protenix/postprocess.py`) is the entire
"between inference and scoring" surface, and it does exactly **two** things:

1. **Patch each predicted CIF** so OpenStructure can read it: set `occupancy = 1`,
   `B_iso_or_equiv = 0`, and rebuild the `entity` block (polymer / non-polymer from
   `group_PDB`).
2. **Write `prediction_reference.csv`** (`pdb_id, seed, sample, ranking_score,
   prediction_path`) so `evaluate.py` knows what to score.

There is **no chain-relabeling, no assembly expansion, and no stoichiometry/"count"
handling anywhere** in the pipeline. `evaluate.py` then routes to
`evaluation/eval_by_ost.py` (OST: lDDT / DockQ / TM / GDT) and, for nucleic-acid
interfaces, `eval_by_dockqv2.py`; `task_score_summary.py` selects the best sample per
target and computes success rates (interfaces: DockQ ≥ 0.23).

### How `tts_abag` maps onto this

| FoldBench step | `tts_abag` equivalent |
|---|---|
| `preprocess.py` (AF3 JSON → model input, sets `count`) | model-specific input building in `run/esm/*`, `run/protenix/*` (the `count` field expands to consecutive chain letters) |
| `make_predictions.sh` | `run/esm/run_foldbench_abag*.py`, `run/protenix/run_benchmark_parallel.py` |
| `postprocess.py` step 1 (CIF patch) | `postprocess_cif` / `_postprocess_cif_inproc` in `eval/eval_ost_foldbench.py` — **identical** occupancy/B-factor/entity patch |
| `postprocess.py` step 2 (`prediction_reference.csv`) | replaced by directory scanning in `load_interfaces` + `main` |
| `eval_by_ost.py` | `eval/eval_ost_foldbench.py` (one OST run per structure/seed/sample, one row per interface) |
| `task_score_summary.py` (best sample, DockQ ≥ 0.23) | `plot/plot_scaling_ost.py` (bootstrap over seeds, DockQ ≥ 0.23 kept) |

**Conclusion for the "did I forget a step?" worry:** the only post-processing FoldBench
does is the CIF patch (reproduced verbatim) and a reference CSV (replaced). Nothing was
skipped.

---

## 2. What "count == 2" actually is, and where the `-2` lives

A `count: 2` entity means two identical copies of a chain appear in the assembly. FoldBench
encodes the second copy with a `-2` suffix **in two places that already agree**:

- the **ground-truth CIF** auth chain name — e.g. `FoldBench/examples/ground_truths/8im0-assembly1.cif`
  has chains literally named `A` and `B-2`;
- the **targets CSV** — `interface_antibody_antigen.csv` row
  `8im0-assembly1,A,B-2,protein,protein,B-2,A`.

OpenStructure **preserves** that exact label. Proof from our own run
(`Protenix/foldbench_abag/Protenix_dockq.csv`, 8im0 rows):

```
chain_mapping = {"A": "A", "B-2": "B"}   # ref "B-2" → model "B"
dockq         = 0.013
```

So OST's `dockq_interfaces` for 8im0 is `[["A", "B-2", "A", "B"]]` (the first two slots are
reference chains, the last two are model chains). The `-2` is real and carried all the way
through scoring — there is nothing to "expand."

`8im0` is the **only** `-2` row in the antibody–antigen target set.

---

## 3. Why our matcher differs from FoldBench's — and why theirs is *fragile*, not buggy

FoldBench, `evaluation/eval_by_ost.py:54-55`:

```python
for i, interface in enumerate(data['dockq_interfaces']):
    if native_chain_id_1 in interface and native_chain_id_2 in interface:
        dockq = data['dockq'][i]   # note: no break -> last match wins
```

`interface` is the 4-element list `[ref1, ref2, model1, model2]`, and the check tests
membership across **all four** slots (reference *and* model chains). This is only safe
because **FoldBench's own model outputs put model chains in a separate namespace**: in their
example, `chain_mapping = {'A':'A0','B':'B0','C':'C0'}` and interfaces look like
`['A','B','A0','B0']`. Because `A0 ≠ A`, a model chain can never be mistaken for a reference
chain, so the loose test is fine *within FoldBench's ecosystem* — their numbers are correct.

**Our models break that invariant.** ESMFold2-Fast and Protenix (as we run them) name model
chains `A, B, C` — the **same namespace as the reference chains** (the `{"A":"A","B-2":"B"}`
mapping above shows model chains `A,B`, not `A0,B0`). Once the namespaces collide, the loose
`x in interface` test can latch onto a *model* chain in the wrong interface (e.g. it would
find target chain `B` inside `['A','C','A','B']` via the model slot and wrongly score the A–C
interface). That risk is highest for symmetric / `count==2` complexes, where OST permutes
equivalent chains — which is exactly the "doesn't work for count of 2" symptom.

So re-implementing the matcher to compare **only the reference pair** (`iface[0], iface[1]`),
order-independently, was the correct fix.

**One honest caveat:** for the single `count==2` Ab–Ag structure (8im0 = `A`/`B-2`), the
original would *still* have returned the right DockQ (`0.013`) — there is only one interface
and the suffix is preserved, so there is nothing to mis-match. The change was sound
defensive engineering given our chain naming, not a patch over a missing step.

---

## 4. Other similarities and deliberate differences

- **One OST run per structure** (not per interface): FoldBench dedups via the cached OST
  JSON path; we run once per `(structure, seed, sample)` and emit one row per interface.
  Equivalent, ours is a bit cleaner.
- **Confidence scores are ours, added.** FoldBench's OST eval reports DockQ/lDDT/TM/GDT and
  takes `ranking_score` from the model's confidence JSON. We additionally compute
  `pair_iptm`, `pair_ranking_score`, `global_iptm`, `global_ranking_score` from
  `chain_pair_iptm` so any test-time-scaling **selection strategy** can be replayed without
  re-running OST. This is an extension, not a deviation.
- **Aggregation differs by design.** FoldBench's `task_score_summary.py` keeps the best
  sample per target. We save **all** samples and bootstrap over seeds in
  `plot/plot_scaling_ost.py` — because the whole point here is the TTS scaling curve. The
  **success threshold (DockQ ≥ 0.23) is kept identical.**
- **We drop whole-structure metrics** (lDDT/TM/GDT) since the Ab–Ag metric is interface
  DockQ. Re-add them in `extract_dockq`'s caller if a monomer task is ever evaluated.

---

## 5. Known limitation (now fixed): same-letter homo-oligomers

Our original hardening stripped the copy suffix (`iface[0].split("-")[0]`). That was a no-op
for 8im0 (`A` vs `B-2` → `A` vs `B`, still distinct) but **wrong** for *same-letter*
homo-oligomers:

- **`extract_dockq`** — a homo-trimer `A, A-2, A-3` has interfaces `(A,A-2)`, `(A,A-3)`,
  `(A-2,A-3)`; stripping collapses all three to `(A,A)`, so the first match was returned for
  every target row.
- **`compute_confidence_scores`** — `{k.split("-")[0]: v}` collapsed `{"A":"A","A-2":"B"}` to
  `{"A":"B"}`, so both chains resolved to model chain `B` and `pair_iptm` read the diagonal
  `chain_pair_iptm[1][1]` (chain B self-iptm) instead of the A–B pair. This bit even at
  `count==2` for same-letter dimers.

**Fix (applied):** match on the **full, exact** chain labels (`A`, `B-2`, `A-2`), which OST,
the targets CSV, and `chain_mapping` already agree on. This is simpler than stripping, is
copy-aware, and is verified to leave 8im0 unchanged (DockQ `0.013`) while scoring synthetic
homo-trimer/homodimer cases correctly. The Ab–Ag results in this repo were never affected
(8im0 is the only multi-copy case and uses different base letters), but the eval is now
reusable on protein–protein homo-oligomers (e.g. the `interface_protein_protein.csv`
`A,A-2` rows).
