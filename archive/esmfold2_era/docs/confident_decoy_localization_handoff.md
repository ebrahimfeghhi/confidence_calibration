# Handoff: confident decoys, templating, and the localization test

Written 2026-08-07 for a fresh session. Everything below is measured and on disk; file paths are
given so nothing has to be taken on trust. Project root: `/home/ebrahim/confident_decoy_experiments`.
Master index: `/home/ebrahim/confident_decoy_README.md`.

---

## 1. The problem

Protenix (protenix-v2, `N_step 200`, `N_sample 5`) predicts antibody–antigen complexes. For 25
interfaces in our set it confidently selects a **wrong** pose: the highest interface `pair_iptm` pose
has DockQ < 0.23 (below CAPRI-acceptable), even though a correct pose exists somewhere in the model's
own 250-pose pool. We call these **confident decoys**.

By construction, free selection is 0/25. The DockQ oracle (best pose in the same pool) is 25/25. So
the pool contains the answer and selection cannot find it.

## 2. The intervention: templating

We give the model a **template** — one predicted pose, injected as a fake-old mmCIF (release_date
2000-01-01) under a 4-character template id plus a 2-line a3m — and re-run.

**Critical mechanism:** `multichain_mask` (`pairformer.py:1062–1076`) zeroes every *cross-chain*
template feature. A template transmits **only intra-chain backbone conformation**, never the binding
arrangement. So templating cannot hand the model the docking answer.

### Results, all on the same 25 interfaces

| arm | what the template is | rescued |
|---|---|---|
| free (no intervention) | — | **0/25** (by construction) |
| `native50` | the **exact native folds** (oracle) | **17/25** |
| `dockqdonor` | the pool's **best-DockQ pose** (oracle) | **9/25** |
| `leader20` | top-20 ipTM leaders, 5 seeds each, pooled | **3/25** |
| `topiptm26` | the **decoy's own fold** (what ipTM already picked) | **2/25** |
| `minmaxdonor` | pose minimising max(paratope, epitope) CA RMSD | 2/13 (partial) |
| — | DockQ oracle ceiling over the free pool | 25/25 |

Artifacts: `artifacts/{native_arm,dockqdonor,topiptm26,minmaxdonor}_scores.json`,
`artifacts/leader20_scores.json`. Tally with the snippet in §7.

**The spread is the point.** Same mechanism, same compute, wildly different outcomes depending on
*which pose* is the template — 17 vs 9 vs 3 vs 2. Templating on the decoy's own fold (`topiptm26`) is
not merely useless, it is **harmful**: it drives acceptable-pose counts to zero (8slb_AB 65→0,
8the_AB 62→0, 8slb_AC 41→0) and lowers max DockQ by a median −0.222 vs free, while `dockqdonor`
raises it by a median +0.046.

## 3. The current hypothesis (the user's)

Ordinary sampling doesn't learn: each seed is drawn independently, so 250 draws are 250 independent
lottery tickets. **Templating localizes the search around the template's distribution.** With a good
template the search concentrates on a good region, so you sample the right basin densely instead of
occasionally.

Supporting evidence already measured:
- A good template **concentrates**, it doesn't just extend the tail. On `8che_BE`, `dockqdonor`'s
  median pose is DockQ **0.950** and 58% of its 250 poses are above 0.85. Free's median stays ~0.07
  no matter how many poses are drawn.
- A bad template **relocates** the distribution rather than broadening it — the `topiptm26`
  collapses above.

Not yet tested: whether outputs are structurally *near* the template. That is §6.

## 4. Why selection fails, and the 0.6 threshold

Measured conversion — given a pool that **contains** a good pose, how often does top-ipTM pick an
acceptable one (`scripts/why_flat.py`, 10-seed pools, 4000 draws/interface):

| best pose in pool | P(ipTM picks an acceptable pose) |
|---|---|
| 0.23–0.40 | **0.02** |
| 0.40–0.60 | **0.14** |
| 0.60–0.80 | **0.81** |
| 0.80–1.01 | 0.77 |

**ipTM is effectively blind below DockQ ~0.6 and works above it.** So a template's job is not "produce
an acceptable pose" but "produce a pose above ~0.6." All 9 `dockqdonor` rescues have a pool best
≥ 0.656.

Consequence: more sampling doesn't help. Coverage rises (0.433 → 0.920 from 5 to 250 poses) while
conversion falls (0.547 → 0.391), and expected rescues stay flat at ~9/25 for K = 1…50 seeds
(`artifacts/why_flat.json`, `artifacts/dockqdonor_5seed.json`).

## 5. The two big runs

### 5a. `freedeep8che` — DONE. Can free sampling get there at all?

2500 untemplated seeds of `8che` (2349 new + 151 pre-existing) = **11,835 poses**, ~47× the standard
pool. "Untemplated" = the base FoldBench json unchanged, which still carries its natural hmmsearch
template search — identical to the free baseline, **not** templates-disabled. Verified: `Found 4/4/2
templates` on every chain-load.

| | 8che_BE | 8che_AE |
|---|---|---|
| poses | 11,835 | 11,835 |
| best DockQ | **0.630** | 0.630 |
| ≥ 0.23 | 1,514 (13.7%) | 1,484 |
| **≥ 0.6** | **2** (1 in 5,900) | 6 |
| **≥ 0.85** | **0** | **0** |
| top-ipTM selects | **0.064** | 0.115 |

Compare: **one** template (`dockqdonor`'s donor, `seed137/sample3`) produced **147 poses above 0.85 in
250 draws** and selection returned **0.958**.

Also note selection got *worse* with more poses (0.177 at 755 → 0.064 at 11,835), exactly as the
conversion decay predicts.

Artifacts: `artifacts/freedeep8che_scores.json`, `artifacts/free_extra_8che.json`.
Config: `configs/freedeep8che.yaml`. Scorer: `scripts/score_freedeep.py`.

### 5b. `allpose8che10` — RUNNING. The matched-budget test.

Every previous comparison confounded templating with sampling depth. This spends **exactly the same
budget** as `freedeep8che`:

- 250 donor templates (every pose of the free run: seeds 101–150 × samples 0–4)
- **10 seeds each** = 2500 seeds = **12,500 poses** — identical to the free run's 2500 seeds / 12,500 poses
- covers `8che_AE` and `8che_BE` (same predictions, one run scores both)

**Why 10 seeds.** A single seed per donor is unreliable, and we have the receipts:
- `allpose8che` ran all 250 donors at **fixed seed 101** → **0/250** donors produced a >0.6 pose,
  pooled selection 0.079.
- `allpose8cheseed` ran the same 250 donors at **each donor's own seed** → **2/250** above 0.6, pooled
  selection **0.948**.
- Same templates, opposite outcomes. Seed 101 is rank **44/50** for the one donor known to work.
- That donor clears 0.23 on **38/50** seeds, so P(all 10 seeds fail) < 1e-6. Ten seeds makes the
  collapse effectively impossible.

**Seed choice:** each donor gets **its own seed + 9 random** (drawn 101–1100, `rng=0`, reproducible —
rebuild is byte-identical, md5 `d2866d69ae473f6b434c5b04bfb41f23`). Own-seed is included so no donor
is judged purely on foreign seeds; the other 9 are random *per donor* so one unlucky seed cannot
depress every donor at once the way seed 101 did. Own-seed is **not** otherwise special:
`scripts/donor_own_seed.py` measured its rank across 25 interfaces at mean **22.8/50** vs 25.5
expected by chance.

Tradeoff, recorded: donors no longer share a seed set, so donor and seed are not perfectly separated.
`allpose8che` (fixed seed) stays on disk as the perfectly-isolated-but-unlucky comparison.

Config: `configs/allpose8che10.yaml`. Jobs: `artifacts/allpose8che10_jobs.json`.
Score with `scripts/score_allpose.py --run allpose8che10`.

**Expected comparison:** 12,500 free poses (best 0.630, zero above 0.85, selection 0.064) against
12,500 templated poses at identical compute.

## 6. The localization test — TO RUN

**Question:** does templating actually localize the search around the template, as §3 claims?

**Data (all on disk, no GPU, no new generation):** the `allpose8cheseed` arm — 250 templates, one per
free-run pose of `8che`, 5 outputs each = 1250 outputs, each with a known parent template.

**Three numbers per template, all on the same scale:**
1. `donor_vs_native` — how good the template pose itself is
2. `output_vs_native` — how good the poses it produced are (already computed,
   `artifacts/allpose8cheseed_scores.json`)
3. `output_vs_donor` — how close the outputs stayed to their own template ← the localization measure

**Outcomes it distinguishes:**
- outputs cluster *at* the template → (2) ≈ (1), (3) high
- outputs localize but *improve* → (2) > (1), (3) high ← what the hypothesis predicts
- outputs ignore the template → (3) low

### CRITICAL: measure fold similarity, not docking similarity

I ran a first version comparing outputs to templates by **pose–pose DockQ**. Results:
median 0.177; the winning donor's outputs were 0.950 vs native but only **0.245** vs their own
template. **Do not use that as the answer.** DockQ measures the *docking arrangement*, and
`multichain_mask` guarantees the template transmits none of it — so a low value is expected and says
nothing about the hypothesis. It was the wrong measurement.

**The correct test is intra-chain**, since that is all a template carries: per-chain **lDDT or CA
RMSD between each output chain and the corresponding template chain**. Restrict to interface residues
(paratope/epitope) as well as whole-chain, since `scripts/fold_quality.py` already has that
machinery. Prediction under the hypothesis: outputs are tightly localized on the template's **folds**
while free to move in docking space.

The (invalid) pose-space script is `scripts/localization.py` — reuse its donor↔output bookkeeping,
replace the metric. Its output is `artifacts/localization_allpose8cheseed.json`.

### Chain-naming trap — this silently dropped 87% of the data on my first attempt

When BOTH structures are predicted poses, OST reports `dockq_interfaces` in **model** chain naming and
enumerates every pairing. Filtering on **native** names (`{B,E}`) matched a few unrelated pairings that
happened to be lettered B and E, and 218 of 250 donors were silently dropped. Correct route: read the
model chain names for the interface out of the cached model-vs-native comparison, then require both
chain pairs to equal that. See the fixed `model_chains()` in `scripts/localization.py`.

Related and mandatory: `8che` is one of the PDBs where `label_asym_id` ≠ `auth_asym_id`. Use
`native_chains()` / `native_seqs()` from `scripts/decoy_set.py` / `scripts/build_dockq_templates.py`;
never hand-roll a chain lookup, and make it raise rather than return None.

## 7. Reproducing the tallies

```python
import json
D=json.load(open('artifacts/dockqdonor_scores.json'))
T=json.load(open('artifacts/topiptm26_scores.json'))
L=json.load(open('artifacts/leader20_scores.json'))
N=json.load(open('artifacts/native_arm_scores.json'))
S=sorted(set(D)&set(T))                       # the common 25-interface set
print(len(S),
      sum(1 for k in S if (N[k].get('free_sel_dockq') or 0)>=0.23),   # free      0
      sum(1 for k in S if N[k].get('rescued')),                        # native50 17
      sum(1 for k in S if D[k]['rescued']),                            # dockqdonor 9
      sum(1 for k in S if T[k]['rescued']),                            # topiptm26  2
      sum(1 for k in S if k in L and L[k]['method_rescued']))          # leader20   3
```

## 8. Standing rules for whoever picks this up

- **Ask before running anything.** Do not infer approval from silence or from the user asking for a
  status. I violated this and it was not acceptable.
- **Never kill a running job** without an explicit instruction.
- **Preflight every launch**: `scripts/run_health.py --preflight` (exit 1 = do not launch), and attach
  `--watch --interval 1800` for the run's life.
- **Verify the intervention applied.** Grep for `Found N templates`; `Found 0 templates` means the arm
  is void. Three arms were scored and believed for a week while silently untemplated — cause was
  **5-character template ids** (Protenix requires exactly 4).
- **Log the method before launch**, not after results (`configs/*.yaml`).
- Every number must link the script that made it and the on-disk artifact it can be re-tallied from.
- GPUs 0–3 only (4–7 belong to co-workers). CPU 64 soft cap. Check measured usage within 10 s of
  every launch.
- Report measurements; leave interpretation to the user.
