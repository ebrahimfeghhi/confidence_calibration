# Confident-Decoy Rescue — Run Log

Durable, append-only registry of every generation/scoring run. Newest first.
Companion to `confident_decoy_FINDINGS.md` (interpretation) — this file is the raw provenance.

## Schema (fill every field per run)

```
### <run_id>            e.g. abonly_decoys_2026-07-27
- date:                YYYY-MM-DD
- status:              planned | running | done | aborted
- purpose/hypothesis:  what question this run answers; predicted outcome
- set:                 decoys (15, meta.json) | goods (15, safety15.json) | both
- chains templated:    which chains get a template (antibody / antigen / both / none)
- fold source:         native | predicted-isolation | predicted-freerun | none
- fixed vs varying:    fixed (one template all seeds) | varying (per-seed) | n/a
- selection of template: how the template fold was chosen (best-pLDDT / sample_0 / exact-native / ...)
- seeds / samples:     e.g. 50 seeds (0-49) x 5 samples = 250 poses/interface
- inference flags:     N_step, use_template, model, checkpoint
- GPUs:                which GPUs, worker layout
- builder:             script that made the template JSONs
- template JSONs:      dir
- output dir:          where predictions land
- scorer:              script + selection rule (e.g. top interface-pair_iptm pose's DockQ>=0.23)
- result:              rescued/preserved count /15 + per-interface list
- notes / caveats:     confounds, failures, resource incidents, deviations
```

## Rescue-count convention
- **Rescued (decoys):** selection (top interface-pair_iptm pose) has DockQ >= 0.23.
- **Preserved (goods):** selection still DockQ >= 0.23 (safety — must not break good cases).
- Always re-tally from the score JSON; never trust a remembered number.

## Reference results (baselines, already established)
| run | set | chains templated | fold | result |
|---|---|---|---|---|
| free | decoys | none | none | 0/15 rescued |
| native50 | decoys | both (Ab+Ag) | native | 14/15 |
| safety (native goods) | goods | both | native | 15/15 preserved |
| agonly | decoys | antigen only | native | 8/15 |
| expb | decoys | both | predicted-isolation, fixed | 2/15 |
| expc | decoys | both | predicted-freerun, varying (sample_0) | 2/15 |
| expd | decoys | both | predicted-freerun, fixed best-pLDDT | 0/15 |

---

### fullmsa_decoys_2026-07-31   (no MSA row subsampling; + seed-matched rescoring of the free baseline)
- date:                2026-07-29 → 2026-07-31
- status:              DONE (all 12 pdbs / 15 interfaces at N=50)
- purpose/hypothesis:  does removing per-seed MSA random row subsampling (§6.9) fix the confident decoys?
- set:                 decoys (15, meta.json)
- chains templated:    NONE (free run) — only difference from `free` is MSA depth
- seeds / samples:     seeds 101-150 x 5 samples = 250 poses/interface
- inference flags:     `--data.msa.min_size.test 16384` (forces sample_size = n, no subsampling); otherwise identical to free
- GPUs:                0-3 (global work-queue scheduler, no idle GPUs)
- scheduler:           scripts/fullmsa_queue_sched.py
- output dir:          confident_decoy_experiments/artifacts/fullmsa_out/
- scorer:              scripts/fullmsa_n50_incr.py (incremental, per-pdb) -> artifacts/fullmsa_n50_report.json
- comparison:          scripts/fullmsa_seedmatch_table.py -> artifacts/fullmsa_seedmatch_table.json
- result:              **3/15** rescued {7z2m_AJ, 8aok_AB, 8che_BE}; seed-matched free baseline **1/15** {8ux6_DF} => **net +2**
- **FULL WRITE-UP + per-interface table + column definitions: README §2e** (not duplicated here)
- notes:               (a) SEED-RANGE TRAP — the canonical free CSV mixes seeds 0-49 (8fbw/8q3j/8ux6/8yxi)
                       and 101-150 (rest); filtering it to 101-150 silently drops those 4 pdbs. Fixed by
                       scripts/score_free_101_150.py -> artifacts/free_101_150_scores.json. Consequence:
                       "free = 0/15" is seed-range dependent; at 101-150 free is 1/15.
                       (b) Interpretation is WEAK: median dgap +0.004, gap better in only 7/15; fold quality
                       unchanged (median d-lDDT 0.000); full MSA LOSES 8ux6_DF. Reshuffling, not recalibration.
                       (c) Watcher scripts/wait_8bf4.sh — harness background tasks kept getting killed; run
                       long waits with `setsid nohup ... & disown` (PPID 1) or a persistent Monitor.

---

### relabel_fix_2026-07-30   (agonly / abonly / pddag re-run for 8yxi + 8fbw, CORRECTED antigen labels)
- date:                2026-07-30 → reported 2026-08-01
- status:              **DONE, 9/9 targets scored (250 poses each)**
- purpose/hypothesis:  the label bug (README 2) meant agonly/abonly/pddag templated the WRONG chain for
                       8yxi (q0 heavy instead of q2 antigen) and 8fbw (q0 heavy wrongly included).
                       KEY QUESTION: agonly previously "rescued" 8yxi_AC, 8yxi_BC, 8fbw_AC while
                       templating the HEAVY chain. With the antigen correctly templated, do they still
                       rescue (=> genuinely antigen-driven) or flip to not-rescued (=> they were
                       ANTIBODY-driven, which revises Conclusion 1's antigen-8/antibody-5 disjoint split)?
- set:                 decoys, 2 PDBs / 3 interfaces (8yxi_AC, 8yxi_BC, 8fbw_AC) x 3 runs = 9 targets
- chains templated:    agonly = antigen only (native fold); abonly = antibody only (native fold);
                       pddag  = antigen only (best-pLDDT PREDICTED fold). Non-templated chains freed.
- fold source:         native (agonly/abonly) | predicted-freerun best-pLDDT (pddag)
- selection of template: corrected artifacts/agonly_antigen_idx.json (8yxi->[2], 8fbw->[2]);
                       8yxi q2 pLDDT pick computed inline = seed_146/sample_2 (pLDDT 0.860)
- seeds / samples:     50 seeds (101-150) x 5 = 250 poses (seed-matched to all other runs)
- inference flags:     N_step 200, N_sample 5, use_template true, protenix-v2, ckpt /home/ebrahim/checkpoint
- GPUs:                0-3, one job per GPU, dynamic refill (6 jobs)
- builder:             scripts/build_relabel_fix.py (native a3m copied to artifacts/native_a3m/ for
                       durability; new pddag tids o960+); scheduler scripts/relabel_sched.py
- template JSONs:      artifacts/relabel_tmpl/{pdb}-{run}.json
- output dir:          artifacts/relabel_out/{pdb}_{run}/seed_101..150/
- scorer:              scripts/relabel_incr_score.py (incremental, same OST flags + top-pair_iptm
                       DockQ>=0.23 rule) -> artifacts/relabel_scores.json + _summary.json
- result:              **ANSWER: they were ANTIBODY-driven.** Both 8yxi interfaces FLIP to not-rescued
                       under agonly (DockQ 0.015 / 0.029) and are RESCUED by abonly (0.789 / 0.939).
                       8fbw_AC is rescued by BOTH (agonly 0.764, abonly 0.586). pddag still 0/3
                       (0.148 / 0.072 / 0.173) => pddag stays 1/15 with fully correct labels.
                       **Set-level: agonly 8->6, abonly 5->8; abonly∩agonly = {8fbw_AC}, so the
                       "disjoint" claim in Conclusion 1 is FALSE and the antigen/antibody split reverses.**
- **FULL WRITE-UP + per-target table: README §2f** (not duplicated here)
- notes / caveats:     other 10 PDBs provably unaffected (labels verified unchanged), so the full-set
                       tallies get updated by substituting these 3 interfaces only.
                       Resource @launch: ~4 cores, 7.6 GB, GPUs 0-3.

### analysis_runs_2026-07-29_30  (no GPU; scoring/analysis only)
- date:                2026-07-29 .. 2026-07-30
- status:              done
- purpose:             characterise WHERE the bottleneck is (selection vs generation) and test the
                       holo/apo leakage hypothesis. All restricted to canonical seeds 101-150.
- runs + scripts + data (all under confident_decoy_experiments/):
   1. ANTIGEN LOCALIZATION — scripts/localize_antigen.py -> artifacts/antigen_localization/
      Q: is the good antigen fold a global structural outlier (=> leaders/NMS could find it)?
      A: NO for domains (oracle vs pLDDT-pick 0.90-0.98 lDDT = same cluster, gap <=0.06);
         YES-ish for peptides (~0.5 apart, gaps 0.19-0.38). README 2b.
   2. pddag — see its own entry below. 1/15.
   3. H3 IN-POOL (lDDT) — scripts/h3_inpool.py -> artifacts/h3_inpool/h3_inpool_results.json
      Q: does the model ever SAMPLE a native CDR-H3? A: short H3 (<=13) yes and pLDDT picks it;
         long H3 (>=14) mostly NO (best-of-250 only 0.56-0.77) => GENERATION-limited. Only 7z2m is
         selection-limited. H3-region pLDDT vs global pLDDT: better 4/13, worse 6/13 (mean -0.003)
         => not a general selector. README 2c.
   4. H3 RMSD (framework-aligned) — scripts/h3_rmsd.py -> artifacts/h3_inpool/h3_rmsd_results.json
      Same partition as lDDT (robust), BUT lDDT UNDERSTATES severity: 8yxi/8the best-of-250 H3 is
      5.0 A (wrong loop) despite lDDT 0.56/0.68; 8slb's pLDDT pick is 10.9 A at lDDT 0.61.
      Adopted defs: good H3 <=1.5 A; generation failure = best-of-250 never <2 A. README 2c.
   5. HOLO/APO — FoldBench cross-match + scripts/binding_dependence.py -> artifacts/binding_dependence/
      Q (user): are native templates helping by accuracy or by leaking BOUND-state geometry?
      A: splits by type. Domain antigens binddep ~0 (and experimental 8aok/7yds same-antigen-different-
         antibody = 1.06 A global / 0.48 A median) => accuracy, little to leak. PEPTIDE antigens
         binddep +0.19..+0.33 at the epitope => holo leak is real and UNFIXABLE (no apo state exists);
         5 of oracle's 8 rescues are peptides. NEW: antibody PARATOPE is binding-dependent even when the
         chain globally is not (8bf4 +0.157, 8slb +0.123, 8the +0.085 at paratope vs +0.066/+0.041/+0.027
         global) => native ANTIBODY templates leak holo geometry too, localized to the paratope.
         Also: interface lDDT is systematically below global (antibody -0.054, antigen -0.022).
         README 2d. STATUS: hypothesis-level; true apo control needs external PDB entries.
- notes / caveats:     Two methodology bugs found+fixed, see README 6.7: (a) multi-copy assemblies —
                       stored chain maps mix Fab/antigen copies (7z2m gave a heavy chain 0 paratope
                       residues); fixed by contact-count copy resolution. (b) partially-ordered peptides
                       (8che 37-mer construct, 15 residues modelled) need containment matching.
                       ALSO FLAGGED (unresolved): FoldBench's 7z2m_AJ interface (A-J) is a 3-residue
                       crystal contact; real units are A/B-K and E/F-J -> 7z2m results suspect.

### pddag_decoys_2026-07-29
- date:                2026-07-29 (completed 2026-07-30)
- status:              done
- purpose/hypothesis:  best-pLDDT ANTIGEN fold as the ONLY template (antibody freed) — deployable
                       analog of agonly (native antigen -> 8/15). Does a pLDDT-SELECTED (non-native)
                       antigen template rescue any decoy? Prediction: ~0 (localization showed pLDDT-pick
                       antigen ≈ oracle ≈ 0.88-0.92 lDDT, not exact; only exact-native crashes decoy ipTM,
                       cf. Exp D=0/15). Cleaner than Exp D: isolates antigen side (Exp D templated both).
- set:                 decoys (15 interfaces, 12 PDBs)
- chains templated:    antigen only (antibody templatesPath=None, == free-run state)
- fold source:         predicted-freerun (best mean-pLDDT antigen chain over seeds 101-150)
- fixed vs varying:    fixed (one antigen template all seeds)
- selection of template: highest mean chain-pLDDT antigen fold (from antigen_localization/localization_results.json)
- seeds / samples:     50 seeds (101-150) x 5 = 250 poses/interface (seed-matched to free/agonly/native50)
- inference flags:     N_step 200, N_sample 5, use_template true, protenix-v2, ckpt /home/ebrahim/checkpoint
- GPUs:                0-3 (grabs 0 after 7z2m expand frees it), one PDB per GPU, dynamic refill
- builder:             scripts/build_pddag.py (a3m+mmCIF tids o940+); scheduler scripts/pddag_sched.py
- template JSONs:      artifacts/pddag_tmpl/{pdb}-pddag.json ; a3m artifacts/pddag_a3m/
- output dir:          artifacts/pddag_out/{pdb}_pddag/seed_101..150/
- scorer:              scripts/pddag_incr_score.py (incremental; OST --dockq + chain_pair_iptm,
                       rule = top interface pair_iptm pose's DockQ>=0.23). Scores
                       artifacts/pddag_scores.json + _summary.json
- result:              **1/15 rescued = {8ux6_DF}** (DockQ 0.253, marginal). Independently re-tallied
                       from raw per-pose scores: 1/15 confirmed. Near-miss 8fas_BC 0.206. All others
                       <=0.17. Clean subset (excluding the 3 label-bug interfaces) = 1/12.
                       vs anchors: free 0/15, agonly (NATIVE antigen) 8/15, ExpD 0/15.
                       CONCLUSION: a pLDDT-selected (non-exact) antigen template does NOT rescue;
                       consistent with 2b (domain antigens have no selection headroom) and with the
                       raise-vs-crash mechanism (high pair_iptm + low DockQ persists: 8fas 0.90->0.21,
                       8bf4 0.83->0.01)
- notes / caveats:     antigen pLDDT-pick drawn from seeds 101-150 for ALL pdbs (incl. regen pdbs whose
                       canonical free pool is 0-49) — internally consistent for this run. Logs
                       artifacts/pddag_logs/. Resource @launch: ~4 cores, 24GB RAM, GPUs 1-3 (0=expand).
                       *** LABEL BUG (found 2026-07-29): agonly_antigen_idx.json mislabels the antigen for
                       8yxi (q0=heavy, not antigen; real antigen q2) and 8fbw (q0 heavy wrongly included).
                       So pddag 8yxi_AC/8yxi_BC and 8fbw_AC templated the WRONG chain -> RE-RUN those 2 pdbs
                       with corrected labels. Same bug affects the ORIGINAL agonly & abonly runs for 8yxi/8fbw:
                       agonly's 8yxi_AC/8yxi_BC/8fbw_AC rescues were heavy-templated -> may be antibody-driven,
                       not antigen (could revise Conclusion 1 disjoint-set decomposition). Truth = kind()
                       classifier in scripts/derive_antigen_idx.py. Details + asterisks in README §2.
                       Other 10 pdbs unaffected; oracle/native50/free/ExpD unaffected (template all/no chains).

### expand_seeds_freepool_2026-07-28
- date:                2026-07-28 (completed 2026-07-30 13:39)
- status:              done
- purpose/hypothesis:  expand the free-run pose pool from 50 -> 150 seeds for the 8 decoy PDBs
                       stuck at 50 seeds, to enlarge the sampling pool (deeper coverage / better
                       selection headroom). No template rescue — pure free-run expansion.
- set:                 decoys — 8 of 12 PDBs (those at 50 seeds): 7z2m,8aok,8av2,8bf4,8che,8fas,8slb,8the
                       (covers interfaces 7z2m_AJ,8aok_AB,8av2_AC,8bf4_AB,8che_AE,8che_BE,8fas_BC,
                       8slb_AB,8slb_AC,8the_AB). Excludes 8fbw/8q3j/8ux6/8yxi (already >50 seeds; 0-100 would collide).
- chains templated:    none (free run, standard PDB templates via --use_template true, no native injection)
- fold source:         none
- fixed vs varying:    n/a
- selection of template: n/a
- seeds / samples:     +101 new seeds (0-100) x 5 samples = 505 new poses/interface; total -> 151 seeds
- inference flags:     N_step 200, N_sample 5, use_template true, model protenix-v2, ckpt /home/ebrahim/checkpoint
- GPUs:                0-3, one PDB per GPU, dynamic refill (2 waves of 4)
- builder:             scripts/expand_seeds_sched.py
- template JSONs:      n/a (inputs: Protenix/foldbench_abag/jsons/{pdb}-final-updated.json)
- output dir:          Protenix/foldbench_abag/predictions_template/{pdb}/seed_0..100/
- scorer:              not yet scored (DockQ+ipTM via eval_ost_foldbench.py when needed)
- result:              GENERATION COMPLETE — all 8 PDBs now at 151 seeds (was 50). Pool expanded;
                       not yet scored (no analysis has needed >50 seeds yet). NOTE: all analyses to
                       date deliberately restrict to the canonical 50 seeds (101-150) for seed-matching.
- notes / caveats:     seeds 0-100 = 101 seeds (1 over the 150 target -> 151). Logs in
                       artifacts/expand_seeds_logs/. Resource check @launch: ~4 cores, ~9GB RAM,
                       GPUs 0-3 only (5=john,6-7=shawn untouched). GPU compute confirmed engaged 18:26.


## Runs (newest first)

### topiptm_decoys_2026-07-28
- date: 2026-07-28
- status: running (GPUs 4-5, parallel with sposefv on 0-3)
- purpose/hypothesis: template with the CONFIDENT DECOY's OWN fold — the single pose (over seeds 101-150) with the highest antibody-antigen interface ipTM, all chains from it, fixed. Tests whether the decoy's confidence is fold-carried (A: level the field → decoy edge shrinks) or dock-carried (B: reinforce decoy → worse). Was the original Exp D design (top-ipTM), never run.
- set: decoys (15); all 12 top-ipTM poses are confident decoys (DockQ<0.23)
- chains templated: both; single top-ab-ag-ipTM pose; fixed
- fold source: predicted-freerun, selected by max interface pair_iptm over 250 poses
- seeds/samples: 101-150 × 5; GPUs 4,5 (7z2m split 2-way)
- builder: topiptm_picks.json (from summary_confidence chain_pair_iptm + ost_oracle chain maps) → build_topiptm.py; runner topiptm_run.py
- config: confident_decoy_experiments/configs/topiptm_decoys.yaml
- output: topiptm_out/; scorer score_generic.py → topiptm_scores.json
- result: PENDING (vs oracle 8/15, expd 0/15, free 0/15)
- notes: 7z2m on only 2 GPUs is slow (~5h), so full result is long-poled; the 11 other pdbs finish sooner and can be scored incrementally.

### sposefv_decoys_2026-07-28
- date: 2026-07-28
- status: running
- purpose/hypothesis: ablation of oracle_decoys — force each Fab's H+L from the SAME free-run pose (self-consistent VH-VL pairing) vs oracle's mix-and-match (H,L from different poses, 9/9 Fabs). Tests whether H-L pairing consistency matters for rescue. Fold-level pre-check: same-pose paratope ~equal/slightly worse (mean Δ −0.006), so predict ~8/15.
- set: decoys (15)
- chains templated: both; Fab H+L from joint-best pose, VHH antibody + antigen unchanged from oracle
- fold source: predicted-freerun, best-lDDT-to-native (oracle), Fab-joint-pose constraint
- seeds/samples: 101-150 × 5; GPUs 0-3 (7z2m auto-split 4-way)
- builder: build_sposefv_folds.py + build_sposefv_tmpl.py; runner sposefv_run.py
- config: confident_decoy_experiments/configs/sposefv_decoys.yaml
- output: sposefv_out/; scorer score_generic.py → sposefv_scores.json
- result: PENDING (vs oracle 8/15)
- notes: only the 9 Fab antibody chains differ from oracle_decoys; 3 VHH pdbs identical to oracle. Launched after killing the goods orchestrator (oracle_goods had failed on a 4-char chain-name PDB-write bug in prep_oracle_goods.py — separate fix pending; abonly_goods already gave 15/15 safety).

### abonly_goods_2026-07-27
- date: 2026-07-27
- status: planned (runs AFTER abonly_decoys)
- purpose/hypothesis: safety check — native antibody-only template must PRESERVE the 15 good cases (expect ~15/15). Mirror of agonly good-set safety.
- set: goods (15, safety15.json)
- chains templated: antibody only (native); antigen FREE (MSA only, templatesPath removed)
- fold source: native (ground truth)
- fixed vs varying: fixed (exact native, one per chain)
- selection of template: exact native chain fold
- seeds / samples: 50 seeds (0-49) x 5 samples = 250 poses/interface
- inference flags: N_step 200, use_template true, protenix-v2, /home/ebrahim/checkpoint
- GPUs: 0-3
- builder: build_abonly.py (good branch; reuses safety_tmpl native templates, drops antigen templatesPath)
- template JSONs: abonly_tmpl/
- output dir: abonly_out/
- scorer: score_abonly.py (top interface-pair_iptm DockQ)
- result: PENDING
- notes: antigen idx for goods derived via J-motif classifier, validated against agonly_antigen_idx on decoys.

### abonly_decoys_2026-07-27
- date: 2026-07-27
- status: running
- purpose/hypothesis: test whether the native ANTIBODY-only template rescues confident decoys. Paratope/CDR-H3 hypothesis predicts ~14/15 (antibody side carries the corrective signal); antigen-only got 8/15, so if abonly >> 8 the signal is antibody-side, if ~8 or less it is antigen/epitope-side or needs both.
- set: decoys (15, meta.json)
- chains templated: antibody only (native H+L or VHH); antigen FREE (MSA only, templatesPath removed)
- fold source: native (ground truth)
- fixed vs varying: fixed (exact native, one per chain)
- selection of template: exact native chain fold
- seeds / samples: 50 seeds (0-49) x 5 samples = 250 poses/interface
- inference flags: N_step 200, use_template true, protenix-v2, /home/ebrahim/checkpoint
- GPUs: 0-3 (ESM paused for this)
- builder: build_abonly.py (decoy branch; reuses tmpl/all native templates, drops antigen templatesPath via agonly_antigen_idx.json)
- template JSONs: abonly_tmpl/
- output dir: abonly_out/
- scorer: score_abonly.py (top interface-pair_iptm DockQ >= 0.23)
- result: PENDING
- notes: ESM paused (workers killed) to free GPUs 0-3; user restricted to GPUs 0-3 only.
