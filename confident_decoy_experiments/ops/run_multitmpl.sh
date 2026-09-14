#!/usr/bin/env bash
# Multi-donor self-templating at N_cycle=1: can the fold come from templates so the recycles don't
# have to re-derive it?
#
# HYPOTHESIS. Within an interface, the folds and interface patches are the same in every pose --
# per-chain lDDT SD 0.007 (antibody) / 0.014 (antigen), patch RMSD SD 0.10-0.18 A -- while epitope
# Jaccard moves with SD 0.13. Poses differ in PLACEMENT, not folding. So paying 10 recycles to
# re-derive a fold we already have may be waste: hand the fold over as templates, run ONE recycle,
# and let the seed still choose the epitope.
#
# THE CROSS-CHAIN MASK IS LEFT ON (PROTENIX_MULTICHAIN_MASK is deliberately NOT set). multichain_mask
# zeroes every cross-chain template feature, so the donors transmit only intra-chain backbone and
# CANNOT tell the model where the antigen sat. Giving away the fold while keeping the docking decision
# free is the entire point. Previous self-templating arms here set the mask to 0 to pass the dock
# through -- that is the opposite experiment, and it lost to the free run (0.587 -> 0.511, n=35).
#
# DONORS are seed 101's five samples, which for 7txw are all wrong-surface (DockQ 0.024-0.029,
# epitope Jaccard 0.00-0.17). Correctly folded, mis-docked -- exactly the intended input.
#
# BASELINE is the existing free run on the same seeds (102-150, N_cycle=10, default templates), which
# is already scored, so no baseline compute is needed.
#
# CONFOUND, STATED: this arm changes TWO things at once (N_cycle 10 -> 1, and default templates ->
# self templates). A clean attribution needs a third arm at N_cycle=1 with default templates. That is
# deliberately not run yet -- this is the cheap look first.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
RUN=${RUN:-mt7txw}
PDB=${PDB:-7txw}
GPU=${GPU:-2}
NCYCLE=${NCYCLE:-1}
OUT=$REPO/artifacts/sweep_out/$RUN
# JSON defaults to the self-templated build; point it at the stock FoldBench json to run the CONTROL
# arm (default hmmsearch templates, everything else identical) which separates "1 recycle is not
# enough" from "the self-templates pinned it".
JSON=${JSON:-$REPO/artifacts/selftmpl_tmpl/$RUN/${PDB}_${RUN}.json}
# SEEDS overridable so one arm can be split across GPUs; each worker takes a disjoint
# subset and they share one output dir (different seed_<n> dirs, no collision).
SEEDS=${SEEDS:-$(seq 102 150 | tr '\n' ',' | sed 's/,$//')}
LOG=$REPO/artifacts/${RUN}.log

[ -s "$JSON" ] || { echo "missing $JSON -- run scripts/build_multitmpl.py first"; exit 1; }
mkdir -p "$OUT"

# TWO SEQUENCE-KEYED DEDUP STAGES stand between 5 donor poses and 5 actual templates, and both fail
# SILENTLY -- the run logs "Found 1 templates" and produces perfectly plausible poses from a single
# donor.
#   1. get_templates() de-duplicates the a3m hits by hit_sequence (template_utils.py:1002). Handled in
#      build_multitmpl.py by offsetting the mutated positions per donor.
#   2. it then de-duplicates the PROCESSED features by the sequence read from the mmCIF
#      (template_utils.py:1047), and five conformations of one molecule share that sequence exactly.
#      PROTENIX_TMPL_NODEDUP=1 makes that key unique per hit.
# PROTENIX_MAX_TEMPLATES caps how many are collected (default 4, and it must be >= the donor count).
cd /home/ebrahim/Protenix || exit 1
echo "### $RUN  gpu$GPU  N_cycle=$NCYCLE  seeds 102-150  $(date +%H:%M)" | tee -a "$LOG"
# The two dedup workarounds apply ONLY to multi-donor self-templating. The control must run stock,
# so they are set only when NTMPL is given.
if [ -n "${NTMPL:-}" ]; then
  export PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=$NTMPL
fi
# MASK=0 turns OFF multichain_mask, so templates transmit INTER-chain geometry -- i.e. the donor's
# actual dock -- instead of intra-chain conformation only. Every arm above ran with the mask ON (the
# default), which is why they could not transmit placement even in principle. With good donors this is
# a positive control: if self-templating cannot reproduce a correct dock even when handed it, the mask
# was never the limiting factor.
# NOMSA=1 pins the MSA to the query row only (PROTENIX_MSA_NROWS=1 + strategy=topk), i.e. no
# co-evolutionary evidence. Doubles as a test of the seed mechanism: the per-seed MSA SUBSAMPLE
# (sample_indices draws a random size AND subset) is what we traced ipTM's between-seed variance to,
# and it reaches the confidence head only through z_trunk. With the depth pinned there is nothing
# left to differ between seeds except diffusion noise, so between-seed ipTM variance should collapse.
if [ "${NOMSA:-0}" = "1" ]; then
  export PROTENIX_MSA_NROWS=1 PROTENIX_MSA_STRATEGY=topk
  echo "### NOMSA: PROTENIX_MSA_NROWS=1 strategy=topk (query sequence only)" | tee -a "$LOG"
fi
if [ "${MASK:-1}" = "0" ]; then
  export PROTENIX_MULTICHAIN_MASK=0
  # KEEP_FRAME stops template_utils.py centering EACH CHAIN on its own centroid, which would
  # superimpose the template's chains and hand the model a complex with no inter-chain geometry.
  # Only meaningful together with MASK=0 and a single donor covering every query chain.
  [ "${KEEPFRAME:-1}" = "1" ] && export PROTENIX_TMPL_KEEP_FRAME=1
  echo "### PROTENIX_MULTICHAIN_MASK=0 -- templates carry inter-chain geometry" | tee -a "$LOG"
fi
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$JSON" \
  --dump_dir "$OUT" \
  --seeds "$SEEDS" \
  --model.N_cycle "$NCYCLE" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true >> "$LOG" 2>&1
rc=$?

# THE SILENT FAILURE: a malformed template id makes Protenix log 'Found 0 templates' and run
# UNTEMPLATED while still emitting perfectly plausible poses. Three arms were scored and believed for
# a week in that state, so this is checked rather than trusted.
echo "### template registration:" | tee -a "$LOG"
# Report the FULL distribution of template counts, not just 0-vs-5. The first attempt logged
# "Found 1 templates": Protenix de-duplicates hits by sequence, so five donors sharing one mutated
# sequence collapsed to one. A 0-vs-5 check would have passed that silently.
grep -oiE "Found [0-9]+ templates" "$LOG" | sort | uniq -c | tee -a "$LOG"
echo "### done rc=$rc  cifs=$(find "$OUT" -name '*_sample_*.cif' 2>/dev/null | wc -l)  $(date +%H:%M)" \
  | tee -a "$LOG"
