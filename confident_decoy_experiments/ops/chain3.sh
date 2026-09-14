#!/usr/bin/env bash
# One GPU's share of three arms, run back to back, then the GPU goes back to PXM.
#
# THE THIRD ARM IS THE POSITIVE CONTROL and is the reason this exists. Every previous template arm was
# launched by ops/run_multitmpl.sh and every one collapsed, while the 75/245 baseline came from a
# DIFFERENT launcher (the original FoldBench sweep). So "self-templates suppress search" was not
# separable from "this wrapper suppresses search". s1_ctl runs the stock JSON, unmodified, through
# this same wrapper at the same N_cycle: if it reproduces ~75/245 the pipeline is validated and the
# self-templating results stand; if it collapses, every arm so far is an artefact.
#
# s1_only  : the single best pose (seed 125 sample 4, DockQ 0.982/0.961) as the ONLY template
# s1_stock : that same pose PLUS the stock hmmsearch templates
# s1_ctl   : stock templates only, no donor  <- the control
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPU=${GPU:?}; SEEDS=${SEEDS:?}
T=$REPO/artifacts/selftmpl_tmpl
# NOTE: an env assignment prefixing a command must be LITERAL at parse time. Writing
# `RUN=$1 ... ${3:+NTMPL=$3} bash ...` makes bash expand ${3:+...} only after it has picked the
# command word, so the expansion is treated as the COMMAND -- "NTMPL=1: command not found" -- and the
# arm exits instantly having run nothing. Use `env` so the assignment is an argument, not a prefix.
run () {
  local extra=()
  [ -n "${3:-}" ] && extra=(NTMPL="$3")
  env RUN="$1" PDB=8vyl GPU="$GPU" NCYCLE=10 SEEDS="$SEEDS" JSON="$2" "${extra[@]}" \
      bash "$REPO/ops/run_multitmpl.sh"
}
run s1_only  "$T/s1_only/8vyl_s1_only.json"   1
run s1_stock "$T/s1_stock/8vyl_s1_stock.json" 5
run s1_ctl   /home/ebrahim/Protenix/foldbench_abag/jsons/8vyl-final-updated.json
echo "$(date +%H:%M:%S) GPU$GPU all three arms done -> restarting pxm$GPU"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025.sh"
