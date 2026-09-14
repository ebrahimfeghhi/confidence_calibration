#!/usr/bin/env bash
# The two single-donor arms on one GPU, run AFTER that GPU's control arm finishes, then hand the card
# back to PXM.
#
# ORDER MATTERS: it waits on ch3_<GPU> so s1_ctl completes first. s1_ctl is the arm that decides
# whether any of this is interpretable -- every template arm so far was launched by
# ops/run_multitmpl.sh and every one collapsed, while the 75/245 baseline came from a DIFFERENT
# launcher, so "self-templates suppress search" was never separable from "this wrapper suppresses
# search". If s1_ctl reproduces the baseline the wrapper is clean; if it collapses these two arms are
# pointless and should be killed.
#
#   s1_only  : seed 125 sample 4 (DockQ 0.982 / 0.961) as the ONLY template
#   s1_stock : that same pose PLUS the stock hmmsearch templates
#
# NOTE on the env prefix: `RUN=$1 ... ${3:+NTMPL=$3} bash ...` does NOT work -- bash fixes the command
# word before expanding, so the expansion becomes the command ("NTMPL=1: command not found") and the
# arm exits having run nothing. That silently killed both donor arms once already. `env` takes the
# assignments as ordinary arguments, so it is safe.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPU=${GPU:?set GPU}
SEEDS=${SEEDS:?set SEEDS}
T=$REPO/artifacts/selftmpl_tmpl

while tmux has-session -t "ch3_$GPU" 2>/dev/null; do sleep 15; done
# ch3_ restarts pxm<GPU> as its last act; stop that so the donor arms get the card
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 5

env RUN=s1_only PDB=8vyl GPU="$GPU" NCYCLE=10 SEEDS="$SEEDS" NTMPL=1 \
    JSON="$T/s1_only/8vyl_s1_only.json" bash "$REPO/ops/run_multitmpl.sh"
env RUN=s1_stock PDB=8vyl GPU="$GPU" NCYCLE=10 SEEDS="$SEEDS" NTMPL=5 \
    JSON="$T/s1_stock/8vyl_s1_stock.json" bash "$REPO/ops/run_multitmpl.sh"

echo "$(date +%H:%M:%S) GPU$GPU donor arms done -> restarting pxm$GPU"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025.sh"
