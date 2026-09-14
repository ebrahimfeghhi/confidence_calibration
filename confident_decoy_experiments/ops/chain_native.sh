#!/usr/bin/env bash
# The native-template arm on one GPU, after that GPU's donor arms finish, then hand the card back.
#
# WHAT THIS SEPARATES. s1_only showed that ONE self-generated pose as the sole template drops the
# good-seed rate from 10/25 to 1/25. Two explanations survive: (a) something about the model's OWN
# output specifically, or (b) any single dominant template does this, self-generated or not. The
# ground truth is a real crystal structure, so if s1_native ALSO collapses, (a) is dead and the
# variable is "one template that agrees with itself", not "self-ness".
#
# THIS IS AN ORACLE CONDITION, not a method: it hands the model the answer's folds. Even so it cannot
# transmit the answer's DOCK, because template coordinates are zero-centred per chain
# (template_utils.py:487) and multichain_mask is on -- so what it transmits is the native FOLDS only,
# which is the same class of information a self-template carries.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPU=${GPU:?set GPU}
SEEDS=${SEEDS:?set SEEDS}
while tmux has-session -t "dn_$GPU" 2>/dev/null; do sleep 15; done
# dn_ restarts pxm<GPU> as its last act; stop it so this arm gets the card
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 5
env RUN=s1_native PDB=8vyl GPU="$GPU" NCYCLE=10 SEEDS="$SEEDS" NTMPL=1 \
    JSON="$REPO/artifacts/selftmpl_tmpl/s1_native/8vyl_s1_native.json" \
    bash "$REPO/ops/run_multitmpl.sh"
echo "$(date +%H:%M:%S) GPU$GPU s1_native done -> restarting pxm$GPU"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025.sh"
