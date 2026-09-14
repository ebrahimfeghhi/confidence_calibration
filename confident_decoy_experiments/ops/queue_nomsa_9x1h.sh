#!/usr/bin/env bash
# Queued behind the N_cycle=20 arms: the same two templated arms at N_cycle=10 with NO MSA.
#
# TWO QUESTIONS AT ONCE.
# 1. Does the good basin hold without co-evolutionary evidence? The template supplies the arrangement;
#    the MSA supplies everything the model knows about the sequences. Whether steering survives with
#    the MSA removed says how much of the effect is the template versus the MSA agreeing with it.
# 2. A DIRECT TEST OF THE SEED MECHANISM. ipTM's variance is 87% between-seed across PXM and ~100% on
#    9x1h; we traced that to sample_indices drawing a random SIZE and SUBSET of MSA rows per seed,
#    reaching the confidence head only via z_trunk. PROTENIX_MSA_NROWS=1 + strategy=topk pins the
#    depth to the query row alone, so nothing is left to differ between seeds except diffusion noise.
#    BETWEEN-SEED ipTM VARIANCE SHOULD COLLAPSE. If it does not, the mechanism is wrong and something
#    else carries the seed signal.
#
# Waits for the N=20 arms so the four runs do not contend for the same GPUs, then hands GPUs 1 and 2
# back to the PXM sweep (it is currently on GPUs 0 and 6 only, so nothing was taken from it).
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
T=$REPO/artifacts/selftmpl_tmpl
SEEDS=400001,400002,400003,400004,400005,400006,400007,400008,400009,400010

while tmux has-session -t g20 2>/dev/null || tmux has-session -t d20 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) N=20 arms finished -> starting the no-MSA arms"

tmux new-session -d -s gnm -c "$REPO" \
  "RUN=x1h_full_nomsa PDB=9x1h GPU=1 NCYCLE=10 NTMPL=1 MASK=0 KEEPFRAME=1 NOMSA=1 \
   SEEDS=$SEEDS JSON=$T/x1h_full/9x1h_x1h_full.json bash ops/run_multitmpl.sh"
tmux new-session -d -s dnm -c "$REPO" \
  "RUN=x1h_decoy_nomsa PDB=9x1h GPU=2 NCYCLE=10 NTMPL=1 MASK=0 KEEPFRAME=1 NOMSA=1 \
   SEEDS=$SEEDS JSON=$T/x1h_decoy/9x1h_x1h_decoy.json bash ops/run_multitmpl.sh"
echo "$(date +%H:%M:%S) launched gnm (gpu1, good donor) and dnm (gpu2, decoy donor), no MSA"
