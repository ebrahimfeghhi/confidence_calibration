#!/usr/bin/env bash
# When the 9x1h seed-300014 repeats finish: commit11 arms on GPU5, free sweep back on GPU2 and GPU6.
#
# Waits for the repeat runners to EXIT rather than counting outputs -- a runner exits only when its
# three folds are done, which is the real definition of finished.
#
# GPU5: the arms for 9s36_AB seed 300003, the one fold whose five poses genuinely disagree
#       (0.026 0.065 0.078 0.117 0.316, span 0.290). Its trunk.npz and donor CIFs are already on
#       disk, so this skips the fold entirely.
# GPU2/6: the 25-seed free sweep (300075-300099, 131 complexes), which is restartable and simply
#       skips any seed that already has its five CIFs.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
LOG=ops/after_repeats.log
say () { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

say "waiting for the 300014 repeats to finish"
while pgrep -u "$(id -u)" -f 'repeat14.sh' >/dev/null 2>&1; do sleep 60; done
say "repeats done"
sleep 30

say "GPU5 -> commit11 arms, 9s36_AB seed 300003"
tmux kill-session -t c11 2>/dev/null
tmux new-session -d -s c11 "cd $REPO && GPU=5 PDB=9s36 IFACE=9s36_AB SEED=300003 bash ops/commit11.sh 2>&1 | tee ops/commit11_9s36.log"

for g in 2 6; do
  say "GPU$g -> free sweep (seeds 300075-300099)"
  tmux kill-session -t pxm$g 2>/dev/null
  tmux new-session -d -s pxm$g "cd $REPO && GPU=$g SEED_LO=300075 NSEEDS=25 bash ops/run_pxm2025_more.sh"
  sleep 5
done

sleep 90
say "running now: commit11=$(pgrep -u "$(id -u)" -fc commit11.sh) freesweep=$(pgrep -u "$(id -u)" -fc run_pxm2025_more.sh)"
for i in 2 5 6; do say "  GPU$i $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i $i)"; done
