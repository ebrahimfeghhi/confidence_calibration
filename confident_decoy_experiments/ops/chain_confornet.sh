#!/usr/bin/env bash
# Wait for the self-templating sweep to finish, then start the ConforNet v3 sweep on GPUs 3+4.
#
# WHY A WATCHER, NOT `tmux send-keys`: send-keys only queues if the target pane's shell is actually
# blocked on a foreground job. Twice today the target shell was idle (or the wrong session), so the
# command fired IMMEDIATELY -- once landing on john's GPU 6 and once double-booking GPU 4 next to a
# running self-templating worker. Polling for the driver's exit is unambiguous.
#
# Starts NOTHING if self-templating exited abnormally -- if there are still self-templating workers
# alive, the sweep is not really done and launching would collide.
set -o pipefail
EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
LOG=$EXP/ops/chain_confornet.log
cd "$EXP" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the self-templating sweep to finish before starting ConforNet"
while pgrep -f "bash ops/sweep_selftmpl.sh" >/dev/null; do sleep 60; done
say "selftmpl driver gone; waiting for its workers to drain"
while ps -eo args= | grep -q "[g]en_selftmpl.py --pdb"; do sleep 60; done

n=$(find artifacts/selftmpl -name "*_sample_*.cif" 2>/dev/null | wc -l)
say "self-templating finished with $n poses"
say "starting ConforNet v3 on GPUs 3+4 (resumes; completed complexes are skipped)"
TAGPFX=v3 MAX_SLOTS=2 EXCLUDE_GPUS=0,1,2,5,6,7 NRUNS=10 STEPS=20 \
  LIST=ops/sweep_38_order.txt bash ops/sweep_49.sh
say "ConforNet sweep exited rc=$?"
