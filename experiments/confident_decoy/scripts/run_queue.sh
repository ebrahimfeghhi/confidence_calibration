#!/bin/bash
# Overnight queue: finish dockqdonor -> score it -> ifacelddt26 -> score -> globallddt26 -> score.
#
# Each arm is scored the moment its generation finishes, so results are on disk without anyone asking.
# Runs strictly sequentially: only one scheduler owns GPUs 0-3 at a time, and scoring is CPU/OST so it
# overlaps the next arm's generation harmlessly.
#
# PATHS ARE ABSOLUTE ON PURPOSE. sweep_sched.py runs inference with cwd=/home/ebrahim/Protenix, so a
# relative --out silently resolves against THAT directory: the first dockqdonor launch wrote every pose
# to /home/ebrahim/Protenix/artifacts/sweep_out/... while the scheduler looked for them under the
# experiments dir and logged "0/50 seeds" for jobs that had actually succeeded. Phase 0 below relocates
# that output; every later phase passes an absolute path so it cannot recur.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
STRAY=/home/ebrahim/Protenix/artifacts/sweep_out/dockqdonor/dockqdonor
DEST=$EXP/artifacts/sweep_out/dockqdonor/dockqdonor
say() { echo "[$(date +%H:%M)] $*"; }

wait_for_sched() {   # $1 = jobs-file fragment identifying the scheduler
  while :; do
    alive=0
    for p in /proc/[0-9]*; do
      c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
      case "$c" in *sweep_sched.py*$1*) alive=1; break;; esac
    done
    [ "$alive" -eq 0 ] && break
    sleep 120
  done
}

# ---------- phase 0: wait for the already-running dockqdonor scheduler ----------
say "waiting for the running dockqdonor scheduler to finish"
wait_for_sched dockq_arm_jobs
say "dockqdonor generation finished"

# relocate the misplaced output, then verify before trusting it
if [ -d "$STRAY" ]; then
  mkdir -p "$DEST"
  rsync -a "$STRAY"/ "$DEST"/ && say "relocated $(du -sh "$DEST" | cut -f1) from the Protenix tree"
fi
say "dockqdonor complete seeds: $(ls $DEST/*/seed_*/predictions/*_summary_confidence_sample_4.json 2>/dev/null | wc -l)/1300"

# ---------- phase 1: score dockqdonor ----------
say "scoring dockqdonor"
$PY $EXP/scripts/score_arm.py --arm dockqdonor \
    --jobs $EXP/artifacts/dockq_arm_jobs.json \
    --out $EXP/artifacts/sweep_out/dockqdonor 2>&1 | tail -40
say "dockqdonor scored -> artifacts/dockqdonor_scores.json"

# ---------- phases 2-5: the two lDDT arms, generate then score ----------
for ARM in ifacelddt26 globallddt26; do
  say "launching $ARM (26 jobs x 50 seeds)"
  $PY $EXP/scripts/sweep_sched.py \
      --jobs $EXP/artifacts/lddt26_${ARM}_jobs.json \
      --out $EXP/artifacts/sweep_out/$ARM \
      --logs $EXP/artifacts/${ARM}_logs \
      --status $EXP/artifacts/${ARM}_status.json 2>&1 | tail -20
  say "$ARM generation finished; scoring"
  $PY $EXP/scripts/score_arm.py --arm $ARM \
      --jobs $EXP/artifacts/lddt26_${ARM}_jobs.json \
      --out $EXP/artifacts/sweep_out/$ARM 2>&1 | tail -40
  say "$ARM scored -> artifacts/${ARM}_scores.json"
done

say "QUEUE COMPLETE"
for f in dockqdonor ifacelddt26 globallddt26; do
  n=$($PY -c "import json;d=json.load(open('$EXP/artifacts/${f}_scores.json'));print(f'{sum(1 for v in d.values() if v[\"rescued\"])}/{len(d)}')" 2>/dev/null)
  say "  $f rescued $n"
done
