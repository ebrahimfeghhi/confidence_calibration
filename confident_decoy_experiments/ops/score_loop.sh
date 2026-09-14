#!/bin/bash
# Hourly scoring + progress, written to ops/score_latest.txt (overwritten) and appended to
# ops/score_history.txt. Runs in tmux so it outlives any one Claude session.
#
# Only ONE scorer may run at a time: score_wnm and score_bothoff share the OST cache, and two
# concurrent passes would race writing the same *_ost/*.json files.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
cd $REPO
while true; do
    {
        echo "=== $(date +'%m-%d %H:%M:%S') ==="
        $PY - <<'PYEOF'
import glob, os
A='artifacts'
def done(q, need):
    if not os.path.exists(f'{A}/{q}'): return None
    arms=[l.strip() for l in open(f'{A}/{q}') if l.strip()]
    d=sum(1 for a in arms
          if len(glob.glob(f'{A}/sweep_out/{a}/**/*_sample_*.cif', recursive=True))>=need)
    return d, len(arms)
for q,need,lab in [('d50_queue.txt',200,'d50 (1e-4, 111-150)'),
                   ('bo50_queue.txt',200,'bo50 (both-off, 111-150)')]:
    r=done(q,need)
    if r: print(f'{lab:<28} {r[0]}/{r[1]} arms')
for a in ['msaperm_perm_8bf4','msaperm_topk_8bf4']:
    n=len(glob.glob(f'{A}/sweep_out/{a}/**/*_sample_*.cif', recursive=True))
    c=len(glob.glob(f'{A}/{a}_cycles/*.npz'))
    print(f'{a:<28} {n}/50 poses, {c} cycle dumps')
PYEOF
        echo "gpus: $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
        echo "cores: $(ps -u ebrahim -o pcpu= | awk '{s+=$1} END {printf "%.0f", s/100}')"
        err=$(grep -hcE "Traceback|out of memory" $REPO/ops/d50w*.log $REPO/ops/bo50w*.log \
                 $REPO/ops/msaperm.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
        echo "errors: ${err:-0}"
        echo "--- scoring ---"
        # The 10-seed ladder and both-off arms are FINISHED -- their inputs cannot change, so
        # re-scoring them hourly just recomputes a constant. Only the 50-seed arms are still growing.
        $PY scripts/score_50seed.py --jobs 6 --arm all
    } > $REPO/ops/score_latest.txt 2>&1
    cat $REPO/ops/score_latest.txt >> $REPO/ops/score_history.txt
    sleep 3600
done
