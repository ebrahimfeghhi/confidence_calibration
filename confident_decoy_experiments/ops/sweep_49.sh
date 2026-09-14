#!/usr/bin/env bash
# 800-pose ConforNet sweep across the 49-complex train set, up to MAX_SLOTS GPUs at a time.
#
# POLITENESS: only claims a GPU with ZERO compute processes, and never more than MAX_SLOTS at once.
# Free memory on someone else's card is not permission -- several cards here sit at <1 GB while
# pinning SMs for another user.
#
# ORDER: smallest complex first. Cost scales ~N_token^2, so the cheap ones bank results early and
# the expensive ones (which may not fit at all) are last, where they cost least if we stop.
#
# OOM: peak was 66.1 GiB at 942 tokens, so anything past ~1150 tokens is expected to exceed a 97.9
# GiB card at k=2. Those are attempted with --chunk-size 32 on a retry, then recorded as deferred
# rather than silently skipped -- a sweep that quietly drops 11 of 49 reads as "done" when it isn't.
#
# RESUMABLE: a complex whose 20 seeds all have complete CSVs is skipped.
# NOTE: no `set -u`. Expanding an EMPTY associative array ("${!PID2P[@]}") trips "unbound variable"
# on the first scheduling pass, which aborted the loop and fell through to "ALL DONE" with 0 poses --
# a silent no-op that looked like success. pipefail is kept.
set -o pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
MAX_SLOTS=${MAX_SLOTS:-4}
STEPS=${STEPS:-20}
NRUNS=${NRUNS:-20}
LOG=$EXP/ops/sweep_49.log
LIST=${LIST:-$EXP/ops/sweep_49_order.txt}   # "<pdb> <n_token>" per line, ascending

cd "$EXP" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# GPUs any OTHER confornet process of ours is already assigned to, whether or not it currently holds
# a CUDA context. This is the fix for the self-collision that cost 8che five seeds: a worker takes
# ~105 s to load its checkpoint and featurize, and during that window nvidia-smi shows ZERO compute
# processes on its card -- so a naive free_gpus() sees the gap between two sequential seeds as "free"
# and double-books it. Reading the --gpu flag off the process table closes the window, because the
# process exists for the whole startup even though the GPU looks idle.
ours_gpus() {
  pgrep -fa "confornet_protenix.py" 2>/dev/null | grep -oP -- "--gpu \K[0-9]+" | sort -u
}

free_gpus() {
  local busy mine
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  mine=$(ours_gpus)
  # EXCLUDE_GPUS lets a second driver be told about cards it cannot see in the process table yet.
  mine=$(printf '%s\n%s\n' "$mine" "$(echo "${EXCLUDE_GPUS:-}" | tr ',' '\n')" | sed '/^$/d' | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read -r i u; do
    u=$(echo "$u" | tr -d ' '); i="${i// /}"
    grep -qx "$u" <<<"$busy" && continue
    grep -qx "$i" <<<"$mine" && continue
    echo "$i"
  done
}

complex_done() {                       # all NRUNS seeds present with full CSVs?
  local p=$1 n=0
  for r in $(seq 0 "$STEPS" $(( NRUNS*STEPS - STEPS ))); do
    local c="$EXP/artifacts/confornet/${p}_${TAGPFX:-sweep}_s${r}/training_loss.csv"
    # STEPS+2 lines = header + STEPS+1 data rows (the loop runs STEPS+1 iterations so the last one
    # can decode a STEPS-update affine). STEPS+1 would accept a run that died one iteration early.
    [ -f "$c" ] && [ "$(wc -l <"$c")" -ge "$((STEPS+2))" ] && n=$((n+1))
  done
  [ "$n" -ge "$NRUNS" ]
}

# An OOM means "this complex is too big" ONLY if it was alone on the card. 7yds (585 tokens, ~25 GiB
# predicted) was recorded as DEFERRED purely because another of our jobs held 74.9 GiB of the same
# GPU -- a wrong label on a perfectly runnable complex, which is worse than a missing one because it
# looks like a finding. Predicted peak comes from the measured 66.1 GiB at 942 tokens, scaling ~N^2.
plausible_oom() {                      # $1=n_token ; true if this complex really cannot fit alone
  awk -v t="$1" 'BEGIN { exit !(66.1*(t/942)^2 > 97.9) }'
}

run_complex() {                        # $1=pdb $2=gpu $3=n_token ; runs all NRUNS seeds sequentially
  local p=$1 g=$2 tok=${3:-0} rc=0
  for r in $(seq 0 "$STEPS" $(( NRUNS*STEPS - STEPS ))); do
    local tag="${p}_${TAGPFX:-sweep}_s${r}"
    local csv="$EXP/artifacts/confornet/$tag/training_loss.csv"
    [ -f "$csv" ] && [ "$(wc -l <"$csv")" -ge "$((STEPS+2))" ] && continue
    $PY scripts/confornet_protenix.py --pdb "$p" --gpu "$g" --k 2 --seed "$r" \
        --steps "$STEPS" --tag "$tag" >>"$EXP/ops/sweep49_${p}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] && grep -q "out of memory" "$EXP/ops/sweep49_${p}.log"; then
      say "$p: OOM at k=2, retrying seed $r with --chunk-size 32"
      rc=0
      $PY scripts/confornet_protenix.py --pdb "$p" --gpu "$g" --k 2 --seed "$r" \
          --steps "$STEPS" --tag "$tag" --chunk-size 32 >>"$EXP/ops/sweep49_${p}.log" 2>&1 || rc=$?
      if [ "$rc" -ne 0 ]; then
        if plausible_oom "$tok"; then
          say "$p: STILL OOM with chunking at $tok tokens -- genuinely too large, DEFERRED"; return 2
        fi
        # Small complex that OOM'd => something else was on the card. Do NOT label it deferred.
        say "$p: OOM at only $tok tokens -- co-tenant on GPU $g, NOT a size limit; REQUEUE needed"
        echo "$p $tok" >> "$EXP/ops/sweep_requeue.txt"; return 3
      fi
    fi
    [ "$rc" -ne 0 ] && { say "$p: seed $r failed rc=$rc -- see ops/sweep49_${p}.log"; return 1; }
  done
  return 0
}

[ -f "$LIST" ] || { say "missing $LIST"; exit 1; }
say "49-complex sweep: max_slots=$MAX_SLOTS runs/complex=$NRUNS steps=$STEPS"

declare -A PID2P PID2G
pending=$(awk '{print $1}' "$LIST")
for p in $pending; do
  if complex_done "$p"; then say "$p: already complete, skipping"; continue; fi

  gpu=""
  while [ -z "$gpu" ]; do                     # wait for a slot AND a genuinely free card
    if [ "${#PID2P[@]}" -gt 0 ]; then
      for pid in "${!PID2P[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"; r=$?
        n=$(find "$EXP/artifacts/confornet" -path "*${PID2P[$pid]}_${TAGPFX:-sweep}_s*" -name "*.cif" 2>/dev/null | wc -l)
        say "${PID2P[$pid]}: finished rc=$r on GPU ${PID2G[$pid]}, $n poses"
        unset "PID2P[$pid]" "PID2G[$pid]"
      done
    fi
    if [ "${#PID2P[@]}" -lt "$MAX_SLOTS" ]; then
      for c in $(free_gpus); do
        inuse=0
        if [ "${#PID2G[@]}" -gt 0 ]; then
          for g in "${PID2G[@]}"; do [ "$g" = "$c" ] && inuse=1; done
        fi
        [ "$inuse" -eq 0 ] && { gpu=$c; break; }
      done
    fi
    [ -z "$gpu" ] && sleep 60
  done

  say "$p: starting on GPU $gpu ($(grep -w "^$p" "$LIST" | awk '{print $2}') tokens)"
  run_complex "$p" "$gpu" "$(grep -w "^$p" "$LIST" | awk '{print $2}')" &
  PID2P[$!]=$p; PID2G[$!]=$gpu
  sleep 20                                    # let it claim the card before the next free_gpus poll
done

for pid in "${!PID2P[@]}"; do
  wait "$pid"; r=$?
  say "${PID2P[$pid]}: finished rc=$r on GPU ${PID2G[$pid]}"
done

tot=$(find "$EXP/artifacts/confornet" -name "*.cif" -path "*_${TAGPFX:-sweep}_s*" 2>/dev/null | wc -l)
say "ALL DONE. total poses across complexes = $tot"
