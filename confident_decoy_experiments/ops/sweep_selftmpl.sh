#!/usr/bin/env bash
# Generational self-templating across the 38-complex set, up to MAX_SLOTS GPUs at a time.
#
# Same GPU-politeness rules as ops/sweep_49.sh: only claims a card with ZERO compute processes, never
# more than MAX_SLOTS at once, and excludes any card another of our own workers already holds -- a
# worker takes ~100s to load before it shows up in nvidia-smi, and a naive check double-books it.
#
# Smallest complex first, so cheap results bank early.
# RESUMABLE: gen_selftmpl.py skips any (generation, lineage) that already has its 5 poses.
set -o pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
MAX_SLOTS=${MAX_SLOTS:-3}
LIST=${LIST:-$EXP/ops/sweep_38_order.txt}
LOG=$EXP/ops/selftmpl.log
EXTRA=${EXTRA:-}

cd "$EXP" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ours_gpus() { pgrep -fa "gen_selftmpl.py|confornet_protenix.py" 2>/dev/null | grep -oP -- "--gpu \K[0-9]+" | sort -u; }
free_gpus() {
  local busy mine force
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  mine=$(printf '%s\n%s\n' "$(ours_gpus)" "$(echo "${EXCLUDE_GPUS:-}" | tr ',' '\n')" | sed '/^$/d' | sort -u)
  # FORCE_GPUS: claim these even when another user holds them. Deliberate co-tenancy, only ever with
  # explicit permission from the person whose job is already there -- it is their memory we are eating
  # into, and if they then need more than what is left, THEY OOM, not us.
  force=$(echo "${FORCE_GPUS:-}" | tr ',' '\n' | sed '/^$/d')
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read -r i u; do
    u=$(echo "$u" | tr -d ' '); i="${i// /}"
    grep -qx "$i" <<<"$mine" && continue
    if ! grep -qx "$i" <<<"$force"; then
      grep -qx "$u" <<<"$busy" && continue
    fi
    echo "$i"
  done
}

n_poses() {                      # poses this complex has produced (V2-V5; V1 is reused, not written)
  find "$EXP/artifacts/selftmpl/${1}_gen/decoded" -name "*_sample_*.cif" 2>/dev/null | wc -l
}

[ -f "$LIST" ] || { say "missing $LIST"; exit 1; }
say "self-templating sweep: max_slots=$MAX_SLOTS complexes=$(wc -l < "$LIST") extra='$EXTRA'"

declare -A PID2P PID2G
for p in $(awk '{print $1}' "$LIST"); do
  # V2-V5 = 4 generations x 10 lineages x 5 samples = 200 poses (V1's 50 come from the free run)
  if [ "$(n_poses "$p")" -ge 200 ]; then say "$p: already complete, skipping"; continue; fi

  gpu=""
  while [ -z "$gpu" ]; do
    if [ "${#PID2P[@]}" -gt 0 ]; then
      for pid in "${!PID2P[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"; r=$?
        say "${PID2P[$pid]}: finished rc=$r on GPU ${PID2G[$pid]}, $(n_poses "${PID2P[$pid]}") poses"
        unset "PID2P[$pid]" "PID2G[$pid]"
      done
    fi
    if [ "${#PID2P[@]}" -lt "$MAX_SLOTS" ]; then
      for c in $(free_gpus); do
        inuse=0
        if [ "${#PID2G[@]}" -gt 0 ]; then for g in "${PID2G[@]}"; do [ "$g" = "$c" ] && inuse=1; done; fi
        [ "$inuse" -eq 0 ] && { gpu=$c; break; }
      done
    fi
    [ -z "$gpu" ] && sleep 60
  done

  say "$p: starting on GPU $gpu ($(grep -w "^$p" "$LIST" | awk '{print $2}') tokens)"
  $PY scripts/gen_selftmpl.py --pdb "$p" --gpu "$gpu" $EXTRA >>"$EXP/ops/selftmpl_${p}.log" 2>&1 &
  PID2P[$!]=$p; PID2G[$!]=$gpu
  sleep 20
done

for pid in "${!PID2P[@]}"; do
  wait "$pid"; r=$?
  say "${PID2P[$pid]}: finished rc=$r on GPU ${PID2G[$pid]}"
done
say "ALL DONE. total poses = $(find "$EXP/artifacts/selftmpl" -name "*_sample_*.cif" 2>/dev/null | wc -l)"
