#!/usr/bin/env bash
# Keep the protein-protein queue topped up so the GPUs never run dry.
#
# run_pp.sh re-reads its complex list on EVERY claim iteration, so appending to that file is picked up
# live by the running workers -- no restart, no idle window. When fewer than THRESH complexes remain
# unfinished, append the next BATCH from artifacts/pp/all_by_size.txt (smallest first, skipping any
# already queued or marked dead).
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
Q=$REPO/artifacts/pp/subset50.txt
ALL=$REPO/artifacts/pp/all_by_size.txt
BATCH=${BATCH:-50}
THRESH=${THRESH:-8}
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/pp_extend.log; }
say "queue extender started (top up by $BATCH when <$THRESH remain)"
while true; do
  remaining=0
  while read -r p _; do
    [ -f "artifacts/pp/locks/$p.dead" ] && continue
    n=$(find "artifacts/pp/out/$p" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -lt 250 ] && remaining=$((remaining+1))
  done < "$Q"
  queued=$(wc -l < "$Q")
  total=$(wc -l < "$ALL")
  if [ "$remaining" -lt "$THRESH" ] && [ "$queued" -lt "$total" ]; then
    added=0
    while read -r p t c; do
      grep -qx "$p $t $c" "$Q" && continue
      grep -q "^$p " "$Q" && continue
      echo "$p $t $c" >> "$Q"; added=$((added+1))
      [ "$added" -ge "$BATCH" ] && break
    done < "$ALL"
    say "only $remaining unfinished -> appended $added complexes (queue now $(wc -l < $Q)/$total)"
  fi
  if [ "$queued" -ge "$total" ] && [ "$remaining" -eq 0 ]; then
    say "ALL $total protein-protein complexes complete"; break
  fi
  sleep 180
done
