#!/usr/bin/env bash
# Hold the assembly-rebuild arm until the ORIGINAL protein-protein sweep drains, then start it on
# whichever GPUs have gone idle. Option 1 of the plan: recover the homodimers with GPU time that
# would otherwise sit unused, without competing with the 66 complexes still queued.
#
# "Drained" = every id in all_by_size.txt is either complete (250 poses) or marked .dead, AND the
# queue has been topped up to the full 239 by ops/pp_extend.sh. Both conditions, so we cannot start
# early just because the current 200-entry queue happens to be momentarily satisfied.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PP=$REPO/artifacts/pp
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a ops/pp_asm_gate.log; }
say "gate armed: waiting for the ASU sweep to drain before starting the 43-target rebuild"
while true; do
  total=$(wc -l < "$PP/all_by_size.txt"); queued=$(wc -l < "$PP/subset50.txt"); left=0
  while read -r p _; do
    [ -f "$PP/locks/$p.dead" ] && continue
    [ "$(find "$PP/out/$p" -name '*_sample_*.cif' 2>/dev/null | wc -l)" -lt 250 ] && left=$((left+1))
  done < "$PP/all_by_size.txt"
  if [ "$queued" -ge "$total" ] && [ "$left" -eq 0 ]; then
    say "ASU sweep drained ($total targets). starting rebuild arm on GPUs 0/3/4"
    for g in 0 3 4; do
      tmux kill-session -t ppasm_$g 2>/dev/null
      tmux new-session -d -s ppasm_$g "cd $REPO && ./ops/run_pp_asm.sh $g 2>&1 | tee -a ops/pp_asm_g$g.log"
      say "  launched ppasm_$g"
    done
    break
  fi
  sleep 300
done
say "gate released"
