#!/usr/bin/env bash
# HEALTH WATCHER for the overnight protein-protein sweep. Detects genuinely failing complexes,
# remediates once, and retires a complex that fails twice so it cannot loop forever.
#
# A watcher that only prints is not enough: an assert once looped a worker 464x and burned a GPU for
# 16h while a health log dutifully flagged it. This one acts.
#
# FAILING means, per complex:
#   - its log contains Traceback / CUDA OOM / CUDA error, or
#   - it has been CLAIMED for > STALL_MIN minutes with zero poses written.
# Remediation: release the claim and delete partial output once, so another GPU retries it.
# Second failure -> .dead marker, skipped from then on (protein-protein complexes vary hugely in size;
# a genuine OOM on a big one should not consume the whole night).
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
LOG=ops/pp_health.log
STALL_MIN=${STALL_MIN:-60}
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "health watcher started (stall ${STALL_MIN}m)"
while true; do
  tot=0; dead=0; done_n=0; oom=0
  while read -r p _; do
    tot=$((tot+1))
    [ -f "artifacts/pp/locks/$p.dead" ] && { dead=$((dead+1)); continue; }
    n=$(find "artifacts/pp/out/$p" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 250 ] && { done_n=$((done_n+1)); continue; }
    [ -d "artifacts/pp/locks/$p" ] || continue          # unclaimed: nothing to judge
    bad=0; why=""
    if grep -qlE "Traceback|out of memory|CUDA error|CUDA out of memory" "artifacts/pp/logs/$p.log" 2>/dev/null; then
      bad=1; why="error in log"
      grep -qi "out of memory" "artifacts/pp/logs/$p.log" 2>/dev/null && { oom=$((oom+1)); why="CUDA OOM"; }
    fi
    if [ "$n" -eq 0 ] && [ -n "$(find "artifacts/pp/locks/$p" -maxdepth 0 -mmin +$STALL_MIN 2>/dev/null)" ]; then
      bad=1; why="claimed >${STALL_MIN}m with 0 poses"
    fi
    if [ "$bad" = 1 ]; then
      if [ -f "artifacts/pp/locks/$p.retried" ]; then
        say "$p: FAILED TWICE ($why) -> marking dead, skipping"
        touch "artifacts/pp/locks/$p.dead"; dead=$((dead+1))
      else
        say "$p: $why -> releasing claim for one retry"
        touch "artifacts/pp/locks/$p.retried"
        rm -rf "artifacts/pp/locks/$p" "artifacts/pp/out/$p" "artifacts/pp/logs/$p.log"
      fi
    fi
  done < artifacts/pp/subset50.txt
  # a GPU held by nobody while work remains is the supervisor's job, but log it if it persists
  idle=0
  for g in 0 3 4; do
    tmux has-session -t "pp_$g" 2>/dev/null || tmux has-session -t "cd_$g" 2>/dev/null || idle=$((idle+1))
  done
  [ "$idle" -gt 0 ] && say "WARNING: $idle GPU worker session(s) missing (supervisor should restart)"
  [ "$oom" -gt 0 ] && say "note: $oom complex(es) hit CUDA OOM -- these are the largest in the subset"
  if [ "$done_n" -eq "$((tot-dead))" ] && [ "$tot" -gt 0 ]; then
    say "all reachable complexes complete ($done_n done, $dead dead) -- watcher exiting"; break
  fi
  sleep 300
done
