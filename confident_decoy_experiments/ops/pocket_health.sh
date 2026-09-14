#!/usr/bin/env bash
# HEALTH WATCHER for the overnight pocket run. Detects a run that is failing rather than merely slow,
# tries one bounded remediation, and if that does not work RETIRES the pocket run and lets the
# supervisor fall through to deep-adaptive scaling -- so the GPUs keep doing useful work either way.
#
# A watcher that only prints is not enough: an 8evd assert once looped a worker 464x and burned a GPU
# for 16h while the health log dutifully flagged it. This one acts.
#
# FAILING means, per interface:
#   - its log contains a Traceback / OOM / CUDA error, or
#   - it has been claimed for > STALL_MIN minutes with zero poses written.
# Remediation: release the claim and delete the partial output once, so another GPU retries it. An
# interface that fails twice is marked dead (a .dead file) and skipped, rather than retried forever.
# If MORE THAN HALF the interfaces end up dead, the pocket experiment is considered broken: the
# remaining jsons are moved aside so pocket_left() goes to 0 and the supervisor starts deep-adaptive.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
LOG=ops/pocket_health.log
STALL_MIN=${STALL_MIN:-45}
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "health watcher started (stall threshold ${STALL_MIN}m)"
while true; do
  total=0; dead=0; done_n=0
  for f in artifacts/pocket_jsons/*_pocket.json; do
    [ -e "$f" ] || continue
    t=$(basename "$f" _pocket.json); total=$((total+1))
    [ -f "artifacts/pocket_locks/$t.dead" ] && { dead=$((dead+1)); continue; }
    n=$(find "artifacts/sweep_out/pocket/pocket_$t" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -ge 250 ] && { done_n=$((done_n+1)); continue; }
    [ -d "artifacts/pocket_locks/$t" ] || continue          # unclaimed: nothing to judge
    bad=0
    grep -qlE "Traceback|out of memory|CUDA error|AssertionError" artifacts/pocket_logs/*${t%%_*}*.log 2>/dev/null && bad=1
    if [ "$n" -eq 0 ] && [ -n "$(find "artifacts/pocket_locks/$t" -maxdepth 0 -mmin +$STALL_MIN 2>/dev/null)" ]; then
      bad=1; say "$t: claimed >${STALL_MIN}m with 0 poses -- treating as stalled"
    fi
    if [ "$bad" = 1 ]; then
      if [ -f "artifacts/pocket_locks/$t.retried" ]; then
        say "$t: FAILED TWICE -> marking dead, will be skipped"
        touch "artifacts/pocket_locks/$t.dead"; dead=$((dead+1))
      else
        say "$t: failing -> releasing claim for one retry"
        touch "artifacts/pocket_locks/$t.retried"
        rm -rf "artifacts/pocket_locks/$t" "artifacts/sweep_out/pocket/pocket_$t"
      fi
    fi
  done
  if [ "$total" -gt 0 ] && [ "$dead" -gt $((total/2)) ]; then
    say "MORE THAN HALF the pocket interfaces are dead ($dead/$total) -- retiring the pocket run"
    mkdir -p artifacts/pocket_jsons_RETIRED
    mv artifacts/pocket_jsons/*_pocket.json artifacts/pocket_jsons_RETIRED/ 2>/dev/null
    say "pocket jsons moved aside; supervisor will now start deep-adaptive scaling"
    break
  fi
  [ "$total" -gt 0 ] && [ "$done_n" -eq "$total" ] && { say "all $total pocket interfaces complete"; break; }
  sleep 300
done
