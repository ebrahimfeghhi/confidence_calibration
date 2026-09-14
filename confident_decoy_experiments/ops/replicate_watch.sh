#!/usr/bin/env bash
# Poll the four 9x1h replications once a minute and announce the first hit loudly.
#
# The workers each print one "seed NNNNNN: max DockQ X -> HIT|miss" line per seed into their own
# driver log. This aggregates those across all four, writes a one-line status every minute, and on
# the first HIT writes a HIT_FOUND banner file that is trivial to spot without reading four logs.
#
# It does NOT stop anything: ops/replicate_nostop.sh removes the STOP sentinel so the surviving
# workers keep drawing seeds. The worker that hit exits on its own (it kills its generator in the
# same step that touches STOP) and cannot be saved without editing a running script.
#
# The running tally is the point: hits/seed-runs is a direct estimate of the good-basin rate under
# the sweep's exact configuration, to be compared against the sweep's own 5/50 and against the 39
# seed-runs at zero that this experiment was built to explain.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
R=$REPO/artifacts/replicate
LOG=$R/watch.log
BANNER=$R/HIT_FOUND
announced=0
while :; do
  n=0; h=0; det=""
  for g in 0 1 2 6; do
    f=$R/driver_g$g.log
    # NOTE: grep -c already prints 0 when nothing matches (and exits 1), so a `|| echo 0` here
    # appends a SECOND zero and the arithmetic below dies. Take the output as-is.
    s=$(grep -cE 'max DockQ' "$f" 2>/dev/null); s=${s:-0}
    hh=$(grep -cE '\-> HIT' "$f" 2>/dev/null); hh=${hh:-0}
    n=$((n+s)); h=$((h+hh))
    alive=$(tmux has-session -t "rep$g" 2>/dev/null && echo up || echo down)
    det="$det g$g=$s/$alive"
  done
  echo "[$(date +%H:%M:%S)] seed-runs=$n hits=$h  |$det" >> "$LOG"
  if [ "$h" -gt 0 ] && [ "$announced" -eq 0 ]; then
    announced=1
    {
      echo "HIT at $(date +%F\ %H:%M:%S) after $n seed-runs across the four replications"
      grep -hE '\-> HIT' "$R"/driver_g*.log 2>/dev/null
      echo "NOTE: this arm runs WITHOUT DUMP_FULL, so the winning seed has poses but NO trunk dump."
    } > "$BANNER"
    echo "[$(date +%H:%M:%S)] *** HIT *** see $BANNER" >> "$LOG"
  fi
  tmux ls 2>/dev/null | grep -qE '^rep[0126]:' || {
    echo "[$(date +%H:%M:%S)] all replications finished; final seed-runs=$n hits=$h" >> "$LOG"; break; }
  sleep 60
done
