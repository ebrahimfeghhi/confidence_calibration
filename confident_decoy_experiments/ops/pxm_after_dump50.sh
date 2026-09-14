#!/usr/bin/env bash
# Wait for the dump50 run to finish, then hand all four GPUs back to the PXM-2025 sweep.
#
# The sweep has been stopped since ~13:35 to free GPUs for the 9x1h good-basin hunt. It is resumable
# per seed-chunk (ops/run_pxm2025_chunked.sh claims chunks via atomic mkdir), so nothing generated so
# far is lost and the workers simply pick up the unclaimed chunks.
#
# RELEASES STALE CHUNK CLAIMS FIRST. A worker killed mid-chunk leaves its claim directory behind, and
# an orphaned claim strands that chunk forever -- no other worker will take it. A chunk is considered
# abandoned here if its first seed has fewer than 5 poses on disk.
#
# Prints the dump50 summary on the way past so the run's outcome is recorded in one place: how many
# seeds hit, and which trunk dumps were kept.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
P=$REPO/artifacts/pxm2025
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) waiting for dump50 to finish"
while tmux ls 2>/dev/null | grep -qE '^d[0126]:'; do sleep 60; done

echo "=== $(date +%F\ %H:%M:%S) dump50 finished"
echo "    seeds scored: $(grep -hcE 'max DockQ' artifacts/dump50/driver_g*.log 2>/dev/null | paste -sd+ | bc)/50"
echo "    hits:"
grep -hE '\-> HIT' artifacts/dump50/driver_g*.log 2>/dev/null | sed 's/^/      /'
echo "    trunk dumps kept:"
ls -lh artifacts/dump50/reps 2>/dev/null | awk 'NR>1{print "      "$9"  "$5}'
echo "    quota: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"

echo "=== $(date +%F\ %H:%M:%S) releasing abandoned chunk claims"
for c in $(ls "$P/out/.claims_chunk" 2>/dev/null); do
  pdb=${c%_*}; st=${c##*_}
  n=$(ls "$P/out/$pdb/$pdb/seed_$st/predictions/"*_sample_*.cif 2>/dev/null | wc -l)
  if [ "$n" -lt 5 ]; then
    rmdir "$P/out/.claims_chunk/$c" 2>/dev/null && echo "    released $c ($n poses)"
  fi
done

echo "=== $(date +%F\ %H:%M:%S) restarting PXM workers on GPUs 0 1 2 6"
for g in 0 1 2 6; do
  tmux has-session -t "pxm$g" 2>/dev/null && { echo "    pxm$g already up"; continue; }
  tmux new-session -d -s "pxm$g" -c "$REPO" "GPU=$g bash ops/run_pxm2025_chunked.sh"
  sleep 3
  tmux has-session -t "pxm$g" 2>/dev/null && echo "    pxm$g up" || echo "    FAILED pxm$g"
done
echo "=== $(date +%F\ %H:%M:%S) done; $(tmux ls 2>/dev/null | grep -cE '^pxm[0-9]') pxm workers running"
