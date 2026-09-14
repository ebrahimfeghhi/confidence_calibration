#!/usr/bin/env bash
# Same as ops/score_x1h_arm.sh but hands the GPU back to the CHUNKED PXM runner.
#
# WHY A SEPARATE FILE instead of editing the original: the original is already executing inside four
# watcher sessions, and bash reads a script incrementally from a byte offset -- editing it in place
# makes a running interpreter resume in shifted code. That happened earlier today with ops/chain3.sh
# and silently re-ran three arms.
#
# WHY CHUNKED: 12 complexes remain, 66 GPU-h, and 8rvr alone is 16.8 GPU-h. With whole-complex claims
# the 4-GPU makespan is 17.0 h -- entirely set by that one job, so extra GPUs buy nothing at the tail.
# ops/run_pxm2025_chunked.sh claims SEED CHUNKS instead, splitting any complex predicted above
# BIG_SECONDS into 10-seed pieces, so 8rvr becomes five ~3.4 h chunks that four GPUs can share.
# All 12 remaining complexes are above that threshold and none is claimed under the old whole-complex
# scheme, so the chunked runner picks up every one of them.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
WATCH=${WATCH:?}; RUN=${RUN:?}; NAME=${NAME:?}
OUT=artifacts/${RUN}_pxm
mkdir -p "$OUT"

while tmux has-session -t "$WATCH" 2>/dev/null; do sleep 20; done
echo "$(date +%H:%M:%S) $WATCH finished -> scoring $RUN"

score () {
  f=$1
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=artifacts/${RUN}_pxm/${s}_${k}.json
  [ -s "$o" ] || envs/pxmeter/bin/pxm -r artifacts/pxm2025/cif/9x1h.cif -m "$f" -o "$o" >/dev/null 2>&1
}
export -f score; export RUN
find "artifacts/sweep_out/$RUN/$NAME" -name '*_sample_*.cif' 2>/dev/null | sort \
  | xargs -P 12 -I{} bash -c 'score "$@"' _ {}
echo "$(date +%H:%M:%S) $RUN scored: $(ls $OUT/*.json 2>/dev/null | wc -l) poses"

if [ -n "${GPU:-}" ]; then
  tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
  sleep 4
  tmux has-session -t "pxm$GPU" 2>/dev/null && echo "  pxm$GPU up (CHUNKED) on GPU$GPU" \
                                            || echo "  FAILED pxm$GPU"
fi
