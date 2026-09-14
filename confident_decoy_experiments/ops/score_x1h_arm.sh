#!/usr/bin/env bash
# Score one 9x1h template arm with PXMeter's own DockQ, the moment its tmux session ends.
#
# PXMeter rather than OST because 9x1h is a PXM-2025 complex and every other number we have for it
# (the 250-pose baseline, the 5 good seeds, the confident-decoy ipTM) came from PXMeter's scorer
# against artifacts/pxm2025/cif/9x1h.cif. Mixing scorers on one target would make the comparison
# meaningless.
#
# ipTM is read from each pose's OWN summary_confidence json and mapped to the reference chain pair
# through PXMeter's per-pose ref_to_model_chain_mapping -- chain_pair_iptm is indexed by asym order,
# not by reference chain name, so the mapping is never assumed.
#
# WATCH  tmux session to wait on
# RUN    artifacts/sweep_out/<RUN>
# NAME   the job dir inside it (<pdb>_<run>)
# GPU    optional: hand this GPU back to the PXM sweep once scoring is done
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
  tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025.sh"
  sleep 4
  tmux has-session -t "pxm$GPU" 2>/dev/null && echo "  pxm$GPU back up on GPU$GPU" || echo "  FAILED pxm$GPU"
fi
