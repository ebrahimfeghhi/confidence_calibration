#!/usr/bin/env bash
# Run a list of seeds on one GPU. FOLD_ONLY=1 makes it a free-run sweep (fold + dump, no arms).
#
# Uses the SHELL driver, which is verified: the fold asserts it produced NS poses, and every arm
# asserts that LOAD_TRUNK fired, that exactly one recycle ran, and that the expected number of
# templates was found. The single-process driver is faster but unvalidated; correctness first.
#
# Each seed keeps its trunk dump (z_trunk + s_trunk) so arms can be re-run or extended without
# refolding. Cost scales with N_token^2: ~1.1 GB per seed at 1474 tokens (9x1h), ~110 MB at 462 (8fas).
#
# THE GPU HOLDER IS ON BY DEFAULT (HOLD=0 disables). This used to be opt-in, living only in the
# hybrid25/allpose20 scale scripts, and every new run or smoke test was a fresh chance to forget it.
# On 2026-09-14 the 9x1h batch ran two hours without one and a neighbour took GPU2. The reason it
# matters: this script re-invokes inference once per seed, and between seeds there is a CPU-only gap
# for template building and scoring, during which our card reads 0 MiB and looks free. The holder
# keeps ~10 GiB resident so `nvidia-smi` shows an ebrahim process -- a SIGNAL, not a reservation.
# It deliberately does not reserve the real 25-45 GiB working set; that would be antisocial and would
# make our own inference OOM.
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
cd "$REPO" || exit 1
GPU=${GPU:?}; PDB=${PDB:?}; IFACE=${IFACE:?}; SEEDS=${SEEDS:?}

HOLD=${HOLD:-1}
HOLD_GIB=${HOLD_GIB:-10}
HOLD_STOP=$REPO/artifacts/.hold_stop_gpu$GPU
_we_started_holder=0

if [ "$HOLD" = "1" ]; then
  # Courtesy: never hold a card another user is already computing on.
  other=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$(id -un)" ] && other=1
  done
  if [ "$other" = "1" ]; then
    echo "!! GPU$GPU already has another user's process -- NOT holding, and not launching here"
    exit 1
  fi
  if tmux has-session -t "hhold$GPU" 2>/dev/null; then
    echo "    holder already up on GPU$GPU (left alone -- someone else's to release)"
  else
    rm -f "$HOLD_STOP"
    tmux new-session -d -s "hhold$GPU" -c "$REPO" \
      "CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/scripts/gpu_hold.py --gib $HOLD_GIB --stop $HOLD_STOP"
    _we_started_holder=1
    sleep 5
    tmux has-session -t "hhold$GPU" 2>/dev/null \
      && echo "    holder up on GPU$GPU (${HOLD_GIB} GiB)" \
      || echo "!! holder FAILED to start on GPU$GPU -- continuing unprotected"
  fi
fi

# Release on ANY exit -- normal end, error, or Ctrl-C -- but only a holder we started ourselves.
# A holder started by hand outlives us on purpose, so a sequence of runs can share one claim.
release_holder() {
  [ "$_we_started_holder" = "1" ] || return 0
  touch "$HOLD_STOP"
  echo "    released holder on GPU$GPU"
}
trap release_holder EXIT INT TERM

for s in $SEEDS; do
  D=$REPO/artifacts/commit11/${IFACE}_s${s}
  # FOLD_ONLY sweeps are done when the trunk exists; full units are done when the last arm exists.
  if [ -n "${FOLD_ONLY:-}" ]; then
    [ -s "$D/trunk.npz" ] && { echo "=== seed $s already folded, skipping"; continue; }
  else
    [ -d "$D/p4" ] && { echo "=== seed $s already complete, skipping"; continue; }
  fi
  echo "=== $(date +%H:%M:%S) GPU$GPU seed $s"
  GPU=$GPU PDB=$PDB IFACE=$IFACE SEED=$s bash "$DC/fold_and_commit.sh"
done
echo "=== $(date +%H:%M:%S) GPU$GPU: all seeds done"
