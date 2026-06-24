#!/bin/bash
# Launch the ConforNet experiment on a NEW server from the portable bundle.
# Points every path at confornet/data/ (the rsync'd bundle), then runs the
# single-GPU wait-and-run launcher. Edit ESM_PYTHON / OST_* for this host.
#
# Usage:  bash confornet/run_newserver.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)            # .../tts_abag/confornet
BUNDLE="$HERE/data"

# --- the esm venv on THIS host (has esm + patched transformers + torch) ---
export ESM_PYTHON="${ESM_PYTHON:-$HOME/esm/.venv/bin/python}"

# --- point tts_abag's paths.py at the bundle (all env-overridable) ---
export CONFORNET_DATA="$BUNDLE"
export ESM_PREDICTIONS="$BUNDLE/baseline"      # resolves forward_grad.BASELINE
export JSON_DIR="$BUNDLE/jsons"                # build_conditioning input
export GROUND_TRUTHS="$BUNDLE/ground_truths"   # DockQ reference
export TARGETS_CSV="$BUNDLE/targets/interface_antibody_antigen.csv"

# --- OST for the decode->DockQ step (set to THIS host's foldbench conda env) ---
export OST_BIN="${OST_BIN:-$HOME/miniconda3/envs/foldbench/bin/ost}"
export OST_LIB="${OST_LIB:-$HOME/miniconda3/envs/foldbench/lib}"

echo "[run_newserver] bundle=$BUNDLE  esm_python=$ESM_PYTHON"
"$ESM_PYTHON" "$HERE/../paths.py" | grep -E 'CONFORNET_DATA|ESM_PREDICTIONS|JSON_DIR|GROUND_TRUTHS|TARGETS_CSV'

# confornet_one.sh waits for ONE fully-idle GPU and runs steer_confornet on it.
# It uses its own PY=.../esm/.venv/bin/python; override there if your venv differs.
exec bash "$HERE/confornet_one.sh"
