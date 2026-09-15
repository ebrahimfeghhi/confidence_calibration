#!/usr/bin/env bash
# Drive fast_driver.py over an interface list for ONE arm on ONE GPU, model loaded once per complex.
#
#   MODE=gen2   template on the previous generation's poses (hybrid25 seeds 300025-300049) and fold
#               10 recycles -> new seeds 300050-300074, arm `gen2`
#   MODE=free   plain folds, seeds 300050-300074, arm `free75` -- the budget-matched partner, without
#               which a 75-seed templated pool would be compared against a 50-seed free pool and
#               self-templating would be credited for what is really just more sampling
#
# The interface list is ordered DECOYS FIRST: `nocov` interfaces have no acceptable pose anywhere
# (oracle ~0.06), so nothing can be rescued there and the first five completed overnight moved the
# delta by -0.0000, which is uninformative by construction. Decoys are where the 50-seed comparison
# showed its +0.030.
#
# MEASURED: ~50 s/seed here vs ~200 s for the per-seed-process version, because the 464M checkpoint is
# loaded once per complex instead of once per seed.
#
# Usage: GPU=5 MODE=gen2 IFACES_FILE=ifaces_by_class.txt bash sweep_gen2_fast.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; MODE=${MODE:?}
IFACES_FILE=${IFACES_FILE:-$DC/ifaces_by_class.txt}
SEEDS=${SEEDS:-"$(seq 300050 300074 | tr '\n' ',' | sed 's/,$//')"}
OUTROOT=${OUTROOT:-$REPO/artifacts/${MODE}_fast}
LOG=${LOG:-$REPO/artifacts/sweep_${MODE}.log}
STOP=${STOP:-$REPO/artifacts/SWEEP_${MODE}_STOP}
H1=$REPO/artifacts/hybrid25
rm -f "$STOP"; mkdir -p "$OUTROOT"
echo "[$(date '+%m-%d %H:%M:%S')] START $MODE gpu$GPU $(wc -l < "$IFACES_FILE") interfaces" | tee -a "$LOG"

while read -r IFACE; do
  [ -n "$IFACE" ] || continue
  [ -f "$STOP" ] && { echo "[$(date '+%H:%M:%S')] stop file seen" | tee -a "$LOG"; exit 0; }
  PDB=${IFACE%%_*}
  ARM=$([ "$MODE" = gen2 ] && echo gen2 || echo free75)
  HAVE=$(grep -c "^${IFACE},.*,${ARM}," "$REPO/artifacts/commit11/results.csv" 2>/dev/null); HAVE=${HAVE:-0}
  if [ "$HAVE" -ge 125 ]; then
    echo "[$(date '+%H:%M:%S')]     $IFACE already has $HAVE $ARM rows, skipping" | tee -a "$LOG"; continue
  fi
  echo "[$(date '+%m-%d %H:%M:%S')] >>> $IFACE ($MODE, has $HAVE rows)" | tee -a "$LOG"

  EXTRA=""
  if [ "$MODE" = gen2 ]; then
    [ -d "$H1/$PDB" ] || { echo "     no gen-1 poses for $PDB, skipping" | tee -a "$LOG"; continue; }
    EXTRA="--mode gen2 --tmpl-root $H1/$PDB"
  else
    EXTRA="--mode free --arm-name free75"
  fi

  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold $PY "$DC/fast_driver.py" \
      --pdb "$PDB" --iface "$IFACE" --seeds "$SEEDS" --out "$OUTROOT/$IFACE" \
      --dataset pxm2025 $EXTRA >> "$REPO/artifacts/${MODE}_g$GPU.log" 2>&1 \
      || echo "     !! $IFACE exited nonzero" | tee -a "$LOG"

  N=$(grep -c "^${IFACE},.*,${ARM}," "$REPO/artifacts/commit11/results.csv" 2>/dev/null); N=${N:-0}
  # A complex that recorded nothing is a failure, not a result -- it exits 0 and looks healthy.
  [ "$N" -lt 125 ] && echo "[$(date '+%H:%M:%S')]     !! $IFACE only $N/$ARM rows (expect 125)" | tee -a "$LOG"
  echo "[$(date '+%m-%d %H:%M:%S')]     $IFACE done: $N $ARM rows" | tee -a "$LOG"
  rm -rf "$OUTROOT/$IFACE"      # scores are in the csv; cifs are the bulk
done < "$IFACES_FILE"
echo "[$(date '+%m-%d %H:%M:%S')] $MODE COMPLETE" | tee -a "$LOG"
