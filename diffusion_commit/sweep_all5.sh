#!/usr/bin/env bash
# Drive commit_all5.sh over a list of interfaces: ONE complex at a time, its seeds split across both
# GPUs, wait for both, then the next complex.
#
# Per seed: fold 10 recycles + diffusion (scored as arm `donor` -- THE CONTROLLED BASELINE, DockQ and
# ipTM at recycle 10), then one further recycle resuming that trunk with all five of its own poses as the
# sole template (scored as `all5_k1`). Same seed, same interface, so donor vs all5_k1 is a paired
# comparison with nothing else varying.
#
# GPU HOLDERS ARE LEFT RUNNING BETWEEN PHASES ON PURPOSE. Each seed is a fresh inference process, and
# between them the card sits at 0 MiB during template building and scoring -- which is when a neighbour
# reasonably takes it. The holders (hhold5/hhold6) and gpu_watch.sh are started OUTSIDE this script and
# outlive every phase; nothing here releases them.
#
# Usage: IFACES_FILE=cd89_interfaces.txt GPUS="5 6" NSEEDS=50 bash sweep_all5.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
IFACES_FILE=${IFACES_FILE:-$DC/cd89_interfaces.txt}
GPUS=${GPUS:-"5 6"}
NSEEDS=${NSEEDS:-50}
DATASET=${DATASET:-pxm2025}
# DRIVER=fast loads the 464M checkpoint ONCE per complex instead of once per seed: 90s/seed vs 169s,
# MEASURED on 8vqf. Less than the 3.75x seen on the 7-stage path because only 2 stages run here and the
# fold is 44-54s of real compute, so there is simply less overhead to remove. DRIVER=shell falls back.
DRIVER=${DRIVER:-fast}
FASTOUT=${FASTOUT:-$REPO/artifacts/commit11/fast_all5}
LOG=${LOG:-$REPO/artifacts/all5sweep.log}
PYBIN=/home/ebrahim/Protenix/.venv/bin/python
STOP=${STOP:-$REPO/artifacts/ALL5_STOP}
rm -f "$STOP"
set -- $GPUS; GA=$1; GB=$2

# split 1..NSEEDS by parity so neither GPU gets a systematically different slice
ODD=""; EVEN=""
for s in $(seq 1 "$NSEEDS"); do
  [ $((s % 2)) -eq 1 ] && ODD="$ODD $s" || EVEN="$EVEN $s"
done

echo "[$(date '+%m-%d %H:%M:%S')] START $(wc -l < "$IFACES_FILE") interfaces x $NSEEDS seeds, GPUs $GA/$GB" | tee -a "$LOG"
while read -r IFACE; do
  [ -n "$IFACE" ] || continue
  [ -f "$STOP" ] && { echo "[$(date '+%H:%M:%S')] stop file seen, exiting" | tee -a "$LOG"; exit 0; }
  PDB=${IFACE%%_*}
  echo "[$(date '+%m-%d %H:%M:%S')] >>> $IFACE" | tee -a "$LOG"

  if [ "$DRIVER" = "fast" ]; then
    mkdir -p "$FASTOUT/$IFACE"
    CUDA_VISIBLE_DEVICES=$GA LAYERNORM_TYPE=openfold $PYBIN "$DC/fast_driver.py" \
        --pdb "$PDB" --iface "$IFACE" --seeds "$(echo $ODD | tr ' ' ',')" \
        --out "$FASTOUT/$IFACE" --dataset "$DATASET" --mode all5 \
        >> "$REPO/artifacts/all5_g$GA.log" 2>&1 &
    PA=$!
    sleep 45        # stagger: two template builds in the same second corrupt release_date_cache.json
    CUDA_VISIBLE_DEVICES=$GB LAYERNORM_TYPE=openfold $PYBIN "$DC/fast_driver.py" \
        --pdb "$PDB" --iface "$IFACE" --seeds "$(echo $EVEN | tr ' ' ',')" \
        --out "$FASTOUT/$IFACE" --dataset "$DATASET" --mode all5 \
        >> "$REPO/artifacts/all5_g$GB.log" 2>&1 &
    PB=$!
  else
    GPU=$GA PDB=$PDB IFACE=$IFACE DATASET=$DATASET SEEDS="$ODD"  bash "$DC/commit_all5.sh" \
        >> "$REPO/artifacts/all5_g$GA.log" 2>&1 &
    PA=$!
    sleep 45
    GPU=$GB PDB=$PDB IFACE=$IFACE DATASET=$DATASET SEEDS="$EVEN" bash "$DC/commit_all5.sh" \
        >> "$REPO/artifacts/all5_g$GB.log" 2>&1 &
    PB=$!
  fi
  wait $PA; wait $PB

  N=$(grep -c "^$IFACE,.*,all5_k1," "$REPO/artifacts/commit11/results.csv" 2>/dev/null || echo 0)
  # A complex that recorded nothing is a failure, not a result -- it has happened, and it exits 0.
  if [ "$N" -lt "$NSEEDS" ]; then
    echo "[$(date '+%H:%M:%S')]     !! $IFACE only $N all5 rows for $NSEEDS seeds" | tee -a "$LOG"
  fi
  echo "[$(date '+%m-%d %H:%M:%S')]     $IFACE done: $N all5 rows" | tee -a "$LOG"
done < "$IFACES_FILE"
echo "[$(date '+%m-%d %H:%M:%S')] SWEEP COMPLETE" | tee -a "$LOG"
