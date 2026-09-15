#!/usr/bin/env bash
# GENERATION 2. For each interface, template on the gen-1 output (hybrid25 seeds 300025-300049) and
# generate 25 new seeds at donor+25, i.e. 300050-300074. One gen-1 seed's 5 poses -> one gen-2 seed,
# mirroring how gen-1 was built from the free seeds.
#
# WHY. hybrid25 at matched budget (50 free vs 25 free + 25 templated, n=278) moved decoys +0.030 and
# solved -0.002, with a MEDIAN delta of exactly 0.000 -- a minority of large rescues, not a broad shift.
# Whether a second generation compounds that or saturates is the open question; on 9x1h a gen-2 held
# DockQ flat (0.826 -> 0.825) while ipTM rose 0.629 -> 0.735, but that was one seed of one interface.
#
# The paired free arm (gen2_free.sh) deepens the free pool to 75 seeds so the comparison stays
# budget-matched: free-75 vs (25 free + 25 gen-1 + 25 gen-2).
#
# Usage: GPU=5 IFACES_FILE=ifaces291.txt bash gen2_sweep.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
IFACES_FILE=${IFACES_FILE:-$DC/ifaces291.txt}
H1=$REPO/artifacts/hybrid25
P=$REPO/artifacts/pxm2025
CSV=$REPO/artifacts/commit11/results.csv
OUT=${OUT:-$REPO/artifacts/gen2}
LOG=${LOG:-$REPO/artifacts/gen2_sweep.log}
STOP=${STOP:-$REPO/artifacts/GEN2_STOP}
NS=${NS:-5}; NCYCLE=${NCYCLE:-10}
rm -f "$STOP"; mkdir -p "$OUT"; cd "$REPO" || exit 1
echo "[$(date '+%m-%d %H:%M:%S')] GEN2 START gpu$GPU $(wc -l < "$IFACES_FILE") interfaces" | tee -a "$LOG"

while read -r IFACE; do
  [ -n "$IFACE" ] || continue
  [ -f "$STOP" ] && { echo "[$(date '+%H:%M:%S')] stop file seen" | tee -a "$LOG"; exit 0; }
  PDB=${IFACE%%_*}
  JOB=$P/jsons/${PDB}-final-updated.json
  REF=$P/cif/$PDB.cif
  [ -s "$JOB" ] && [ -s "$REF" ] || { echo "  !! $IFACE missing inputs"; continue; }
  echo "[$(date '+%m-%d %H:%M:%S')] >>> $IFACE" | tee -a "$LOG"

  for D1 in $(ls -d "$H1/$PDB"/s* 2>/dev/null | sort); do
    DSEED=$(basename "$D1"); DSEED=${DSEED#s}          # the gen-1 donor (a FREE seed, 300000-300024)
    G1DIR=$(find "$D1/out" -type d -name predictions 2>/dev/null | head -1)
    [ -n "$G1DIR" ] || continue
    G1SEED=$(basename "$(dirname "$G1DIR")"); G1SEED=${G1SEED#seed_}   # the gen-1 output seed
    NEW=$((G1SEED + 25))                                # donor + 25  ->  300050-300074
    [ "$(ls "$G1DIR"/*_sample_*.cif 2>/dev/null | wc -l)" -ge "$NS" ] || continue
    grep -q "^${IFACE},${NEW},gen2," "$CSV" 2>/dev/null && continue

    W=$OUT/$IFACE/s$NEW
    mkdir -p "$W/out"
    if [ ! -s "$W/tmpl/job.json" ]; then
      TIDS=""; POSES=""
      for k in $(seq 0 $((NS-1))); do
        f=$(ls "$G1DIR"/*_sample_${k}.cif 2>/dev/null | head -1)
        [ -s "$f" ] || continue
        POSES="$POSES $f"; TIDS="$TIDS $(printf 'g%01d%02d' "$k" $((NEW % 100)))"
      done
      SELFTMPL_JOBS=$P/jsons $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$W/tmpl" \
        --poses $POSES --tids $TIDS >> "$W/build.log" 2>&1
    fi
    [ -s "$W/tmpl/job.json" ] || { echo "  !! $IFACE s$NEW template build failed"; continue; }

    cd /home/ebrahim/Protenix || exit 1
    env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
        PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 \
        PROTENIX_MAX_TEMPLATES=$NS PROTENIX_MAX_TEMPLATE_CANDIDATES=20 \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$W/tmpl/job.json" --dump_dir "$W/out" --seeds "$NEW" \
      --model.N_cycle $NCYCLE --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
      --use_template true > "$W/run.log" 2>&1
    cd "$REPO" || exit 1

    # assert: NS templates found, and NS structures written. Both have failed silently before, and a
    # run that found 0 templates still produces clean-looking numbers.
    T=$(grep -oE 'Found [0-9]+ templates' "$W/run.log" | grep -oE '[0-9]+' | sort -u | tail -1)
    N=$(find "$W/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    if [ "$N" -ne "$NS" ] || [ "${T:-0}" -lt "$NS" ]; then
      echo "  !! $IFACE s$NEW: $N/$NS poses, templates=${T:-0} -- not scoring"
      grep -m1 -E "ERROR|Error" "$W/run.log" 2>/dev/null | cut -c1-130 | sed 's/^/     /'
      rm -rf "$W/out"; continue
    fi
    $PY "$DC/row.py" --dir "$W" --iface "$IFACE" --seed "$NEW" --arm gen2 \
        --donor "$G1SEED" --ref "$REF" --scorer pxm >> "$CSV"
    rm -rf "$W/out"      # scores are in the csv; 291 x 25 x 5 cifs would be ~100 GB
  done
  echo "[$(date '+%m-%d %H:%M:%S')]     $IFACE: $(grep -c "^$IFACE,.*,gen2," "$CSV" 2>/dev/null) gen2 rows" | tee -a "$LOG"
done < "$IFACES_FILE"
echo "[$(date '+%m-%d %H:%M:%S')] GEN2 COMPLETE" | tee -a "$LOG"
