#!/bin/bash
# baseline_continuous.sh [gpus]
# Keeps GPUs busy generating mask0.1/dropout0.3 baseline seeds indefinitely.
#  1) waits for the current progressive_baseline_queued.sh (budget-capped) to finish,
#  2) then takes over the given GPUs (default "3 7") and runs progressive seed rounds
#     (skip-existing) on a cost-balanced shard, with NO GPU-hour cap,
#  3) until steer/experiments/basecont_state/STOP is created (that's the off switch).
# Also keeps the incremental OST scorer running. Off switch: touch basecont_state/STOP
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RUN=esmfold2fast_tts
GPUS=${1:-"3 7"}
SEED_START=50; INCREMENT=10; MAX_SEED=1000; MONSTER_CAP=0   # 0 = SKIP monsters (L>1600): they hit CUDA illegal-memory near the VRAM ceiling
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_verify.csv"
STATE=steer/experiments/basecont_state; rm -rf "$STATE"; mkdir -p "$STATE" results/csv_files
LOG=steer/experiments/baseline_continuous.log; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 1) wait for the current budget-capped sweep to finish (so we don't double-up on 3/7)
log "armed: waiting for progressive_baseline_queued.sh to finish before taking over [$GPUS]"
while pgrep -f "progressive_baseline_queued.sh" >/dev/null; do
  [ -f "$STATE/STOP" ] && { log "STOP set before takeover — exiting"; exit 0; }
  sleep 120
done
log "current sweep finished -> taking over GPUs [$GPUS], no budget cap, until STOP"

# 2) cost-balanced shard across GPUS (fold-time ~ L^2.8; monsters weighted by cap)
$PY - "$STATE" "$MONSTER_CAP" $GPUS <<'PYEOF'
import sys
sys.path.insert(0,"/home/ebrahim/tts_abag"); sys.path.insert(0,"/home/ebrahim/tts_abag/run/esm")
from paths import JSON_DIR; from run_foldbench_abag import load_complexes
state=sys.argv[1]; cap=int(sys.argv[2]); gpus=sys.argv[3:]
Ls={n:sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences) for n,s in load_complexes(str(JSON_DIR))}
T=1600; NOM=100
cost=lambda n:(Ls[n]/642)**2.8
weight=lambda n:cost(n)*(cap if Ls[n]>T else NOM)
load={g:0.0 for g in gpus}; shard={g:[] for g in gpus}
for n in sorted(Ls, key=lambda n:-weight(n)):
    g=min(load, key=load.get); shard[g].append(n); load[g]+=weight(n)
for g in gpus:
    norm=sorted([n for n in shard[g] if Ls[n]<=T], key=lambda n:Ls[n])
    mon =sorted([n for n in shard[g] if Ls[n]> T], key=lambda n:Ls[n])
    open(f"{state}/normal_{g}.txt","w").write(" ".join(norm))
    open(f"{state}/monster_{g}.txt","w").write(" ".join(mon))
    print(f"GPU {g}: {len(norm)} normal + {len(mon)} monster")
PYEOF

# 3) per-GPU worker: progressive seed rounds until STOP (skip-existing fast-forwards done seeds)
worker(){ local g=$1 norm mon k=0
  norm=$(cat "$STATE/normal_$g.txt"); mon=$(cat "$STATE/monster_$g.txt")
  while :; do
    [ -f "$STATE/STOP" ] && break
    local s0=$((SEED_START+INCREMENT*k)) s1=$((SEED_START+INCREMENT*k+INCREMENT-1))
    [ "$s1" -gt "$MAX_SEED" ] && break
    [ -n "$norm" ] && CUDA_VISIBLE_DEVICES="$g" $PY run/esm/run_foldbench_abag.py --run_name "$RUN" \
        --complexes $norm --seeds $(seq "$s0" "$s1") --lm_mask_pct 0.1 --lm_dropout 0.3 \
        --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$STATE/gpu_$g.log" 2>&1
    if [ -n "$mon" ] && [ "$s1" -lt "$((SEED_START+MONSTER_CAP))" ]; then
      CUDA_VISIBLE_DEVICES="$g" $PY run/esm/run_foldbench_abag.py --run_name "$RUN" \
        --complexes $mon --seeds $(seq "$s0" "$s1") --lm_mask_pct 0.1 --lm_dropout 0.3 \
        --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$STATE/gpu_$g.log" 2>&1
    fi
    log "GPU $g finished seeds $s0-$s1"; k=$((k+1))
  done
  log "GPU $g worker exited"; }

# incremental OST scorer (self-stops on STOP)
( while [ ! -f "$STATE/STOP" ]; do $PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >>"$STATE/score.log" 2>&1; sleep 900; done ) &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT

pids=""; for g in $GPUS; do worker "$g" & pids="$pids $!"; done
wait $pids
kill $SP 2>/dev/null; trap - EXIT
log "baseline_continuous DONE (STOP set or MAX_SEED reached)"
