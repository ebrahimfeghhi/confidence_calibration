#!/bin/bash
# progressive_baseline_queued.sh [budget_gpu_hours]
#
# Per-GPU INDEPENDENT: each of GPUs 1,7,3 starts the moment IT frees (no waiting on the
# others) and runs the mask0.1/dropout0.3 FUSED baseline at seeds 50+ on its own
# cost-balanced share of the 113 complexes, 10 seeds per round. Stops when total GPU-time
# reaches BUDGET_GPU_HOURS (default 100). Early-stop any time: touch prog_state/STOP, or kill.
#
# Leaves existing seeds 0-49 untouched (skip-existing); writes new seeds at indices 50+ in
# the same esmfold2fast_tts dir so the recipe can be verified later by comparing
# DockQ(0-49) vs DockQ(50+) per complex. Per-sample it saves (via the runner): the CIF,
# ranking_score/iptm/ptm/chain_pair_iptm (=pair_iptm), and plddt + pae (.npy).
# Monsters (L>1600) capped at MONSTER_CAP new seeds. Note: per-GPU sharding means complexes
# on a late-freeing GPU end with fewer seeds (fine — every curve uses its own seeds).
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RUN=esmfold2fast_tts
GPUS="1 7 3"
SEED_START=50; INCREMENT=10; MAX_SEED=400; MONSTER_CAP=20
BUDGET_GPU_HOURS=${1:-100}
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_verify.csv"     # whole dir (0-49 + 50+); split by seed
LOG=steer/experiments/progressive_baseline_queued.log
STATE=steer/experiments/prog_state
rm -rf "$STATE"; mkdir -p "$STATE" results/csv_files; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# cost-balanced structure shards per GPU (fold-time ~ L^2.8; monsters weighted by their cap)
$PY - "$STATE" "$MONSTER_CAP" $GPUS <<'PYEOF'
import sys
sys.path.insert(0,"/home/ebrahim/tts_abag"); sys.path.insert(0,"/home/ebrahim/tts_abag/run/esm")
from paths import JSON_DIR; from run_foldbench_abag import load_complexes
state=sys.argv[1]; cap=int(sys.argv[2]); gpus=sys.argv[3:]
Ls={n:sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences) for n,s in load_complexes(str(JSON_DIR))}
T=1600; NOM=100
cost=lambda n:(Ls[n]/642)**2.8
weight=lambda n:cost(n)*(cap if Ls[n]>T else NOM)      # total-work estimate over the run
load={g:0.0 for g in gpus}; shard={g:[] for g in gpus}
for n in sorted(Ls, key=lambda n:-weight(n)):          # greedy: heaviest -> least-loaded gpu
    g=min(load, key=load.get); shard[g].append(n); load[g]+=weight(n)
mx=max(load.values())
for g in gpus:
    norm=sorted([n for n in shard[g] if Ls[n]<=T], key=lambda n:Ls[n])
    mon =sorted([n for n in shard[g] if Ls[n]> T], key=lambda n:Ls[n])
    open(f"{state}/normal_{g}.txt","w").write(" ".join(norm))
    open(f"{state}/monster_{g}.txt","w").write(" ".join(mon))
    print(f"GPU {g}: {len(norm)} normal + {len(mon)} monster (rel load {load[g]/mx:.2f})")
PYEOF

wait_for_gpu(){ local g=$1
  while :; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 2000 ] && break
    log "waiting for GPU $g (${u:-?}MiB)"; sleep 60
  done; log "GPU $g free -> starting"; }

worker(){ local g=$1
  wait_for_gpu "$g"; date +%s > "$STATE/start_$g"
  local norm mon k=0; norm=$(cat "$STATE/normal_$g.txt"); mon=$(cat "$STATE/monster_$g.txt")
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
  date +%s > "$STATE/done_$g"; log "GPU $g worker exited"; }

log "recipe=mask0.1/dropout0.3 FUSED seeds ${SEED_START}+  gpus=[$GPUS]  budget=${BUDGET_GPU_HOURS} GPU-h  (per-GPU independent start)"
# record recipe provenance once (workers use --no_config_log to avoid a config.json race)
$PY - "$PRED_DIR" <<'PYEOF'
import json,sys,os
d=sys.argv[1]; os.makedirs(d,exist_ok=True); p=os.path.join(d,"config.json")
h=json.load(open(p)) if os.path.exists(p) else []
h=h if isinstance(h,list) else [h]
h.append({"note":"seeds 50+ by progressive_baseline_queued.sh","lm_mask_pct":0.1,"lm_dropout":0.3,
          "num_diffusion_samples":5,"num_loops":16,"num_sampling_steps":16,"fused":True})
json.dump(h,open(p,"w"),indent=2)
PYEOF

# background OST scorer (CPU; incremental)
( while :; do $PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >>"$STATE/score.log" 2>&1; sleep 900; done ) &
SP=$!; trap 'kill $SP 2>/dev/null; touch "$STATE/STOP"' EXIT

pids=""
for g in $GPUS; do worker "$g" & pids="$pids $!"; done

# watchdog: sum per-GPU active time -> STOP at budget
BUDGET_SEC=$((BUDGET_GPU_HOURS*3600)); total=0
while :; do
  sleep 120; now=$(date +%s); total=0
  for g in $GPUS; do
    [ -f "$STATE/start_$g" ] || continue
    st=$(cat "$STATE/start_$g"); en=$now; [ -f "$STATE/done_$g" ] && en=$(cat "$STATE/done_$g")
    total=$((total + en - st))
  done
  log "GPU-hours: $(awk "BEGIN{printf \"%.1f\",$total/3600}") / $BUDGET_GPU_HOURS"
  [ "$total" -ge "$BUDGET_SEC" ] && { log "budget reached -> STOP"; touch "$STATE/STOP"; break; }
  running=0; for p in $pids; do kill -0 "$p" 2>/dev/null && running=1; done
  [ "$running" -eq 0 ] && { log "all workers finished"; break; }
done
wait; kill $SP 2>/dev/null; trap - EXIT
$PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >>"$STATE/score.log" 2>&1
log "ALL DONE. ~$(awk "BEGIN{printf \"%.1f\",$total/3600}") GPU-hours. Scored -> $OUT_CSV"
