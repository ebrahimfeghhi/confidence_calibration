#!/bin/bash
# deepen27.sh — collect 1024 baseline seeds (mask0.1/dropout0.3, fused, 5 diffusion samples) for the
# 27 SEVERE selection-failure complexes (gap>0.30 in iptm_selection_gap_baseline.csv), cost-balanced
# across GPUs 3 4 5 7, skip-existing (never refolds a seed already on disk). Model loads once/GPU.
# No distograms saved. Off switch: kill the workers (touch deepen27_state/STOP just records intent).
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RUN=esmfold2fast_tts
GPUS="3 4 5 7"
MAXSEED=1023
STATE=steer/experiments/deepen27_state; rm -rf "$STATE"; mkdir -p "$STATE"
LOG="$STATE/deepen27.log"; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# cost-balanced shard of the 27 severe complexes across the GPUs (weight = remaining-seeds * L^2.8)
$PY - "$STATE" $GPUS <<'PYEOF'
import sys,csv,os
sys.path.insert(0,"/home/ebrahim/tts_abag"); sys.path.insert(0,"/home/ebrahim/tts_abag/run/esm")
from paths import JSON_DIR; from run_foldbench_abag import load_complexes
state=sys.argv[1]; gpus=sys.argv[2:]
comps=dict(load_complexes(str(JSON_DIR)))
def L(s): return sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences)
PRED=os.path.expanduser("~/esm/foldbench_predictions/esmfold2fast_tts")
def have(cx):
    d=os.path.join(PRED,cx); return sum(1 for x in os.listdir(d) if x.startswith("seed_") and x[5:].isdigit()) if os.path.isdir(d) else 0
sev=[r["complex"] for r in csv.DictReader(open("results/csv_files/iptm_selection_gap_baseline.csv"))
     if int(r["selection_fail"]) and float(r["gap"])>0.30]
def Lof(cx):
    k=[n for n in comps if n.startswith(cx)]; return L(comps[k[0]]) if k else 642
weight=lambda cx: max(0,1024-have(cx))*((Lof(cx)/642)**2.8)
load={g:0.0 for g in gpus}; shard={g:[] for g in gpus}
for cx in sorted(sev,key=lambda c:-weight(c)):
    g=min(load,key=load.get); shard[g].append(cx); load[g]+=weight(cx)
tot=sum(max(0,1024-have(c)) for c in sev)
print(f"27 severe complexes, {tot} seeds to fold")
for g in gpus:
    open(f"{state}/cx_{g}.txt","w").write(" ".join(shard[g]))
    print(f"  GPU {g}: {shard[g]}")
PYEOF

worker(){ local g=$1 cxs; cxs=$(cat "$STATE/cx_$g.txt")
  [ -z "$cxs" ] && { log "GPU $g: empty shard"; return; }
  log "GPU $g START: $cxs"
  CUDA_VISIBLE_DEVICES="$g" $PY run/esm/run_foldbench_abag.py --run_name "$RUN" \
    --complexes $cxs --seeds $(seq 0 $MAXSEED) --lm_mask_pct 0.1 --lm_dropout 0.3 \
    --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$STATE/gpu_$g.log" 2>&1
  log "GPU $g DONE (exit $?)"; }

for g in $GPUS; do worker "$g" & done
wait
log "DEEPEN27 COMPLETE"
