#!/bin/bash
# deepen_to_1024.sh — bring every confident-decoy-set complex that is under 1024 seeds UP TO 1024,
# using the exact pool settings (mask0.1/dropout0.3, 5 diffusion samples, fused, chunk_size 32),
# skip-existing (never refolds a seed already on disk). Cost-balanced across GPUs 3 4 5 7 by
# remaining-seeds * L^2.8, launched into their tmux sessions. See docs/esmfold2fast_tts_benchmark.md.
#   bash steer/experiments/deepen_to_1024.sh        # prepare + print the shard (no launch)
#   bash steer/experiments/deepen_to_1024.sh go     # launch into tmux gpu3/4/5/7
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
RF=run/esm/run_foldbench_abag.py
RUN=esmfold2fast_tts
TARGET=1024
GPUS="3 4 5 7"
PRED=$HOME/esm/foldbench_predictions/$RUN
STATE=steer/experiments/deepen1024_state; rm -rf "$STATE"; mkdir -p "$STATE"

# under-target complexes from the confident-decoy set + cost-balanced shard across GPUs
$PY - "$STATE" "$PRED" "$TARGET" $GPUS <<'PYEOF'
import sys,os,csv
sys.path.insert(0,"/home/ebrahim/tts_abag"); sys.path.insert(0,"/home/ebrahim/tts_abag/run/esm")
from run_foldbench_abag import load_complexes
from paths import JSON_DIR
state,pred,target=sys.argv[1],sys.argv[2],int(sys.argv[3]); gpus=sys.argv[4:]
struct=sorted(set(r["cx"][:4] for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv"))))
comps=dict(load_complexes(str(JSON_DIR)))
def L(s): return sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences)
def folds(cx):
    cand=[d for d in os.listdir(pred) if d[:4]==cx] if os.path.isdir(pred) else []
    if not cand: return 0
    d=os.path.join(pred,cand[0]); return sum(1 for x in os.listdir(d) if x.startswith("seed_") and x[5:].isdigit())
def Lof(cx):
    k=[m for m in comps if m.startswith(cx)]; return L(comps[k[0]]) if k else 642
need={cx:target-folds(cx) for cx in struct if folds(cx)<target}
weight=lambda cx: need[cx]*((Lof(cx)/642)**2.8)
load={g:0.0 for g in gpus}; shard={g:[] for g in gpus}
for cx in sorted(need,key=lambda c:-weight(c)):
    g=min(load,key=load.get); shard[g].append(cx); load[g]+=weight(cx)
print(f"{len(need)} complexes under {target}; total seeds to fold = {sum(need.values())}")
for g in gpus:
    open(f"{state}/cx_{g}.txt","w").write(" ".join(shard[g]))
    print(f"  GPU {g}: " + ", ".join(f"{c}(+{need[c]})" for c in shard[g]))
PYEOF

for g in $GPUS; do
  cxs=$(cat "$STATE/cx_$g.txt" 2>/dev/null)
  w=$STATE/worker_g$g.sh
  {
    echo "#!/bin/bash"; echo "cd /home/ebrahim/tts_abag"
    echo "echo \"[deepen1024 g$g] START \$(date): $cxs\""
    if [ -n "$cxs" ]; then
      echo "CUDA_VISIBLE_DEVICES=$g $PY $RF --run_name $RUN --complexes $cxs \\"
      echo "  --seeds \$(seq 0 $((TARGET-1))) --lm_mask_pct 0.1 --lm_dropout 0.3 \\"
      echo "  --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >> $STATE/g$g.log 2>&1"
    fi
    echo "echo \"[deepen1024 g$g] DONE \$(date) exit \$?\""
  } > "$w"; chmod +x "$w"
done
echo "workers written to $STATE/worker_g*.sh"

if [ "${1:-}" = "go" ]; then
  for g in $GPUS; do
    [ -s "$STATE/cx_$g.txt" ] || { echo "  gpu$g empty shard, skip"; continue; }
    tmux send-keys -t gpu$g "cd /home/ebrahim/tts_abag && bash $STATE/worker_g$g.sh" Enter
    echo "  dispatched worker_g$g -> tmux gpu$g"
  done
  echo "LAUNCHED deepen-to-1024. Logs: $STATE/g{3,4,5,7}.log"
else
  echo "PREPARE-ONLY. To launch: bash steer/experiments/deepen_to_1024.sh go"
fi
