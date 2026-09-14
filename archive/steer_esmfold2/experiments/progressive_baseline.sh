#!/bin/bash
# progressive_baseline.sh <gpus> [run_name]
#   e.g.  ./progressive_baseline.sh "5 7"
#
# Progressive round-robin baseline sweep over the full FoldBench ab-ag benchmark,
# recipe = mask 0.1 / dropout 0.3 / 5 samples / 16 loops / 16 steps, FUSED kernel.
#
# Reuses the existing esmfold2fast_tts run (96 complexes already at 50 seeds) via
# the runner's skip-existing: at each level N it brings every complex up to N,
# skipping seeds that already exist, so nothing is regenerated. NORMAL complexes
# (L<=1600) run first each level (fast — most are already at 50); the 12 MONSTER
# complexes (L>1600) run after, capped at MONSTER_CAP seeds (they cost 6-10x more
# per fold and none are in the failure set). Stop with Ctrl-C between levels; the
# next launch resumes exactly where it left off.
#
# PRECONDITION: the existing esmfold2fast_tts seeds must be mask0.1/dropout0.3 —
# if that dir is a different recipe, point run_name at a FRESH dir instead (arg 2)
# so recipes are never mixed.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
GPUS=${1:?need gpu list, e.g. "5 7"}
RUN=${2:-esmfold2fast_tts}
MONSTER_CAP=20
FILL_LEVELS="10 20 30 40 50"          # bring the 17 missing up (96 already @50 -> skipped)
EXT_LEVELS="60 70 80 100 150 200"     # extend all normal complexes beyond 50
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_progressive_all.csv"
LOG=steer/experiments/progressive_baseline.log
mkdir -p results/csv_files

# normal (L<=1600, cheap-first) vs monster (L>1600) complex lists, computed live
LISTS=$($PY - <<'PYEOF'
import sys; sys.path.insert(0,"/home/ebrahim/tts_abag"); sys.path.insert(0,"/home/ebrahim/tts_abag/run/esm")
from paths import JSON_DIR; from run_foldbench_abag import load_complexes
Ls={n:sum(len(p.sequence)*(len(p.id) if isinstance(p.id,list) else 1) for p in s.sequences) for n,s in load_complexes(str(JSON_DIR))}
T=1600
print(" ".join(sorted([n for n in Ls if Ls[n]<=T], key=lambda n:Ls[n])))
print(" ".join(sorted([n for n in Ls if Ls[n]> T], key=lambda n:Ls[n])))
PYEOF
)
NORMAL=$(echo "$LISTS" | sed -n 1p)
MONSTER=$(echo "$LISTS" | sed -n 2p)

# background scorer (CPU-only OST DockQ; incremental, skips already-scored)
score_loop(){ while true; do $PY eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" >> steer/experiments/progressive_score.log 2>&1; sleep 900; done; }
score_loop & SP=$!; trap 'kill $SP 2>/dev/null' EXIT

gen(){ # gen "<complexes>" <N>
  local comps="$1" N="$2"; [ -z "$comps" ] && return
  $PY run/esm/run_foldbench_abag_parallel.py --run_name "$RUN" \
      --complexes $comps --seeds $(seq 0 $((N-1))) --gpus $GPUS \
      --lm_mask_pct 0.1 --lm_dropout 0.3 --num_diffusion_samples 5 \
      --chunk_size 32 --fused 2>&1 | tail -2
}

echo "===== [$(date)] START run=$RUN gpus=$GPUS  normal=$(echo $NORMAL|wc -w) monster=$(echo $MONSTER|wc -w) =====" | tee -a "$LOG"
for N in $FILL_LEVELS $EXT_LEVELS; do
  echo "===== [$(date)] LEVEL N=$N  (normal->$N, monster->min($N,$MONSTER_CAP)) =====" | tee -a "$LOG"
  gen "$NORMAL" "$N"                                   # normal first (fast)
  MN=$(( N < MONSTER_CAP ? N : MONSTER_CAP )); gen "$MONSTER" "$MN"   # monsters, capped
  echo "===== [$(date)] LEVEL N=$N done =====" | tee -a "$LOG"
done
kill $SP 2>/dev/null; trap - EXIT
echo "===== [$(date)] ALL LEVELS DONE (run=$RUN) =====" | tee -a "$LOG"
