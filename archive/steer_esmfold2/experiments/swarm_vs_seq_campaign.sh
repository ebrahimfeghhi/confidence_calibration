#!/bin/bash
# swarm_vs_seq_campaign.sh — head-to-head on hard targets: SWARM (gen_cloud) vs SEQUENTIAL, 3 trials each.
# Queue-based: 4 GPU workers pull "method complex" jobs from a shared queue (flock-atomic) until drained + NOMORE.
# Append more jobs any time:  echo "swarm 9en2" >> steer/experiments/campaign_state/queue.txt
# Signal no-more-coming:       touch steer/experiments/campaign_state/NOMORE
# Hard stop:                   touch steer/experiments/campaign_state/STOP
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
GPUS="3 4 5 7"
STATE=steer/experiments/campaign_state; mkdir -p "$STATE"
QUEUE="$STATE/queue.txt"; LOCK="$STATE/queue.lock"
LOG="$STATE/campaign.log"; : > "$LOG"
rm -f "$STATE/STOP" "$STATE/NOMORE"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# seed the queue with the 3 ready complexes x 2 methods (9en2/8ahn appended after reps are built)
cat > "$QUEUE" <<'EOF'
swarm 8jyr
seq 8jyr
swarm 8z8v
seq 8z8v
swarm 8im0
seq 8im0
EOF

recipe(){ local m=$1 cx=$2
  if [ "$m" = swarm ]; then
    echo "--target $cx --lever slm --gen-cloud 16 --M 5 --K 3 --lr 0.3 --init-frac 0.05 --bptt 0 --trials 3 --runs 12 --no-score --fused --chunk-size 32 --n-samples 5 --tag ${cx}_swarm"
  else
    echo "--target $cx --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean --bptt 0 --trials 3 --runs 32 --cycle-decode 16 --no-score --fused --chunk-size 32 --n-samples 5 --tag ${cx}_seq"
  fi
}

pop(){ exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && sed -i '1d' "$QUEUE"
  flock -u 9; echo "$job"; }

worker(){ local g=$1 job m cx
  while :; do
    [ -f "$STATE/STOP" ] && { log "GPU$g: STOP"; break; }
    job=$(pop)
    if [ -z "$job" ]; then
      [ -f "$STATE/NOMORE" ] && { log "GPU$g: queue drained, done"; break; }
      sleep 30; continue
    fi
    m=$(echo "$job" | awk '{print $1}'); cx=$(echo "$job" | awk '{print $2}')
    log "GPU$g START $m $cx"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $PY \
      steer/experiments/repel_proven.py $(recipe "$m" "$cx") > "$STATE/log_${cx}_${m}.out" 2>&1
    log "GPU$g DONE $m $cx (exit $?)"
  done; }

log "campaign start on GPUs [$GPUS]; 6 ready jobs queued (8jyr,8z8v,8im0 x swarm+seq)"
for g in $GPUS; do worker "$g" & done
wait
log "CAMPAIGN COMPLETE"
