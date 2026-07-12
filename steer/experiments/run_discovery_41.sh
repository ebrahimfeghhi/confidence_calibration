#!/usr/bin/env bash
# Leader discovery over the 41 held-out confident-decoy interfaces (Option B: all 5 diffusion
# samples, N ~ 5125). Runs CONC interfaces concurrently, NPROC workers each.
#
# Shared box: NEVER make it unresponsive.
#   - cores : CONC x NPROC = 4 x 32 = 128 of 288
#   - memory: ~26 GB per interface (bounded structure cache) -> ~105 GB peak, of ~890 GB free
#   - each python process runs its own 30s watchdog (flag 80%, kill 90% system memory)
#   - this driver also aborts the whole run if system memory crosses ABORT_PCT
# Resumable: leader_discovery.py skips any interface whose leaders_*.json already exists.
set -u
cd /home/ebrahim/tts_abag
PY=.venv/bin/python
LOGDIR=steer/experiments/logs
mkdir -p "$LOGDIR" results/leader_rect

export NPROC="${NPROC:-32}"
export K="${K:-100}"
CONC="${CONC:-4}"
ABORT_PCT="${ABORT_PCT:-85}"
MAIN="$LOGDIR/discovery_41.log"

IFACES=(8ux6_AF 8ux6_DF 8be1_AC 7yv1_AB 7yv1_AC 7ua2_AC 9jbq_AC 8vzo_AC 8sgn_BC 8sgn_AC
        8vvl_AB 8vvl_AC 8be3_AB 8slb_AB 8slb_AC 8av2_AC 8oxx_AB 8oxx_AC 8w84_AC 8w84_AD
        8w84_BC 8w84_BD 8a67_AD 8dtn_AB 7uvs_AC 7uvs_BC 8aok_AB 8oni_AF 8oni_BF 8oni_BE
        8oni_AE 8uky_AC 8uky_BC 8dcm_AC 8dcm_BC 7zoz_AB 8ivx_AB 8ivx_AC 8q3j_AB 8q3j_AC
        7z1x_AB)

echo "=== $(date '+%F %T') leader discovery: ${#IFACES[@]} interfaces | K=$K | CONC=$CONC x NPROC=$NPROC ===" | tee -a "$MAIN"

sysmem_pct() { awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%.0f", 100*(t-a)/t}' /proc/meminfo; }

done_n=0
for f in "${IFACES[@]}"; do
  # hard stop if the box is under memory pressure — protect other users
  pct=$(sysmem_pct)
  if [ "$pct" -ge "$ABORT_PCT" ]; then
    echo "!!! $(date '+%T') system memory ${pct}% >= ${ABORT_PCT}% — ABORTING before $f" | tee -a "$MAIN"
    break
  fi
  # throttle to CONC concurrent jobs
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done

  if [ -s "results/leader_rect/leaders_${f}.json" ]; then
    echo "$(date '+%T') skip (done): $f" | tee -a "$MAIN"; continue
  fi
  echo "$(date '+%T') start $f   [system mem ${pct}%]" | tee -a "$MAIN"
  $PY steer/experiments/leader_discovery.py "$f" > "$LOGDIR/disc_${f}.log" 2>&1 &
  sleep 5
done

wait
n=$(ls results/leader_rect/leaders_*.json 2>/dev/null | wc -l)
echo "=== $(date '+%F %T') DISCOVERY COMPLETE — $n/${#IFACES[@]} interfaces have leaders ===" | tee -a "$MAIN"
grep -h "LEADER UNION" "$LOGDIR"/disc_*.log 2>/dev/null | tee -a "$MAIN"
