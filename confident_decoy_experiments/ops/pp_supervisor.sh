#!/usr/bin/env bash
# Keep GPUs 0/3/4 busy overnight. Priority 1: the protein-protein sweep. When its claimers exit
# (nothing left unclaimed), fall back to RESUMING the pocket-constrained confident-decoy run, which
# stopped at 19/31 with all poses on disk and a claim-based runner that resumes cleanly.
# Counts only work that is BOTH incomplete AND unclaimed -- counting merely-incomplete work once made
# an idle GPU spawn a claimer that immediately exited, spinning every 25s while the card sat at 0%.
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" >> ops/pp_supervisor.log; }
pp_left(){
  while read -r p _; do
    [ -d "artifacts/pp/locks/$p" ] && continue
    n=$(find "artifacts/pp/out/$p" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -lt 250 ] && { echo 1; return; }
  done < artifacts/pp/subset50.txt
  echo 0
}
busy(){
  for p in $(pgrep -f "runner/inference.py" 2>/dev/null); do
    [ -r /proc/$p/environ ] || continue
    g=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ "$g" = "$1" ] && { echo 1; return; }
  done
  tmux has-session -t "pp_$1" 2>/dev/null && { echo 1; return; }
  tmux has-session -t "cd_$1" 2>/dev/null && { echo 1; return; }
  echo 0
}
say "supervisor armed for GPUs 0 3 4"
while true; do
  for g in 0 3 4; do
    [ "$(busy $g)" = "1" ] && continue
    if [ "$(pp_left)" = "1" ]; then
      say "GPU$g idle -> restarting PP claimer"
      tmux kill-session -t "pp_$g" 2>/dev/null
      tmux new-session -d -s "pp_$g" "cd $REPO && bash ops/run_pp.sh $g 2>&1 | tee -a ops/pp_g${g}.log"
    else
      say "GPU$g idle, PP complete -> resuming pocket-constrained run"
      tmux kill-session -t "cd_$g" 2>/dev/null
      tmux new-session -d -s "cd_$g" "cd $REPO && ARM=pocketcd JD=$REPO/artifacts/pocket_cd_jsons bash ops/run_pocket.sh $g 2>&1 | tee -a ops/pocketcd_g${g}.log"
    fi
    sleep 5
  done
  sleep 30
done
