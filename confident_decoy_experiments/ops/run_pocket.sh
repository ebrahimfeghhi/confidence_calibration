#!/usr/bin/env bash
# PRIORITY 1: pocket-constrained inference on the coverage failures, with the pocket set to the TRUE
# epitope. Oracle experiment -- separates "the model cannot FIND the epitope" from "the model cannot
# BUILD the complex even when told where". If a target fails here, repulsion/steering cannot save it.
# Protenix supports this natively (pocket_embedder enabled in protenix-v2); every run so far logged
# #pocket:0, i.e. the feature has never been exercised in this project.
#
# Work-stealing via atomic mkdir claims, own lock dir. 50 seeds x 5 samples per interface.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=$1
ARM=${ARM:-pocket}
LOCKS=$A/${ARM}_locks
JD=${JD:-$A/pocket_jsons}
NEED=${NEED:-250}
mkdir -p "$LOCKS" "$A/sweep_out/$ARM" "$A/${ARM}_logs"
cd "$REPO" || exit 1
n_poses(){ find "$A/sweep_out/$ARM/${ARM}_$1" -name '*_sample_*.cif' 2>/dev/null | wc -l; }
echo "### POCKET worker GPU$GPU $(date +%H:%M)"
while true; do
  claimed=""
  for f in $(ls $JD/*_pocket.json 2>/dev/null); do
    tag=$(basename "$f" _pocket.json)
    [ "$(n_poses "$tag")" -ge "$NEED" ] && continue
    # self-heal: a lock left behind by a killed worker would otherwise strand its interface forever
    if [ -d "$LOCKS/$tag" ] && [ -n "$(find "$LOCKS/$tag" -maxdepth 0 -mmin +${STALE_MIN:-90} 2>/dev/null)" ]; then
      pgrep -f "sweep_sched.py.*${tag}" >/dev/null 2>&1 || { echo "### reclaiming stale lock $tag"; rmdir "$LOCKS/$tag" 2>/dev/null; }
    fi
    if mkdir "$LOCKS/$tag" 2>/dev/null; then claimed="$tag"; claimed_f="$f"; break; fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU: nothing left, exiting $(date +%H:%M)"; break; }
  echo "### GPU$GPU claimed $claimed $(date +%H:%M)"
  jf=$A/${ARM}_job_${claimed}.json
  $PY - "$claimed" "$claimed_f" "$jf" <<'PYEOF'
import json,sys
tag,src,jf=sys.argv[1],sys.argv[2],sys.argv[3]
pdb=tag.split("_")[0]
json.dump([{"run":f"{__import__('os').environ.get('ARM','pocket')}_{tag}","pdb":pdb,"json":src,"name":pdb,
            "seeds":list(range(101,151))}], open(jf,"w"))
PYEOF
  $PY $REPO/scripts/sweep_sched.py --jobs "$jf" --out "$A/sweep_out/$ARM" \
      --logs "$A/${ARM}_logs" --status "$A/${ARM}_status_g${GPU}.json" --gpus "$GPU" --n-sample 5
  echo "### GPU$GPU finished $claimed poses=$(n_poses "$claimed") $(date +%H:%M)"
  rm -f "$jf"
done
