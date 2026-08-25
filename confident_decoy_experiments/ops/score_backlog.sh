#!/bin/bash
# Score arms the watcher missed. 8dtn needs the OST identity-chain-map fallback (now in
# score_allpose.py); its auto-mapper crashes on every pose.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
cd $REPO
run() {
    echo "### SCORING $1 $(date +%H:%M)"
    $PY scripts/score_allpose.py --run "$1" --pdb "$2" --ifaces "$3" \
        --donor-cache abag_all_ost --cores 0-47 --workers 8 2>&1 | tail -10
    if [ -s "artifacts/$1_scores.json" ]; then
      n=$($PY -c "import json;print(len(json.load(open('artifacts/$1_scores.json'))['rows']))")
      [ "$n" -gt 0 ] && echo "### OK $1: $n donor rows" || echo "*** $1: scores.json has 0 ROWS ***"
    else
      echo "*** $1: NO scores.json ***"
    fi
}
run allpose8dtn5 8dtn "8dtn_AB:A:B"
echo "### BACKLOG DONE $(date +%H:%M)"
