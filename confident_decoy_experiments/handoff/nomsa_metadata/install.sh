#!/bin/bash
# Install the arm metadata the no-MSA runs need, at the ABSOLUTE paths the template JSONs reference.
# Everything here is small (~185 KB); the 87 GB of run artifacts is NOT needed.
set -euo pipefail
DEST=${DEST:-/home/ebrahim/confident_decoy_experiments/artifacts}
SRC="$(cd "$(dirname "$0")" && pwd)/artifacts"
mkdir -p "$DEST/sweep_tmpl/dockqdonor" "$DEST/sweep_a3m"
cp -n "$SRC"/*.json                        "$DEST/"
cp -n "$SRC"/sweep_tmpl/dockqdonor/*.json  "$DEST/sweep_tmpl/dockqdonor/"
cp -n "$SRC"/sweep_a3m/*.a3m               "$DEST/sweep_a3m/"
echo "installed into $DEST"
echo "  metadata : $(ls "$DEST"/dockq_arm_jobs.json "$DEST"/dockq_arm_donors.json "$DEST"/dockqdonor_scores.json 2>/dev/null | wc -l)/3"
echo "  templates: $(ls "$DEST"/sweep_tmpl/dockqdonor/*.json | wc -l)"
echo "  a3m      : $(ls "$DEST"/sweep_a3m/*.a3m | wc -l)"
echo
echo "verifying every referenced path resolves ..."
python3 - "$DEST" <<'PYEOF'
import json, glob, os, sys
dest = sys.argv[1]; bad = []
for f in glob.glob(os.path.join(dest, "sweep_tmpl", "dockqdonor", "*.json")):
    o = json.load(open(f)); o = o[0] if isinstance(o, list) else o
    for e in o["sequences"]:
        p = e.get("proteinChain")
        if not p: continue
        tp = p.get("templatesPath")
        if tp and not os.path.exists(tp):
            bad.append(tp)
if bad:
    print(f"  MISSING {len(bad)} template a3m, e.g. {bad[:3]}"); sys.exit(1)
print("  all templatesPath entries resolve OK")
PYEOF
echo
echo "NOTE: pairedMsaPath/unpairedMsaPath in these JSONs still point at the origin box's MSA files."
echo "That is fine -- nomsa_donors.py REWRITES both to a 1-sequence a3m it generates locally."
echo "It never reads the originals, which is why the 43 GB MSA prep is not needed."
