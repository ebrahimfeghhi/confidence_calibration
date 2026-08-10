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
echo "merging release dates for the injected donor templates ..."
# THE FAILURE THIS EXISTS TO PREVENT: Protenix filters every template hit through a release-date
# lookup (template_utils.py:369). A donor mmCIF that is present but absent from
# release_date_cache.json is rejected as date-unknown and SILENTLY skipped -- the run logs
# `Found 0 templates` and folds with no template while looking completely healthy. This cost a
# second machine ~50 min of 2 GPUs on 2026-08-10 because the first version of this bundle shipped
# the mmCIFs and a3ms but not these entries.
python3 - "$(cd "$(dirname "$0")" && pwd)" <<'PYEOF2'
import json, os, sys
pkg = sys.argv[1]
root = os.environ.get("PROTENIX_ROOT_DIR", os.path.expanduser("~"))
f = os.path.join(root, "common", "release_date_cache.json")
os.makedirs(os.path.dirname(f), exist_ok=True)
cache = json.load(open(f)) if os.path.exists(f) else {}
new = json.load(open(os.path.join(pkg, "release_date_entries.json")))
added = [k for k in new if k not in cache]
cache.update(new)
json.dump(cache, open(f, "w"))
print(f"  release_date_cache: +{len(added)} entries ({len(cache)} total) -> {f}")
missing_cif = [k for k in new if not os.path.exists(os.path.join(root, "mmcif", k + ".cif"))]
if missing_cif:
    print(f"  WARNING: {len(missing_cif)} donor mmCIFs absent from {root}/mmcif: {missing_cif}")
    sys.exit(1)
print(f"  all {len(new)} donor mmCIFs present in {root}/mmcif")
PYEOF2

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
