#!/bin/bash
# Run this ON ALDERAAN before naboo is shut down. Reports what is present vs the naboo reference.
# Expected counts are from naboo 2026-08-10 18:05. Nothing here is destructive.
echo "=== against naboo reference (2026-08-10) ==="
chk () { printf "%-34s %10s   expected %-10s %s\n" "$1" "$2" "$3" "$([ "$2" -ge "$3" ] 2>/dev/null && echo OK || echo '<-- SHORT')"; }
R=$(python3 -c "import sys;sys.path.insert(0,'$(cd "$(dirname "$0")/.." && pwd)');from paths import EXP;print(EXP)" 2>/dev/null)
[ -z "$R" ] && R=/home/ebrahim/tts_abag/confident_decoy_experiments
echo "resolved project root: $R"; echo

echo "-- IRREPLACEABLE (months of GPU/CPU; cannot be regenerated cheaply) --"
chk "artifacts/ top-level entries"  "$(ls $R/artifacts 2>/dev/null | wc -l)"                 366
chk "artifacts/*.json (results)"    "$(ls $R/artifacts/*.json 2>/dev/null | wc -l)"           239
chk "artifacts/*_ost cache dirs"    "$(ls -d $R/artifacts/*_ost 2>/dev/null | wc -l)"          29
chk "  total cached OST scores"     "$(find $R/artifacts -path '*_ost/*.json' 2>/dev/null | wc -l)" 202210
chk "artifacts/sweep_out runs"      "$(ls $R/artifacts/sweep_out 2>/dev/null | wc -l)"          30
chk "predictions/ pdbs"             "$(ls $R/predictions 2>/dev/null | wc -l)"                   4
chk "\$HOME/mmcif/*.cif (templates)" "$(ls $HOME/mmcif/*.cif 2>/dev/null | wc -l)"             4581
chk "release_date_cache entries"    "$(python3 -c "import json;print(len(json.load(open('$HOME/common/release_date_cache.json'))))" 2>/dev/null || echo 0)" 223152

echo
echo "-- REGENERABLE BUT EXPENSIVE (GPU time / downloads) --"
chk "foldbench predictions pdbs"    "$(ls /home/ebrahim/Protenix/foldbench_abag/predictions_template 2>/dev/null | wc -l)" 127
chk "foldbench jsons"               "$(ls /home/ebrahim/Protenix/foldbench_abag/jsons/*.json 2>/dev/null | wc -l)"         339
chk "FoldBench ground truths"       "$(ls /home/ebrahim/FoldBench/examples/ground_truths/*.cif 2>/dev/null | wc -l)"       116
chk "protenix checkpoint files"     "$(ls /home/ebrahim/checkpoint 2>/dev/null | wc -l)"                                     1

echo
echo "-- SIZES (naboo: artifacts 91G, foldbench_abag 43G, mmcif 4.4G) --"
du -sh $R/artifacts $R/predictions /home/ebrahim/Protenix/foldbench_abag $HOME/mmcif 2>/dev/null

echo
echo "-- CODE (safe on GitHub, just confirm the branch) --"
git -C /home/ebrahim/tts_abag rev-parse --abbrev-ref HEAD 2>/dev/null
git -C /home/ebrahim/tts_abag log --oneline -1 2>/dev/null
echo
echo "NOTE: ~/.claude on naboo (185 MB) holds the Claude session transcript and the project"
echo "      memory rules. It is NOT in git and NOT covered by this check -- copy it separately:"
echo "      tar czf claude_backup.tgz -C /home/ebrahim .claude"
