#!/usr/bin/env bash
# Prepare PXMeter PXM-2025 antibody-antigen inputs for Protenix.
#
# 131 complexes / 314 ab-ag interfaces, from
# artifacts/pxmeter/PXM-2025/supported_data/RecentPDB_low_homology.csv rows whose `subset` contains
# "antibody". Zero overlap with FoldBench, so this is an independent replication set for the
# seed-scaling result.
#
# UNLIKE ops/pp_prep.sh there is NO RCSB download step: PXMeter ships the structures
# (supported_data/mmcif, author-defined assemblies with chains already split), so we convert those
# directly. Verified on 8ror that `protenix json` reproduces PXMeter's own chain lengths
# (1457/218/222, count 1 each), i.e. the assembly CIFs do not produce the entity-copy mismatch that
# the ASU-vs-assembly confusion caused for the protein-protein set.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
P=$REPO/artifacts/pxm2025
CLI=/home/ebrahim/Protenix/.venv/bin/protenix
LOG=$REPO/ops/pxm2025_prep.log
mkdir -p "$P/jsons" "$P/prep"
cd /home/ebrahim/Protenix || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "STEP 1: protenix json (assembly CIF -> input json), $(ls $P/cif/*.cif | wc -l) cifs"
$CLI json --input "$P/cif" --out_dir "$P/jsons" >> "$LOG" 2>&1
say "STEP 1 done: $(ls $P/jsons/*.json 2>/dev/null | wc -l) jsons"

say "STEP 2: protenix mt (MSA + template search)"
for j in "$P"/jsons/*.json; do
  b=$(basename "$j" .json)
  case "$b" in *-update-msa|*-final-updated) continue;; esac
  [ -s "$P/jsons/${b}-final-updated.json" ] && continue
  $CLI mt --input "$j" --out_dir "$P/prep" >> "$LOG" 2>&1 || say "  MT FAILED $b"
done
say "STEP 2 done: $(ls $P/jsons/*-final-updated.json 2>/dev/null | wc -l) final jsons"
say "PREP COMPLETE"
