#!/usr/bin/env bash
# Prepare FoldBench protein-protein inputs using ONLY Protenix's own tooling.
#   1. download the deposited ASYMMETRIC UNIT from RCSB (not assembly1): the ab-ag inputs were built
#      from the ASU -- 7ST5's ASU has 6 chains / 2 copies per entity, matching "count": 2, while
#      assembly1 has 3 chains and would give count 1.
#   2. `protenix json`  -> <pdb>.json   (applies CRYSTALLIZATION_AIDS, merges covalent glycan bonds)
#      VERIFIED to reproduce foldbench_abag/jsons/{7st5,8c5h}.json exactly.
#   3. `protenix mt`    -> <pdb>-update-msa.json then <pdb>-final-updated.json
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PP=$REPO/artifacts/pp
CLI=/home/ebrahim/Protenix/.venv/bin/protenix
cd /home/ebrahim/Protenix || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a $REPO/ops/pp_prep.log; }

say "STEP 1: downloading asymmetric units from RCSB"
n=0
while read -r p; do
  [ -s "$PP/cif/$p.cif" ] && continue
  curl -sS --max-time 120 -o "$PP/cif/$p.cif.gz" "https://files.rcsb.org/download/${p^^}.cif.gz" \
    && gunzip -f "$PP/cif/$p.cif.gz" 2>/dev/null
  n=$((n+1)); [ $((n%25)) -eq 0 ] && say "  downloaded $n"
done < "$PP/ids.txt"
say "STEP 1 done: $(ls $PP/cif/*.cif 2>/dev/null | wc -l) cifs"

say "STEP 2: protenix json (CIF -> input json)"
$CLI json --input "$PP/cif" --out_dir "$PP/jsons" >> $REPO/ops/pp_prep.log 2>&1
say "STEP 2 done: $(ls $PP/jsons/*.json 2>/dev/null | wc -l) jsons"

say "STEP 3: protenix mt (MSA + template search)"
for j in "$PP"/jsons/*.json; do
  b=$(basename "$j" .json)
  case "$b" in *-update-msa|*-final-updated) continue;; esac
  [ -s "$PP/jsons/${b}-final-updated.json" ] && continue
  $CLI mt --input "$j" --out_dir "$PP/prep" >> $REPO/ops/pp_prep.log 2>&1 \
    || say "  MT FAILED $b"
done
say "STEP 3 done: $(ls $PP/jsons/*-final-updated.json 2>/dev/null | wc -l) final jsons"
say "PREP COMPLETE"
