#!/usr/bin/env bash
# FoldBench ground truth = the RCSB biological assembly 1, verified byte-identical to the
# examples/ground_truths files FoldBench references (checked on 8gy5, 8qf4, 8c5h, 7st5).
GT=/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/pp/gt
fetch(){ p=$1; f=$GT/$p-assembly1.cif
  [ -s "$f" ] && return 0
  curl -sSL --retry 3 --max-time 120 -o "$f.gz" "https://files.rcsb.org/download/$p-assembly1.cif.gz" \
    && gunzip -f "$f.gz" || { rm -f "$f.gz" "$f"; echo "FAIL $p"; }
}
export -f fetch; export GT
xargs -a "$1" -P 8 -I{} bash -c 'fetch {}'
