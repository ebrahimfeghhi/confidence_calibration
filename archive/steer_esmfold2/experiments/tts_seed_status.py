#!/usr/bin/env python
"""tts_seed_status.py — one-command status of the esmfold2fast_tts benchmark pool.

Answers "how deep is each complex, and is it folded vs scored?" without re-deriving it by hand.
For each complex: folds on disk (seed_ dirs) vs distinct seeds SCORED in the canonical deep CSV.

  python steer/experiments/tts_seed_status.py            # severe/confident-decoy set (34 cx)
  python steer/experiments/tts_seed_status.py --all      # every complex in the run dir

Canonical scores: results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv  (fixed --dockq pipeline,
mask0.1/dropout0.3 — see docs/esmfold2fast_tts_benchmark.md). Folds: ~/esm/foldbench_predictions/
esmfold2fast_tts/<cx>/seed_N/.  baseline_seed/ is the only mask0/drop0 fold (a clean reference).
"""
import csv, os, sys, argparse, collections

PRED = os.path.expanduser("~/esm/foldbench_predictions/esmfold2fast_tts")
DEEP = "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
STRUCT_CSV = "results/csv_files/decoy_structural_analysis.csv"

def severe_set():
    return sorted(set(r["cx"][:4] for r in csv.DictReader(open(STRUCT_CSV))))

def scored_seeds(cxset):
    per = collections.defaultdict(set)
    if not os.path.exists(DEEP): return per
    for r in csv.DictReader(open(DEEP)):
        cx = r["assembly_id"][:4]
        if cx in cxset and r["seed"].lstrip("-").isdigit():
            per[cx].add(int(r["seed"]))
    return per

def folds(cx):
    cand = [d for d in os.listdir(PRED) if d[:4] == cx] if os.path.isdir(PRED) else []
    if not cand: return 0
    d = os.path.join(PRED, cand[0])
    return sum(1 for x in os.listdir(d) if x.startswith("seed_") and x[5:].isdigit())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all complexes in the run dir, not just severe")
    a = ap.parse_args()
    if a.all:
        cxs = sorted(set(d[:4] for d in os.listdir(PRED))) if os.path.isdir(PRED) else []
    else:
        cxs = severe_set()
    cxset = set(cxs)
    sc = scored_seeds(cxset)
    print(f"{'cx':>6} {'folds':>6} {'scored':>7} {'unscored':>9}")
    tf = ts = 0
    for c in sorted(cxs, key=lambda c: -(folds(c) - len(sc.get(c, set())))):
        fd = folds(c); s = len(sc.get(c, set())); tf += fd; ts += s
        flag = " <== needs scoring" if fd - s > 5 else (" (deep)" if s > 200 else "")
        print(f"{c:>6} {fd:>6} {s:>7} {fd-s:>9}{flag}")
    print(f"\nTOTAL folds={tf}  scored={ts}  unscored={tf-ts}"
          f"  ({100*(tf-ts)/tf:.0f}% unscored)" if tf else "\n(no folds found)")

if __name__ == "__main__":
    main()
