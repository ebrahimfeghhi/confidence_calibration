#!/usr/bin/env python
"""Score the DEEP pose pools for antibody-antigen complexes that look hopeless at 50 seeds.

Seven of the sixteen "no acceptable pose" interfaces have 1024-1078 seeds already generated but only
151-297 of them scored, so the verdict "this target never produces a good pose" rests on ~5% of the
poses that exist. Deep sampling raised oracle DockQ by +0.230 on the confident decoys, so it is worth
checking whether these are genuinely uncoverable or merely undersampled.

Poses come from scripts/pose_index.py (the deep arms live outside predictions_template).
Writes into the same artifacts/abag_all_ost cache the rest of the project reads, so every existing
analysis picks the new scores up automatically.

OST flags match the rest of the project: --dockq only, never --rigid-scores (which aborts on some
targets and throws away a valid DockQ).
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
from pose_index import poses

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
OUT = f"{EXP}/artifacts/abag_all_ost"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
# Pin each OST process to ONE thread. Without this a worker uses ~1.6 cores, so a 64-worker pool
# took ~106 cores and starved the GPU jobs (GPU0 and GPU4 fell to 0-2% utilisation).
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1")
FREE = ("predictions_template", "freedeep", "allpose", "free1k")
PDBS = ["8a44", "8ivx", "8sfz", "8udz", "8vzo", "8z8v", "7yv1"]


def run(a):
    cif, native, out = a
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 0
    try:
        subprocess.run([OST, "compare-structures", "-m", cif, "-r", native, "-o", out,
                        "--fault-tolerant", "--min-pep-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return 0
    return 1


def main():
    jobs = []
    for pdb in PDBS:
        nat = f"{GT}/{pdb}-assembly1.cif"
        if not os.path.exists(nat):
            print(f"{pdb}: no ground truth"); continue
        seen = set()
        for s, k, arm, f in poses(pdb):
            if not any(t in arm for t in FREE) or "nomsa" in arm or "fullmsa" in arm:
                continue
            if (s, k) in seen:
                continue
            seen.add((s, k))
            o = f"{OUT}/{pdb}_{s}_{k}.json"
            if not (os.path.exists(o) and os.path.getsize(o) > 0):
                jobs.append((f, nat, o))
        print(f"{pdb}: {len(seen)} poses in the free pool, {sum(1 for j in jobs if f'/{pdb}_' in j[2])} unscored",
              flush=True)
    print(f"\nscoring {len(jobs)} poses with {160} workers", flush=True)
    n = [0]
    def work(j):
        n[0] += run(j)
        if n[0] % 1000 == 0:
            print(f"  {n[0]}/{len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=160) as ex:
        list(ex.map(work, jobs))
    print(f"done: {n[0]} new OST jsons")


if __name__ == "__main__":
    main()
