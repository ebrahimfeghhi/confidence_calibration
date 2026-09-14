#!/usr/bin/env python
"""randopt_watch.py — incremental scorer for the Phase-1 screen.

Polls the screen output dir; the moment a cell's 16 complexes finish folding it OST-scores that
cell and prints ONE line: the baseline's mean best-DockQ, or a perturbation's DockQ lift vs baseline
(mean lift + #complexes improved). Exits when all 13 cells are scored (or the deadline). Designed to
be run under Monitor so each printed line becomes a live update.
"""
import os, sys, time, csv, subprocess, collections
import numpy as np

PY = "/home/ebrahim/esm/.venv/bin/python"
PRED = os.path.expanduser("~/esm/foldbench_predictions/randopt_screen")
OUT = "results/csv_files/randopt_screen"; os.makedirs(OUT, exist_ok=True)
CX = ["8ts0","8ath","7yxu","7zoz","8a44","8w84","7ua2","8dcm",
      "8oxw","7ucx","8vui","8aok","8oni","8q3j","8ivx","8vzo"]
CELLS = ["baseline"] + [f"s{s}_seed{d}" for s in ("0.0005","0.001","0.002","0.005","0.01") for d in (1,2,3,4)]
IMP = 0.05
DEADLINE = time.time() + 3000

def n_folded(cell):
    d = os.path.join(PRED, cell)
    if not os.path.isdir(d):
        return 0
    k = 0
    for cx in CX:
        pd = os.path.join(d, cx, "seed_0", "predictions")
        if os.path.isdir(pd) and any(f.endswith(".cif") for f in os.listdir(pd)):
            k += 1
    return k

def score(cell):
    subprocess.run([PY, "eval/eval_ost_foldbench.py",
                    "--pred_dir", f"{PRED}/{cell}", "--out_csv", f"{OUT}/{cell}.csv"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def per_complex_best(cell):
    best = collections.defaultdict(float)
    p = f"{OUT}/{cell}.csv"
    if not os.path.exists(p):
        return {}
    for r in csv.DictReader(open(p)):
        if r.get("dockq_status") == "chain_unmapped":
            continue
        try:
            best[r["assembly_id"][:4]] = max(best[r["assembly_id"][:4]], float(r["dockq"]))
        except (ValueError, KeyError):
            pass
    return dict(best)

def main():
    scored, base = set(), None
    # pre-seed from any already-scored cells so a re-run only streams NEW cells
    if per_complex_best("baseline"):
        base = per_complex_best("baseline"); scored.add("baseline")
    for cell in CELLS:
        if cell != "baseline" and per_complex_best(cell):
            scored.add(cell)
    while len(scored) < len(CELLS) and time.time() < DEADLINE:
        for cell in CELLS:
            if cell in scored:
                continue
            if n_folded(cell) < 16:
                continue
            if cell != "baseline" and base is None:
                continue                      # need baseline scored first for the lift
            score(cell)
            b = per_complex_best(cell)
            if cell == "baseline":
                base = b
                print(f"[baseline] scored: mean best-DockQ {np.mean(list(b.values())):.3f} "
                      f"over {len(b)}/16", flush=True)
            else:
                common = [c for c in CX if c in base and c in b]
                d = np.array([b[c] - base[c] for c in common]) if common else np.array([0.0])
                nimp = int((d > IMP).sum()); ndeg = int((d < -IMP).sum())
                print(f"[{cell}] lift {d.mean():+.3f}  ({nimp}/{len(common)} improved, "
                      f"{ndeg} degraded)", flush=True)
            scored.add(cell)
        time.sleep(15)
    print(f"WATCH_DONE scored={len(scored)}/{len(CELLS)}", flush=True)

if __name__ == "__main__":
    main()
