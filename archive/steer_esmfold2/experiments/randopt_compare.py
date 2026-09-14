#!/usr/bin/env python
"""randopt_compare.py — 3-way comparison of weight-perturbation mining at σ=0.001/0.002/0.003
(100 draws each, set-A 16 complexes, mask0/dropout0). Per sigma: per-draw mean-lift distribution,
density, oracle-of-100 (best-direction-per-complex). Plus per-complex responsiveness across sigmas.
Runs on whatever is scored so far (safe to run mid-flight)."""
import csv, os, collections
import numpy as np

CX = ["8ts0","8ath","7yxu","7zoz","8a44","8w84","7ua2","8dcm",
      "8oxw","7ucx","8vui","8aok","8oni","8q3j","8ivx","8vzo"]
BASE = "results/csv_files/randopt_screen/baseline.csv"
SCR = "results/csv_files/randopt_screen"
RUNS = {"0.001":"results/csv_files/randopt_mine_s001.csv",
        "0.002":"results/csv_files/randopt_mine_s002.csv",
        "0.003":"results/csv_files/randopt_mine_s003.csv"}

def best_map(path):
    b = collections.defaultdict(float)
    if not os.path.exists(path): return {}
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") == "chain_unmapped": continue
        try: b[r["assembly_id"][:4]] = max(b[r["assembly_id"][:4]], float(r["dockq"]))
        except (ValueError, KeyError): pass
    return dict(b)

def draws_for(sigma, path):
    d = {}
    if sigma == "0.002":                       # 0.002 = 96 mine (seeds 5-100) + 4 screen draws (1-4)
        for n in (1,2,3,4):
            m = best_map(f"{SCR}/s0.002_seed{n}.csv")
            if m: d[n] = m
    by = collections.defaultdict(lambda: collections.defaultdict(float))
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("dockq_status") == "chain_unmapped": continue
            try: s = int(r["seed"]); dq = float(r["dockq"])
            except (ValueError, KeyError): continue
            by[s][r["assembly_id"][:4]] = max(by[s][r["assembly_id"][:4]], dq)
    for s, m in by.items(): d[s] = dict(m)
    return d

def main():
    base = best_map(BASE)
    if not base: print("no baseline"); return
    print(f"baseline mean best-DockQ {np.mean([base[c] for c in CX if c in base]):.3f}  (16 complexes, mask0/dropout0)\n")
    print(f"{'sigma':>6} {'draws':>5} {'meanLift':>9} {'median':>7} {'max':>7} {'%pos':>5} {'%>.06':>6} {'oracle100':>10}")
    per_sig_draws = {}
    for sig, path in RUNS.items():
        draws = draws_for(sig, path); per_sig_draws[sig] = draws
        lifts = []
        for m in draws.values():
            common = [c for c in CX if c in base and c in m]
            if len(common) < 12: continue
            lifts.append(np.mean([m[c]-base[c] for c in common]))
        if not lifts: print(f"{sig:>6}  (none scored yet)"); continue
        lifts = np.array(lifts)
        oracle = np.mean([max([m.get(c,0) for m in draws.values()] + [0]) for c in CX if c in base])
        print(f"{sig:>6} {len(lifts):>5} {lifts.mean():+9.3f} {np.median(lifts):+7.3f} "
              f"{lifts.max():+7.3f} {100*(lifts>0).mean():4.0f}% {100*(lifts>0.06).mean():5.0f}% {oracle:10.3f}")
    # per-complex responsiveness across sigmas (# draws lifting it >0.10)
    print("\nper-complex responsiveness — # draws lifting DockQ >0.10 (of ~100 each):")
    print(f"{'complex':>8} {'base':>5}   " + "  ".join(f"σ{s}" for s in RUNS))
    for c in CX:
        if c not in base: continue
        cells = []
        for sig in RUNS:
            draws = per_sig_draws[sig]
            n = sum(1 for m in draws.values() if c in m and m[c]-base[c] > 0.10)
            cells.append(f"{n:>5}")
        print(f"{c:>8} {base[c]:5.2f}   " + "  ".join(cells))

if __name__ == "__main__":
    main()
