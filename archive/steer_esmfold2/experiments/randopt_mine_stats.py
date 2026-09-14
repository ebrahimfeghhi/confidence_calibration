#!/usr/bin/env python
"""randopt_mine_stats.py — density + expert-selection readout at σ=0.002 (up to 100 draws).

Combines the 4 screen draws (randopt_screen/s0.002_seed{1-4}.csv) with the 96 mining draws
(randopt_mine_s002.csv, keyed by fold-seed = perturbation index). Reports the per-draw mean DockQ
lift distribution, density (fraction of draws that help), top-k candidate experts, and per-complex
responsiveness — the inputs to selecting an expert and testing transfer.
"""
import csv, os, collections
import numpy as np

# Set-aware via env (defaults = set A / the original 16). For set B:
#   RANDOPT_CX="8fas 8aon ..." RANDOPT_SCR=results/csv_files/randopt_screen_B \
#   RANDOPT_MINE=results/csv_files/randopt_mine_s002_B.csv python randopt_mine_stats.py
CX = os.environ.get("RANDOPT_CX",
      "8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo").split()
SCR = os.environ.get("RANDOPT_SCR", "results/csv_files/randopt_screen")
MINE_CSV = os.environ.get("RANDOPT_MINE", "results/csv_files/randopt_mine_s002.csv")

def best_map(path):
    """path -> {complex: best dockq over interfaces x samples}."""
    b = collections.defaultdict(float)
    if not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") == "chain_unmapped":
            continue
        try:
            b[r["assembly_id"][:4]] = max(b[r["assembly_id"][:4]], float(r["dockq"]))
        except (ValueError, KeyError):
            pass
    return dict(b)

def mine_by_seed():
    d = collections.defaultdict(lambda: collections.defaultdict(float))
    if not os.path.exists(MINE_CSV):
        return {}
    for r in csv.DictReader(open(MINE_CSV)):
        if r.get("dockq_status") == "chain_unmapped":
            continue
        try:
            s = int(r["seed"]); dq = float(r["dockq"])
        except (ValueError, KeyError):
            continue
        d[s][r["assembly_id"][:4]] = max(d[s][r["assembly_id"][:4]], dq)
    return {s: dict(v) for s, v in d.items()}

def main():
    base = best_map(f"{SCR}/baseline.csv")
    if not base:
        print("no baseline yet"); return
    draws = {}
    for n in (1, 2, 3, 4):
        m = best_map(f"{SCR}/s0.002_seed{n}.csv")
        if m:
            draws[n] = m
    for s, m in mine_by_seed().items():
        draws[s] = m

    rows = []
    for did, m in draws.items():
        common = [cx for cx in CX if cx in base and cx in m]
        if len(common) < 12:      # skip still-folding draws
            continue
        d = np.array([m[cx] - base[cx] for cx in common])
        rows.append((did, d.mean(), int((d > 0.05).sum()), len(common)))
    if not rows:
        print("no complete draws scored yet"); return
    lifts = np.array([r[1] for r in rows])
    print(f"draws scored: {len(rows)} (target 100); baseline mean best-DockQ "
          f"{np.mean(list(base.values())):.3f}\n")
    print("per-draw MEAN lift over 16 complexes:")
    print(f"  min {lifts.min():+.3f} | median {np.median(lifts):+.3f} | "
          f"mean {lifts.mean():+.3f} | max {lifts.max():+.3f}")
    print(f"DENSITY: {100*(lifts>0).mean():.0f}% net-positive | "
          f"{100*(lifts>0.03).mean():.0f}% lift>0.03 | {100*(lifts>0.06).mean():.0f}% lift>0.06")
    print("\ntop 12 draws (candidate experts):")
    for did, m, nimp, n in sorted(rows, key=lambda x: -x[1])[:12]:
        print(f"  draw {did:>3}: mean lift {m:+.3f}  ({nimp}/{n} improved)")
    print("\nper-complex responsiveness (# draws lifting it >0.10):")
    for cx in CX:
        if cx not in base:
            continue
        resc = sum(1 for m in draws.values() if cx in m and m[cx] - base[cx] > 0.10)
        print(f"  {cx} (base {base[cx]:.2f}): {resc}/{len(draws)} draws")

if __name__ == "__main__":
    main()
