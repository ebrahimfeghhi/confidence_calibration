#!/usr/bin/env python
"""The ab-ag selection protocol, applied to FoldBench protein-protein.

WHY: "seed scaling helps ab-ag but not PP" was an inference from two indirect measurements (lower
DockQ spread across seeds, and Heaps alpha 0.26 vs 0.34). It had never been measured with the same
1-seed-vs-50-seed protocol used for ab-ag, so the two arms were not comparable. This closes that gap.

PROTOCOL (identical to scripts/selection_protocol.py):
  1-seed  -- within ONE seed take the pose its own interface ipTM ranks first among that seed's
             samples; do that for every seed, then report the one with the MEDIAN DockQ (lower
             median, so the reported value belongs to a real pose rather than an interpolation).
             Picking one arbitrary seed would make the baseline a coin flip.
  50-seed -- the pose ipTM ranks first pooled over all of that interface's poses.
Both arms are ipTM-selected, so the difference is what extra SAMPLING buys, never an oracle.

INPUT: artifacts/pp/dockq_pp.csv, written by scripts/score_pp.py from OST `compare-structures
--dockq`. Its pair_iptm column is already the per-interface chain_pair_iptm value for the OST-matched
model chains, so no confidence files are re-read here.

Usage: .../python scripts/selection_protocol_pp.py
Writes artifacts/pp_selection_protocol.csv (one row per interface).
"""
import collections
import csv

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SRC = f"{EXP}/artifacts/pp/dockq_pp.csv"
OUT = f"{EXP}/artifacts/pp_selection_protocol.csv"
MIN_SEEDS, MIN_POSES = 25, 100


def main():
    poses = collections.defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        poses[f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'].append(
            dict(seed=int(r["seed"]), sample=int(r["sample"]), iptm=float(r["pair_iptm"]),
                 dockq=float(r["dockq"]), fnat=float(r["fnat"])))

    rows, skipped = [], 0
    for iface, ps in sorted(poses.items()):
        by_seed = collections.defaultdict(list)
        for p in ps:
            by_seed[p["seed"]].append(p)
        if len(by_seed) < MIN_SEEDS or len(ps) < MIN_POSES:
            skipped += 1
            continue
        per_seed = [max(v, key=lambda x: x["iptm"]) for v in by_seed.values()]
        per_seed.sort(key=lambda x: x["dockq"])
        one = per_seed[(len(per_seed) - 1) // 2]
        allsel = max(ps, key=lambda x: x["iptm"])
        # ORACLE arm, same shape: within a seed take its BEST DockQ (not ipTM's pick), then the
        # median seed. Upper bound on what one seed could give if selection were perfect.
        best_per_seed = sorted(max(p["dockq"] for p in v) for v in by_seed.values())
        row = dict(interface=iface, n_seeds=len(by_seed), n_poses=len(ps),
                   oracle1_dockq=round(best_per_seed[(len(best_per_seed) - 1) // 2], 4),
                   oracle_dockq=round(max(p["dockq"] for p in ps), 4))
        for tag, p in (("sel1", one), ("sel50", allsel)):
            row[f"{tag}_seed"], row[f"{tag}_sample"] = p["seed"], p["sample"]
            row[f"{tag}_iptm"] = round(p["iptm"], 4)
            row[f"{tag}_dockq"] = round(p["dockq"], 4)
            row[f"{tag}_fnat"] = round(p["fnat"], 4)
        rows.append(row)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    a = np.array([r["sel1_dockq"] for r in rows])
    b = np.array([r["sel50_dockq"] for r in rows])
    orc = np.array([r["oracle_dockq"] for r in rows])
    print(f"{len(rows)} PP interfaces (>= {MIN_SEEDS} seeds, >= {MIN_POSES} poses); "
          f"{skipped} skipped for too little data\n")
    print(f"{'metric':<10}{'N':>5}{'1 seed':>16}{'50 seeds':>16}{'delta':>16}{'improved':>10}")
    for name, x, y in (("dockq", a, b),
                       ("fnat", np.array([r["sel1_fnat"] for r in rows]),
                        np.array([r["sel50_fnat"] for r in rows]))):
        d = y - x
        print(f"{name:<10}{len(x):>5}{x.mean():>9.3f} +/-{x.std():<5.3f}"
              f"{y.mean():>9.3f} +/-{y.std():<5.3f}{d.mean():>+9.3f} +/-{d.std():<5.3f}"
              f"{100 * (d > 0).mean():>9.0f}%")
    print(f"\n{'acceptable (DockQ>=0.23)':<28} 1 seed {100 * (a >= .23).mean():.1f}%"
          f"   50 seeds {100 * (b >= .23).mean():.1f}%   oracle {100 * (orc >= .23).mean():.1f}%")
    print(f"{'mean oracle DockQ':<28} {orc.mean():.3f}   50-seed selection captures "
          f"{100 * (b.mean() - a.mean()) / (orc.mean() - a.mean()):.0f}% of the headroom")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
