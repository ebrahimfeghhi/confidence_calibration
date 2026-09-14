#!/usr/bin/env python
"""The 1-seed vs 50-seed selection protocol, applied to PXMeter's PXM-2025 ab-ag set.

Independent replication target: PXM-2025 shares ZERO PDB entries with FoldBench and is drawn from
2025 depositions, so it tests whether the seed-scaling result is a property of the model or of the
FoldBench targets.

PROTOCOL (identical to scripts/selection_protocol.py and selection_protocol_pp.py):
  1-seed  -- within ONE seed take the pose its own interface ipTM ranks first, do that for every
             seed, then report the one with the MEDIAN DockQ (lower median, so a real pose).
  50-seed -- the pose ipTM ranks first pooled over all of that interface's poses.
  oracle  -- best DockQ available, reported for both depths the same way.

METRIC IS PXMETER'S OWN DockQ, not OST. Their benchmark, their implementation, so these numbers line
up with their leaderboard -- but they are NOT comparable in absolute terms to
artifacts/abag_selection_protocol.csv. Compare RELATIVE effects (the 1-seed vs 50-seed gain) only.

Runs on whatever is scored so far, so it can be re-run while the sweep continues; interfaces with
too few seeds are skipped and counted.

Usage: .../python scripts/selection_protocol_pxm.py [--min-seeds 25]
"""
import argparse
import collections
import csv
import os

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SRC = f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv"
EPI = f"{EXP}/artifacts/pxm2025/epitope_overlap_pxm.csv"
PARA = f"{EXP}/artifacts/pxm2025/paratope_overlap_pxm.csv"
OUT = f"{EXP}/artifacts/pxm2025/pxm2025_selection_protocol.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=25)
    ap.add_argument("--min-poses", type=int, default=100)
    A = ap.parse_args()

    # per-pose epitope Jaccard, keyed on the UNORDERED chain pair: dockq_pxm2025.csv stores the pair
    # as PXMeter emitted it, epitope_overlap_pxm.csv stores it as (antibody, antigen), and those two
    # orders do not always coincide.
    def _load(path):
        d = {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                d[(r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))),
                   int(r["seed"]), int(r["sample"]))] = float(r["jaccard"])
        return d
    epi, para = _load(EPI), _load(PARA)

    poses = collections.defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        poses[f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'].append(
            dict(seed=int(r["seed"]), sample=int(r["sample"]),
                 iptm=float(r["pair_iptm"]), dockq=float(r["dockq"]),
                 epi=epi.get((r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))),
                              int(r["seed"]), int(r["sample"]))),
                 para=para.get((r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))),
                                int(r["seed"]), int(r["sample"])))))

    rows, skipped = [], 0
    for iface, ps in sorted(poses.items()):
        by_seed = collections.defaultdict(list)
        for p in ps:
            by_seed[p["seed"]].append(p)
        if len(by_seed) < A.min_seeds or len(ps) < A.min_poses:
            skipped += 1
            continue
        per_seed = [max(v, key=lambda x: x["iptm"]) for v in by_seed.values()]
        per_seed.sort(key=lambda x: x["dockq"])
        one = per_seed[(len(per_seed) - 1) // 2]
        allsel = max(ps, key=lambda x: x["iptm"])
        best_per_seed = sorted(max(p["dockq"] for p in v) for v in by_seed.values())

        # Epitope columns mirror scripts/selection_protocol.py exactly, so the PXM regression is the
        # same estimator as the FoldBench r2 = 0.82 it is meant to replicate:
        #   sel50_epi_jac    epitope Jaccard of the pooled ipTM pick
        #   sel1med_epi_jac  MEDIAN epitope Jaccard over the 50 per-seed ipTM picks -- the x baseline
        #                    in "per-axis" mode, a different pose from the DockQ-median one
        #   sel1_epi_jac     that same DockQ-median pose's incidental epitope Jaccard -- the
        #                    "consistent" mode baseline, which is noisier and attenuates r2
        # None (pose not yet epitope-scored) propagates as an empty cell rather than a fake 0.
        def side(tag):
            sj = [q[tag] for q in per_seed if q[tag] is not None]
            med = round(sorted(sj)[(len(sj) - 1) // 2], 4) if sj else None
            allv = [q[tag] for q in ps if q[tag] is not None]
            bes = sorted(max(q[tag] for q in v if q[tag] is not None)
                         for v in by_seed.values() if any(q[tag] is not None for q in v))
            return (med, round(max(allv), 4) if allv else None,
                    round(bes[(len(bes) - 1) // 2], 4) if bes else None, len(allv))
        med_epi, orc_epi, oracle1_epi, n_epi = side("epi")
        med_par, orc_par, oracle1_par, n_par = side("para")
        rows.append(dict(interface=iface, n_seeds=len(by_seed), n_poses=len(ps),
                         n_epi=n_epi, n_para=n_par,
                         sel1_epi_jac=one["epi"], sel1med_epi_jac=med_epi,
                         sel50_epi_jac=allsel["epi"],
                         oracle1_epi_jac=oracle1_epi, oracle_epi_jac=orc_epi,
                         sel1_para_jac=one["para"], sel1med_para_jac=med_par,
                         sel50_para_jac=allsel["para"],
                         oracle1_para_jac=oracle1_par, oracle_para_jac=orc_par,
                         oracle1_dockq=round(best_per_seed[(len(best_per_seed) - 1) // 2], 4),
                         oracle_dockq=round(max(p["dockq"] for p in ps), 4),
                         sel1_seed=one["seed"], sel1_sample=one["sample"],
                         sel1_iptm=round(one["iptm"], 4), sel1_dockq=round(one["dockq"], 4),
                         sel50_seed=allsel["seed"], sel50_sample=allsel["sample"],
                         sel50_iptm=round(allsel["iptm"], 4),
                         sel50_dockq=round(allsel["dockq"], 4)))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    F = lambda k: np.array([r[k] for r in rows], float)
    a, b = F("sel1_dockq"), F("sel50_dockq")
    o1, o = F("oracle1_dockq"), F("oracle_dockq")
    d = b - a
    print(f"{len(rows)} PXM-2025 ab-ag interfaces "
          f"(>= {A.min_seeds} seeds, >= {A.min_poses} poses); {skipped} skipped\n")
    print(f"{'arm':<28}{'1 seed':>16}{'50 seeds':>16}{'delta':>16}")
    print(f"{'ipTM-selected DockQ':<28}{a.mean():>9.3f} +/-{a.std():<5.3f}"
          f"{b.mean():>9.3f} +/-{b.std():<5.3f}{d.mean():>+9.3f} +/-{d.std():<5.3f}")
    print(f"{'oracle DockQ':<28}{o1.mean():>9.3f} +/-{o1.std():<5.3f}"
          f"{o.mean():>9.3f} +/-{o.std():<5.3f}{(o-o1).mean():>+9.3f} +/-{(o-o1).std():<5.3f}")
    print(f"\n{'interfaces improved':<28}{100*(d>0).mean():.0f}%")
    print(f"{'acceptable (>=0.23)':<28}1 seed {100*(a>=.23).mean():.1f}%   "
          f"50 seeds {100*(b>=.23).mean():.1f}%   oracle {100*(o>=.23).mean():.1f}%")
    print(f"{'selection delivers':<28}{100*d.mean()/max((o-o1).mean(),1e-9):.0f}% of the "
          f"coverage gain")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
