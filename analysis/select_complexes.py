"""
Select a representative subset of FoldBench complexes spanning the difficulty
spectrum, for cheaper sweeps that still cover the interesting regimes.

Per-seed performance is computed the way the rest of the repo selects samples
(see plot/plot_tts_slopegraph.py, plot/plot_tts_scatter.py): within each
(interface, seed) we pick the single diffusion sample with the highest
confidence score (--conf-col, default pair_iptm) and keep THAT sample's DockQ
-- i.e. the confidence head chooses the sample, not an oracle. For multi-
interface complexes we average the confidence-selected DockQ over interfaces to
get one value per (complex, seed).

Difficulty axis = "best_seed" = the best per-seed value any seed achieves for a
complex. We rank complexes by best_seed and pick N evenly spaced across the
ranking (quantile sampling), so the subset always includes:
  * complexes no seed does well on  (low best_seed, bottom of the ranking), and
  * complexes a seed does really well on (high best_seed, top of the ranking),
with even coverage of everything in between.

Usage:
    .venv/bin/python analysis/select_complexes.py --n 30
    .venv/bin/python analysis/select_complexes.py --n 30 --conf-col pair_ranking_score
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DIR / "dockq_ost_esmfold2fast_tts_all_samples.csv"))
    ap.add_argument("--n", type=int, default=30, help="number of complexes to select")
    ap.add_argument("--conf-col", default="pair_iptm",
                    help="confidence column used to pick the best sample within a "
                         "(interface, seed) (default: pair_iptm, matching the repo's "
                         "tts plots).")
    args = ap.parse_args()

    # samples[(complex, iface, seed)] = list of (conf, dockq) over diffusion samples
    samples = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if row["seed"] == "baseline_seed":
                continue
            try:
                dq = float(row["dockq"])
                conf = float(row[args.conf_col])
            except (KeyError, ValueError):
                continue
            if dq != dq or conf != conf:  # NaN dockq or NaN confidence -> unusable
                continue
            cplx = row["assembly_id"].split("-")[0]
            samples[(cplx, (row["chain1"], row["chain2"]), row["seed"])].append((conf, dq))

    # confidence-selected DockQ per (complex, iface, seed)
    sel = defaultdict(dict)   # sel[(complex, seed)][iface] = dockq of top-confidence sample
    for (cplx, iface, seed), pairs in samples.items():
        _, best_dq = max(pairs, key=lambda x: x[0])
        sel[(cplx, seed)][iface] = best_dq

    # one value per (complex, seed) = mean of confidence-selected DockQ over interfaces
    per_seed = defaultdict(list)
    for (cplx, seed), ifaces in sel.items():
        per_seed[cplx].append(statistics.mean(ifaces.values()))

    stats = []
    for cplx, vals in per_seed.items():
        stats.append({
            "complex": cplx,
            "n_seed": len(vals),
            "best_seed": max(vals),
            "med_seed": statistics.median(vals),
        })

    ranked = sorted(stats, key=lambda s: s["best_seed"])
    N = len(ranked)
    n = min(args.n, N)
    # evenly spaced indices across the full ranking, inclusive of both ends
    idx = sorted({round(i * (N - 1) / (n - 1)) for i in range(n)})
    pool = iter([j for j in range(N) if j not in set(idx)])
    while len(idx) < n:
        idx.append(next(pool))
    idx = sorted(idx)
    selected = [ranked[i] for i in idx]

    print(f"{N} complexes ranked by best_seed (best sample per (interface,seed) "
          f"chosen by {args.conf_col}); selecting {n} evenly across the spectrum.\n")
    print(f"{'rank':>4s} {'complex':10s} {'best_seed':>9s} {'med_seed':>8s}")
    for i, s in zip(idx, selected):
        flag = "  <- no seed clears 0.23" if s["best_seed"] < 0.23 else ""
        gap = "  (lucky-seed: med<<best)" if s["best_seed"] - s["med_seed"] > 0.4 else ""
        print(f"{i:4d} {s['complex']:10s} {s['best_seed']:9.3f} {s['med_seed']:8.3f}{flag}{gap}")

    n_hard = sum(1 for s in selected if s["best_seed"] < 0.23)
    print(f"\nselected: {n_hard} with no seed clearing 0.23, "
          f"{n - n_hard} with at least one good seed")
    print("\n--complexes " + " ".join(s["complex"] for s in selected))


if __name__ == "__main__":
    main()
