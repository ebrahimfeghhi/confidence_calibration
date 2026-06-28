"""
Show how much DockQ varies across seeds, per interface.

For each interface (assembly_id, chain1, chain2) we reduce each seed's 5 samples to
a single "seed performance" value (default: mean DockQ over samples), then draw the
distribution of those per-seed values as a horizontal box + jittered strip. Interfaces
are sorted by median per-seed DockQ. The baseline_seed value (if present) is overlaid
as a diamond so you can see where "no masking/dropout" lands within the seed spread.

This answers: for a given interface, does the seed matter a lot or a little, and is the
spread different across interfaces?

Usage:
    .venv/bin/python plot/plot_seed_variation.py --csv results/csv_files/dockq_ost_esmfold2fast_tts_all_samples.csv
    .venv/bin/python plot/plot_seed_variation.py --agg max --per-sample
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import RESULTS_DIR

AGG_FUNCS = {"mean": statistics.mean, "median": statistics.median, "max": max, "min": min}


def load(csv_path, agg, per_sample):
    """Return (per_iface, baseline) keyed by (assembly_id, chain1, chain2).

    per_iface[key] = list of per-seed DockQ values (one per seed) unless --per-sample,
      in which case every (seed, sample) DockQ is its own point.
    baseline[key]  = per-seed-aggregated baseline_seed DockQ, or None.
    """
    # raw[key][seed] = [dockq, ...] over samples
    raw = defaultdict(lambda: defaultdict(list))
    base_raw = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                dq = float(row["dockq"])
            except (KeyError, ValueError):
                continue
            if dq != dq:  # NaN
                continue
            key = (row["assembly_id"], row["chain1"], row["chain2"])
            if row["seed"] == "baseline_seed":
                base_raw[key].append(dq)
            else:
                raw[key][row["seed"]].append(dq)

    reduce = AGG_FUNCS[agg]
    per_iface = {}
    for key, seeds in raw.items():
        if per_sample:
            per_iface[key] = [dq for vals in seeds.values() for dq in vals]
        else:
            per_iface[key] = [reduce(vals) for vals in seeds.values() if vals]
    baseline = {key: reduce(vals) for key, vals in base_raw.items() if vals}
    return per_iface, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RESULTS_DIR / "csv_files"
                    / "dockq_ost_esmfold2fast_tts_all_samples.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--agg", choices=list(AGG_FUNCS), default="mean",
                    help="how to reduce a seed's samples to one value (default: mean)")
    ap.add_argument("--per-sample", action="store_true",
                    help="plot every (seed,sample) point instead of one mean per seed")
    ap.add_argument("--threshold", type=float, default=0.23)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    per_iface, baseline = load(csv_path, args.agg, args.per_sample)
    if not per_iface:
        print("No interfaces loaded.")
        return

    unit = "sample" if args.per_sample else f"seed ({args.agg} over samples)"
    # sort by median so the best interfaces sit at the top of the horizontal axis
    keys = sorted(per_iface, key=lambda k: statistics.median(per_iface[k]))
    spreads = {k: (max(v) - min(v)) for k, v in per_iface.items()}

    print(f"Interfaces: {len(keys)}   point = one {unit}")
    print(f"Median per-interface seed spread (max-min DockQ): "
          f"{statistics.median(spreads.values()):.3f}")
    print("\nMost seed-sensitive interfaces (largest DockQ spread across seeds):")
    for k in sorted(spreads, key=spreads.get, reverse=True)[:8]:
        aid, c1, c2 = k
        v = per_iface[k]
        print(f"  spread={spreads[k]:.3f}  median={statistics.median(v):.3f}  "
              f"{aid} {c1}|{c2}  (n={len(v)})")

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    # one summary per interface: median, 10-90th pct band, IQR spread
    med = np.array([statistics.median(per_iface[k]) for k in keys])
    p10 = np.array([np.percentile(per_iface[k], 10) for k in keys])
    p90 = np.array([np.percentile(per_iface[k], 90) for k in keys])
    p25 = np.array([np.percentile(per_iface[k], 25) for k in keys])
    p75 = np.array([np.percentile(per_iface[k], 75) for k in keys])
    iqr = p75 - p25
    x = np.arange(len(keys))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # --- Left: interfaces sorted by median, vertical 10-90 band + median dot ---
    axL.vlines(x, p10, p90, color="steelblue", alpha=0.5, linewidth=1.4, zorder=2)
    axL.vlines(x, p25, p75, color="steelblue", linewidth=3.2, zorder=3)
    axL.scatter(x, med, s=10, color="navy", zorder=4)
    # baseline (no masking/dropout) DockQ, where available, at each interface's slot
    bx = [xi for xi, k in zip(x, keys) if k in baseline]
    by = [baseline[k] for k in keys if k in baseline]
    axL.scatter(bx, by, marker="D", s=16, color="darkorange",
                edgecolor="black", linewidths=0.3, zorder=5)
    axL.axhline(args.threshold, color="red", linestyle="--", linewidth=1,
                label=f"DockQ={args.threshold}")
    axL.set_xlim(-1, len(keys))
    axL.set_ylim(0, 1)
    axL.set_xlabel("interfaces (sorted by median DockQ)")
    axL.set_ylabel("DockQ across seeds")
    axL.set_title(f"Per-interface seed spread  (n={len(keys)} interfaces)")
    axL.grid(True, axis="y", alpha=0.3)
    axL.legend(handles=[
        mlines.Line2D([], [], color="steelblue", linewidth=3.2, label="25–75th pct"),
        mlines.Line2D([], [], color="steelblue", alpha=0.5, linewidth=1.4, label="10–90th pct"),
        mlines.Line2D([], [], marker="o", color="navy", linestyle="None", markersize=5, label="median"),
        mlines.Line2D([], [], marker="D", color="darkorange", linestyle="None",
                      markersize=6, markeredgecolor="black", label="baseline"),
        mlines.Line2D([], [], color="red", linestyle="--", label=f"DockQ={args.threshold}"),
    ], loc="lower right", fontsize=8)

    # --- Right: how big is the seed effect, and where? median vs IQR spread ---
    axR.scatter(med, iqr, s=18, color="steelblue", alpha=0.6, zorder=3)
    axR.axvline(args.threshold, color="red", linestyle="--", linewidth=1)
    axR.set_xlim(0, 1)
    axR.set_ylim(0, max(iqr.max() * 1.05, 0.05))
    axR.set_xlabel("median DockQ (interface quality)")
    axR.set_ylabel("seed spread (IQR of DockQ)")
    axR.set_title("Seed sensitivity vs. quality")
    axR.grid(True, alpha=0.3)

    fig.suptitle(f"Seed-to-seed DockQ variation  (point = one {unit})", y=1.02)

    out = args.out or str(RESULTS_DIR / (csv_path.stem.replace("dockq_ost_", "seedvar_")
                                         + f"_{args.agg}_compact.png"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
