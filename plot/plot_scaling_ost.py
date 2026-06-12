"""
Plot TTS scaling curve from dockq_ost.csv (FoldBench OST evaluation).

For N seeds: draw 300 bootstrap samples of N seeds per interface,
pick best by score_col, compute pass rate (DockQ >= threshold).

One curve per available selection score (pair_ranking_score, global_ranking_score,
pair_iptm, global_iptm); older CSVs using new_ranking_score/iptm are read via aliases.
Only N values where every N=1 interface has >= N seeds are plotted.

Usage:
    .venv/bin/python plot_scaling_ost.py --csv path/to/dockq.csv
    .venv/bin/python plot_scaling_ost.py --threshold 0.23 --out results/tts.png
"""
import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import RESULTS_DIR

# Map each canonical score column to the names it may appear under (new schema
# first, then older aliases) so plots work on both old and new eval CSVs.
SCORE_ALIASES = {
    "pair_ranking_score":   ["pair_ranking_score", "new_ranking_score"],
    "global_ranking_score": ["global_ranking_score"],
    "pair_iptm":            ["pair_iptm"],
    "global_iptm":          ["global_iptm", "iptm"],
}


def resolve_col(fieldnames, canonical):
    """Return the column actually present in the CSV for a canonical score name."""
    for name in SCORE_ALIASES.get(canonical, [canonical]):
        if name in fieldnames:
            return name
    return None


def load_data(csv_path, score_col):
    """Load CSV and return (by_iface, baseline_by_iface).

    by_iface: {(assembly_id, chain1, chain2): [(seed, score, dockq)]}
      - excludes baseline_seed rows; picks best sample per seed by score_col.

    baseline_by_iface: {(assembly_id, chain1, chain2): (score, dockq)}
      - best sample from baseline_seed per interface, or empty if none present.
    """
    raw          = defaultdict(lambda: defaultdict(list))
    baseline_raw = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dq   = float(row["dockq"])
                rs   = float(row[score_col])
                seed = row["seed"]
            except (ValueError, KeyError):
                continue
            if dq != dq or rs != rs:  # skip NaN
                continue
            key = (row["assembly_id"], row["chain1"], row["chain2"])
            if seed == "baseline_seed":
                baseline_raw[key].append((rs, dq))
            else:
                raw[key][seed].append((rs, dq))

    by_iface = defaultdict(list)
    for key, seeds in raw.items():
        for seed, samples in seeds.items():
            if samples:
                best_rs, best_dq = max(samples, key=lambda x: x[0])
                by_iface[key].append((seed, best_rs, best_dq))

    baseline_by_iface = {
        key: max(samples, key=lambda x: x[0])
        for key, samples in baseline_raw.items()
        if samples
    }
    return by_iface, baseline_by_iface


def compute_tts(by_iface, ns, n_bootstrap=300, threshold=0.23):
    n1_count = len(by_iface)
    results  = []
    for n in ns:
        eligible = {k: v for k, v in by_iface.items() if len(v) >= n}
        if len(eligible) < n1_count:
            continue  # not all interfaces have enough seeds — skip this N
        if n == 1:
            # Mean over interfaces of each interface's pass fraction — the analytic
            # n=1 bootstrap. Weights interfaces equally (matching n>=2) instead of by
            # seed count, so the n=1 point stays comparable when seed counts differ.
            per_iface = [sum(1 for _, _, dq in rows if dq >= threshold) / len(rows)
                         for rows in eligible.values()]
            mean_pr = sum(per_iface) / len(per_iface)
            results.append({"n": n, "pass_rate": mean_pr, "sd": 0.0, "n_ifaces": len(eligible)})
        else:
            pass_rates = []
            for _ in range(n_bootstrap):
                successes = sum(
                    1 for rows in eligible.values()
                    if max(random.sample(rows, n), key=lambda x: x[1])[2] >= threshold
                )
                pass_rates.append(successes / len(eligible))
            mean_pr = sum(pass_rates) / len(pass_rates)
            sd_pr   = math.sqrt(sum((x - mean_pr) ** 2 for x in pass_rates) / max(len(pass_rates) - 1, 1))
            results.append({"n": n, "pass_rate": mean_pr, "sd": sd_pr, "n_ifaces": len(eligible)})
    return results


def baseline_pass_rate(baseline_by_iface, threshold):
    if not baseline_by_iface:
        return None
    dqs = [dq for _, dq in baseline_by_iface.values()]
    return sum(1 for dq in dqs if dq >= threshold) / len(dqs)


def print_seed_summary(by_iface, ns):
    """Show the seed-count landscape so you can choose --ns sensibly.

    Strict mode only plots an N where EVERY interface has >= N seeds, so the
    curve's max N is the minimum seed count across interfaces. This prints, for
    each requested N, how many interfaces qualify and whether that N will plot,
    then names the interfaces holding the curve back.
    """
    counts = sorted(len(v) for v in by_iface.values())
    n_ifaces = len(counts)
    if not n_ifaces:
        print("\nNo interfaces with seeds loaded.")
        return
    min_seeds = counts[0]
    print(f"\nSeed counts across {n_ifaces} interfaces: "
          f"min={min_seeds}  median={counts[n_ifaces // 2]}  max={counts[-1]}")
    print(f"  {'N':>5}  {'interfaces w/ >= N':>18}")
    for n in sorted(set(ns)):
        k = sum(1 for c in counts if c >= n)
        flag = "all (plotted)" if k == n_ifaces else f"{n_ifaces - k} short (dropped)"
        print(f"  {n:>5}  {k:>18}   {flag}")
    plotted = [n for n in sorted(ns) if n <= min_seeds]
    print(f"  -> strict curve reaches N={max(plotted) if plotted else None} "
          f"(min seed count = {min_seeds})")
    laggards = sorted(((len(v), k) for k, v in by_iface.items() if len(v) < counts[-1]),
                      key=lambda x: x[0])[:6]
    if laggards:
        print("  interfaces below max seed count:")
        for cnt, (aid, c1, c2) in laggards:
            print(f"      {cnt:>3} seeds   {aid} {c1}|{c2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="foldbench_abag/Protenix_dockq.csv")
    parser.add_argument("--out",       default=None)
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--ns",        type=int, nargs="+", default=[1, 2, 5, 10, 20, 30, 50])
    args = parser.parse_args()

    csv_path = Path(args.csv)
    NS       = sorted(args.ns)
    out_path = args.out or str(
        RESULTS_DIR / (csv_path.stem.replace("dockq_ost_", "tts_")
                       + "_n" + "_".join(str(n) for n in NS) + ".png")
    )

    random.seed(42)

    with open(csv_path) as f:
        fieldnames = csv.DictReader(f).fieldnames or []

    SCORE_COLS = [
        ("pair_ranking_score",   "blue"),
        ("global_ranking_score", "green"),
        ("pair_iptm",            "orange"),
        ("global_iptm",          "red"),
    ]
    curves = []  # (canonical, actual_column, color)
    for canonical, color in SCORE_COLS:
        actual = resolve_col(fieldnames, canonical)
        if actual:
            curves.append((canonical, actual, color))
    if not curves:
        curves = [("ranking_score", "ranking_score", "blue")]

    all_results = {}
    baseline = None
    baseline_color = "blue"
    for canonical, actual, color in curves:
        by_iface, bl = load_data(csv_path, score_col=actual)
        print(f"Interfaces loaded: {len(by_iface)}  (scored by {canonical})")
        all_results[canonical] = (by_iface, compute_tts(by_iface, NS, threshold=args.threshold))
        if canonical == "pair_ranking_score":
            baseline, baseline_color = bl, color

    # Seed-count landscape (based on the primary score column) so you can pick --ns.
    print_seed_summary(all_results[curves[0][0]][0], NS)

    for canonical, _actual, _color in curves:
        _, results = all_results[canonical]
        print(f"\n[{canonical}]")
        print(f"{'N':>4}  {'Pass Rate':>10}  {'±SD':>6}  {'Interfaces':>12}")
        print("-" * 40)
        for r in results:
            print(f"{r['n']:>4}  {r['pass_rate']:>10.1%}  {r['sd']:>6.3f}  {r['n_ifaces']:>12}")

    bpr = baseline_pass_rate(baseline, args.threshold)
    if bpr is not None:
        print(f"\nBaseline (no masking/dropout): {bpr:.1%}  ({len(baseline)} interfaces)")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        for canonical, actual, color in curves:
            _, results = all_results[canonical]
            n_vals = [r["n"] for r in results]
            pr     = [r["pass_rate"] for r in results]
            sd     = [r["sd"] for r in results]
            LABELS = {
                "pair_ranking_score":   "0.8*pair_iptm + 0.2*ptm - clash",
                "global_ranking_score": "0.8*global_iptm + 0.2*ptm - clash",
                "pair_iptm":            "pair_iptm",
                "global_iptm":          "global_iptm",
            }
            label = LABELS.get(canonical, canonical)
            ax.plot(n_vals, [p * 100 for p in pr], "-o", color=color, markersize=6, label=label)
            ax.fill_between(n_vals,
                            [(p - s) * 100 for p, s in zip(pr, sd)],
                            [(p + s) * 100 for p, s in zip(pr, sd)],
                            alpha=0.15, color=color)

        if bpr is not None:
            ax.plot(1, bpr * 100, "*", color=baseline_color, markersize=14,
                    zorder=5, label="baseline (no mask/dropout)")
        ax.legend()

        first_results = all_results[curves[0][0]][1]
        n_ifaces = first_results[0]["n_ifaces"] if first_results else "?"
        ax.set_xscale("log")
        # Union of N across all curves (each curve may drop different N values when
        # its interface set differs), so ticks aren't just the last curve's.
        all_n = sorted({r["n"] for _, results in all_results.values() for r in results})
        ax.set_xticks(all_n)
        ax.set_xticklabels([str(n) for n in all_n])
        ax.set_xlabel("# of seeds")
        ax.set_ylabel(f"DockQ ≥ {args.threshold:.2f} Success Rate (%)")
        ax.set_title(f"{csv_path.stem}  (n={n_ifaces} interfaces)")
        ax.grid(True, alpha=0.3)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")

        # --- strip plot: jittered DockQ dots per interface + baseline star ---
        by_iface_bar = all_results[curves[0][0]][0]
        iface_means  = {k: sum(dq for _, _, dq in v) / len(v)
                        for k, v in by_iface_bar.items()}
        sorted_ifaces = sorted(iface_means, key=lambda k: iface_means[k], reverse=True)
        labels        = [f"{aid}\n{c1}|{c2}" for aid, c1, c2 in sorted_ifaces]

        fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_ifaces) * 0.5), 5))
        rng = random.Random(0)
        for xi, key in enumerate(sorted_ifaces):
            dqs = [dq for _, _, dq in by_iface_bar[key]]
            jitter = [xi + rng.uniform(-0.25, 0.25) for _ in dqs]
            ax2.scatter(jitter, dqs, color="steelblue", alpha=0.5, s=18, zorder=3)
            if baseline:
                entry = baseline.get(key)
                if entry is not None:
                    ax2.scatter([xi], [entry[1]], marker="*", color="gray",
                                s=150, zorder=5)

        ax2.axhline(args.threshold, color="red", linestyle="--", linewidth=1,
                    label=f"DockQ={args.threshold}")
        # legend proxies
        import matplotlib.lines as mlines
        ax2.legend(handles=[
            plt.scatter([], [], color="steelblue", alpha=0.6, s=18, label="seeds"),
            mlines.Line2D([], [], marker="*", color="gray", linestyle="None",
                          markersize=10, label="baseline"),
            mlines.Line2D([], [], color="red", linestyle="--", label=f"DockQ={args.threshold}"),
        ])
        ax2.set_xticks(range(len(sorted_ifaces)))
        ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax2.set_ylabel("DockQ")
        ax2.set_title(f"{csv_path.stem} — DockQ per interface")
        ax2.grid(True, axis="y", alpha=0.3)

        bar_path = out_path.replace(".png", "_barplot.png")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        print(f"Barplot saved to {bar_path}")

    except Exception as e:
        print(f"Could not plot: {e}")


if __name__ == "__main__":
    main()
