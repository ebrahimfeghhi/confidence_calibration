"""
Plot performance vs input-mask fraction across a sweep of mask CSVs.

Each CSV is one mask setting (e.g. dockq_ost_7txw_mask0.15_none_all_samples.csv);
the mask fraction is read from the filename (the number after "mask"). Within a
CSV every (interface, seed) is reduced to one DockQ via its highest-`--score-col`
sample (default pair_iptm) — the same confidence-based, oracle-free selection as
plot_scaling_ost.py / plot_scaling_compare.py. Each interface present in a CSV
becomes one line; the x-axis is the mask fraction so you can read off the trend.

Two panels:
  * mean DockQ  — mean over seeds of each seed's best-sample DockQ;
  * success rate — fraction of seeds with best-sample DockQ >= --threshold.
Bands are bootstrap SD over seeds (--n-boot resamples) at each mask fraction.

Usage:
    # defaults to the four 7txw mask CSVs (0.125 / 0.15 / 0.175 / 0.2)
    .venv/bin/python plot/plot_mask_sweep.py

    # explicit CSVs (sorted by mask fraction), custom selection/threshold
    .venv/bin/python plot/plot_mask_sweep.py A.csv B.csv C.csv \
        --score-col pair_iptm --threshold 0.23 --out results/mask_sweep.png
"""
import argparse
import random
import re
import statistics
import sys
from pathlib import Path

# --- repo-root bootstrap + sibling import of the scaling loader ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CSV_DIR, RESULTS_DIR  # noqa: E402
from plot_scaling_ost import load_data, resolve_col  # noqa: E402

# Filenames look like dockq_ost_7txw_mask0.15_none_all_samples.csv; grab the
# fraction right after "mask" (int or float).
_MASK_RE = re.compile(r"mask([0-9]*\.?[0-9]+)")


def mask_fraction(path):
    """Parse the mask fraction out of a CSV filename, or exit if absent."""
    m = _MASK_RE.search(Path(path).stem)
    if not m:
        sys.exit(f"{path}: filename has no 'mask<fraction>' to read the x value from")
    return float(m.group(1))


def seed_dockqs(csv_path, score_col):
    """{(assembly_id, chain1, chain2): [dockq per seed]} for one CSV.

    Each seed contributes the DockQ of its highest-`score_col` sample (load_data
    already does that per seed); baseline_seed rows and NaNs are dropped upstream.
    """
    by_iface, _ = load_data(csv_path, score_col=score_col)
    return {key: [dq for _seed, _rs, dq in rows] for key, rows in by_iface.items()}


def point(dqs, threshold, n_boot, rng):
    """Mean DockQ and success rate with bootstrap-over-seeds SDs for one cell.

    Returns (mean_dockq, dockq_sd, success_rate, success_sd). SDs come from
    resampling the seeds with replacement n_boot times.
    """
    n = len(dqs)
    mean_dq = statistics.mean(dqs)
    succ = sum(d >= threshold for d in dqs) / n
    if n < 2 or n_boot <= 0:
        return mean_dq, 0.0, succ, 0.0
    dq_draws, su_draws = [], []
    for _ in range(n_boot):
        sample = [dqs[rng.randrange(n)] for _ in range(n)]
        dq_draws.append(statistics.mean(sample))
        su_draws.append(sum(d >= threshold for d in sample) / n)
    return mean_dq, statistics.stdev(dq_draws), succ, statistics.stdev(su_draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*",
                    help="mask-sweep CSVs (default: the four 7txw mask CSVs in CSV_DIR)")
    ap.add_argument("--score-col", default="pair_iptm",
                    help="confidence column used to pick each seed's best sample "
                         "(default pair_iptm)")
    ap.add_argument("--threshold", type=float, default=0.23,
                    help="DockQ success threshold (default 0.23)")
    ap.add_argument("--n-boot", type=int, default=300,
                    help="bootstrap resamples over seeds for the SD bands (default 300)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(42)

    csvs = args.csvs or sorted(
        str(p) for p in CSV_DIR.glob("dockq_ost_7txw_mask*_none_all_samples.csv")
    )
    if not csvs:
        sys.exit("No mask CSVs given and none found in CSV_DIR matching "
                 "dockq_ost_7txw_mask*_none_all_samples.csv")
    # sort by parsed mask fraction so the x-axis is monotonic regardless of arg order
    csvs = sorted(csvs, key=mask_fraction)

    # series[(c1, c2)] = list of (fraction, mean_dq, dq_sd, succ, su_sd, n_seeds)
    series = {}
    fractions = []
    for path in csvs:
        frac = mask_fraction(path)
        fractions.append(frac)
        with open(path) as f:
            import csv as _csv
            fields = _csv.DictReader(f).fieldnames or []
        col = resolve_col(fields, args.score_col)
        if not col:
            sys.exit(f"{path}: no column for '{args.score_col}' (have: {fields})")
        for key, dqs in seed_dockqs(path, col).items():
            if not dqs:
                continue
            m_dq, dq_sd, succ, su_sd = point(dqs, args.threshold, args.n_boot, rng)
            series.setdefault(key[1:], []).append((frac, m_dq, dq_sd, succ, su_sd, len(dqs)))

    if not series:
        sys.exit("No interface data loaded from the given CSVs.")

    # report
    print(f"Mask sweep over {len(csvs)} CSVs   select by {args.score_col}   "
          f"DockQ>={args.threshold}   bootstrap={args.n_boot} (over seeds)")
    for (c1, c2), pts in sorted(series.items()):
        print(f"\n[{c1}|{c2}]")
        print(f"  {'mask':>6} {'seeds':>6}  {'meanDockQ':>16}  {'success':>14}")
        for frac, m_dq, dq_sd, succ, su_sd, n in sorted(pts):
            print(f"  {frac:>6.3g} {n:>6}  {m_dq:>7.3f} ± {dq_sd:<5.3f}  "
                  f"{succ*100:>5.1f}% ± {su_sd*100:<4.1f}")

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for i, (key, pts) in enumerate(sorted(series.items())):
        c1, c2 = key
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        color = cmap(i % 10)
        label = f"{c1}|{c2}"

        m_dq = [p[1] for p in pts]; dq_sd = [p[2] for p in pts]
        axL.plot(xs, m_dq, "-o", color=color, label=label)
        axL.fill_between(xs, [m - s for m, s in zip(m_dq, dq_sd)],
                         [m + s for m, s in zip(m_dq, dq_sd)], color=color, alpha=0.15)

        su = [p[3] * 100 for p in pts]; su_sd = [p[4] * 100 for p in pts]
        axR.plot(xs, su, "-o", color=color, label=label)
        axR.fill_between(xs, [m - s for m, s in zip(su, su_sd)],
                         [m + s for m, s in zip(su, su_sd)], color=color, alpha=0.15)

    xticks = sorted(set(fractions))
    for ax in (axL, axR):
        ax.set_xlabel("input mask fraction")
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axL.set_ylabel("mean DockQ (best sample/seed by %s)" % args.score_col)
    axL.set_title("Mean DockQ vs mask fraction")
    axR.set_ylabel(f"DockQ ≥ {args.threshold:g} success rate (%)")
    axR.set_title(f"Success@{args.threshold:g} vs mask fraction")

    fig.suptitle("Mask-fraction sweep  (band = bootstrap SD over seeds)", y=1.02)
    out = args.out or str(RESULTS_DIR / "mask_sweep.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
