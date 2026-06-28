"""
Overlay test-time-scaling curves for TWO DockQ runs on the subset they share.

For a budget of N seeds you keep the single generation the confidence head likes
best and read off its DockQ. This plots, vs N, the resulting (a) mean DockQ and
(b) success rate (DockQ >= threshold), for both runs on the SAME interfaces and
seeds, so the two curves are directly comparable.

Method (same bootstrap as plot_scaling_ost.py):
  * within each (interface, seed) keep the diffusion sample with the highest
    --score-col (default pair_iptm)  -> one (score, dockq) per seed;
  * compare only interfaces present in BOTH runs, and per interface only the
    seeds present in BOTH (if seed counts differ, the extra seeds are ignored);
  * for each N: draw N random seeds, keep the most-confident one's DockQ, repeat
    --n-boot times and average. Done per interface, then averaged over interfaces.

Usage:
    .venv/bin/python plot/plot_scaling_compare.py A.csv B.csv
    .venv/bin/python plot/plot_scaling_compare.py A.csv B.csv \
        --label-a mask10_nodrop --label-b tts --score-col pair_iptm
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

# --- repo-root bootstrap + sibling import of the scaling loader ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS_DIR
from plot_scaling_ost import load_data, resolve_col  # noqa: E402


def intersect(by_a, by_b):
    """Restrict both runs to shared interfaces and, per interface, shared seeds.

    Returns rows_a, rows_b: {key: [(score, dockq), ...]} over the common seeds,
    keyed identically, dropping interfaces with no common seed.
    """
    rows_a, rows_b = {}, {}
    for key in set(by_a) & set(by_b):
        a = {seed: (s, dq) for seed, s, dq in by_a[key]}
        b = {seed: (s, dq) for seed, s, dq in by_b[key]}
        common = set(a) & set(b)
        if not common:
            continue
        rows_a[key] = [a[s] for s in common]
        rows_b[key] = [b[s] for s in common]
    return rows_a, rows_b


def _mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def scaling_curve(rows, ns, threshold, n_boot, rng):
    """Bootstrap best-of-N-by-score, macro-averaged over interfaces.

    For each N: draw N random seeds for each interface, keep the most-confident
    seed's DockQ, and average those DockQs over interfaces -> one number per draw.
    Repeat n_boot times; report the mean and SD across draws (same as
    plot_scaling_ost.compute_tts, but for mean DockQ as well as success rate).

    Returns dq (mean DockQ), dq_sd, su (success rate), su_sd, n_ifaces -- one
    entry per N.
    """
    dq, dq_sd, su, su_sd, n_ifaces = [], [], [], [], []
    for n in ns:
        eligible = [v for v in rows.values() if len(v) >= n]
        if not eligible:
            continue
        dq_draws, su_draws = [], []
        for _ in range(n_boot):
            picked = [max(rng.sample(v, n), key=lambda x: x[0])[1] for v in eligible]
            dq_draws.append(sum(picked) / len(picked))
            su_draws.append(sum(d >= threshold for d in picked) / len(picked))
        m_dq, s_dq = _mean_sd(dq_draws)
        m_su, s_su = _mean_sd(su_draws)
        dq.append(m_dq); dq_sd.append(s_dq)
        su.append(m_su); su_sd.append(s_su)
        n_ifaces.append(len(eligible))
    return dq, dq_sd, su, su_sd, n_ifaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--score-col", default="pair_iptm",
                    help="confidence column used to select samples/seeds (default pair_iptm)")
    ap.add_argument("--threshold", type=float, default=0.23)
    ap.add_argument("--ns", type=int, nargs="+", default=None,
                    help="seed budgets to plot (default: every N from 1..min common "
                         "seed count, so the interface set is fixed across the curve)")
    ap.add_argument("--n-boot", type=int, default=300,
                    help="bootstrap draws per N (default 300)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(42)

    label_a = args.label_a or Path(args.csv_a).stem.replace("dockq_ost_", "")
    label_b = args.label_b or Path(args.csv_b).stem.replace("dockq_ost_", "")

    # resolve the score column against each CSV's header (handles aliases)
    def loaded(path):
        import csv as _csv
        with open(path) as f:
            fields = _csv.DictReader(f).fieldnames or []
        col = resolve_col(fields, args.score_col)
        if not col:
            sys.exit(f"{path}: no column for '{args.score_col}' (have: {fields})")
        by_iface, _ = load_data(path, score_col=col)
        return by_iface, col

    by_a, col_a = loaded(args.csv_a)
    by_b, col_b = loaded(args.csv_b)
    rows_a, rows_b = intersect(by_a, by_b)

    if not rows_a:
        sys.exit("No shared interface with overlapping seeds; nothing to plot.")

    seed_counts = [len(v) for v in rows_a.values()]
    min_s, max_s = min(seed_counts), max(seed_counts)
    ns = sorted(args.ns) if args.ns else list(range(1, min_s + 1))
    ns = [n for n in ns if n <= max_s]

    da, da_sd, sa, sa_sd, ifa = scaling_curve(rows_a, ns, args.threshold, args.n_boot, rng)
    db, db_sd, sb, sb_sd, ifb = scaling_curve(rows_b, ns, args.threshold, args.n_boot, rng)
    # both runs share interfaces and seed counts, so ns/eligible counts match
    ns_used = ns[:len(da)]

    print(f"Run A: {label_a}  (score={col_a})")
    print(f"Run B: {label_b}  (score={col_b})")
    print(f"Shared interfaces: {len(rows_a)}   common seeds/iface: "
          f"min={min_s} max={max_s}   threshold DockQ>={args.threshold}   "
          f"bootstrap={args.n_boot} draws\n")
    print(f"{'N':>4} {'ifaces':>6}  {'meanDockQ_A':>13} {'meanDockQ_B':>13}   "
          f"{'succA':>11} {'succB':>11}   (value ± boot SD)")
    for i, n in enumerate(ns_used):
        print(f"{n:>4} {ifa[i]:>6}  {da[i]:>7.3f}±{da_sd[i]:<5.3f} {db[i]:>7.3f}±{db_sd[i]:<5.3f}   "
              f"{sa[i]:>5.3f}±{sa_sd[i]:<5.3f} {sb[i]:>5.3f}±{sb_sd[i]:<5.3f}")

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    ca, cb = "#1f77b4", "#d62728"

    def band(ax, xs, ys, sds, color, scale=1.0):
        ys = [y * scale for y in ys]
        sds = [s * scale for s in sds]
        ax.fill_between(xs, [y - s for y, s in zip(ys, sds)],
                        [y + s for y, s in zip(ys, sds)], color=color, alpha=0.15)

    axL.plot(ns_used, da, "-o", color=ca, label=label_a)
    axL.plot(ns_used, db, "-o", color=cb, label=label_b)
    band(axL, ns_used, da, da_sd, ca)
    band(axL, ns_used, db, db_sd, cb)
    axL.set_xlabel("# of seeds (N)")
    axL.set_ylabel("mean DockQ (confidence-selected best of N)")
    axL.set_title("Mean DockQ vs N")
    axL.grid(True, alpha=0.3)
    axL.legend()

    axR.plot(ns_used, [v * 100 for v in sa], "-o", color=ca, label=label_a)
    axR.plot(ns_used, [v * 100 for v in sb], "-o", color=cb, label=label_b)
    band(axR, ns_used, sa, sa_sd, ca, scale=100)
    band(axR, ns_used, sb, sb_sd, cb, scale=100)
    axR.set_xlabel("# of seeds (N)")
    axR.set_ylabel(f"DockQ ≥ {args.threshold:g} success rate (%)")
    axR.set_title(f"Success@{args.threshold:g} vs N")
    axR.grid(True, alpha=0.3)
    axR.legend()

    fig.suptitle(f"TTS scaling on shared subset  "
                 f"(n={len(rows_a)} interfaces, select by {col_a})", y=1.02)

    out = args.out or str(RESULTS_DIR / f"scaling_compare_{label_a}_vs_{label_b}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
