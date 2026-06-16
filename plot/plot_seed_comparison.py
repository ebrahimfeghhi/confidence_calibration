"""
Compare seed-to-seed DockQ between two methods per interface, as a grouped bar
chart, using only the seeds shared by both CSVs for each interface.

Each seed is reduced to one DockQ via its most-confident sample (highest
--score-col, default pair_iptm) — the same best-sample-per-seed selection as
plot_scaling_ost.py. For every interface (assembly_id, chain1, chain2) present in
BOTH CSVs we take the seeds present in both; the plot draws one group per
interface with two adjacent bars (one per CSV), each bar = mean over the shared
seeds, with every shared seed overlaid as a jittered point.

Seed intersection is per interface: if both CSVs have 50 seeds for an interface,
all 50 are used; if one CSV has only 10, the 10 shared seeds are used.

Usage:
    .venv/bin/python plot/plot_seed_comparison.py A.csv B.csv
    .venv/bin/python plot/plot_seed_comparison.py A.csv B.csv --label-a tts --label-b wperturb
"""
import argparse
import csv
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


def load(csv_path, score_col):
    """Return (seeds, baseline, resolved_column).

    seeds:    {(assembly_id, chain1, chain2): {seed: dockq}} — per (interface,
              seed) the DockQ of the seed's highest-`score_col` sample.
    baseline: {(assembly_id, chain1, chain2): dockq} — the baseline_seed
              (no mask/perturb) interface DockQ, empty if the CSV has none.
    NaN rows are dropped by load_data.
    """
    with open(csv_path) as f:
        fields = csv.DictReader(f).fieldnames or []
    col = resolve_col(fields, score_col)
    if not col:
        sys.exit(f"{csv_path}: no column for '{score_col}' (have: {fields})")
    by_iface, baseline = load_data(csv_path, score_col=col)
    seeds = {key: {seed: dq for seed, _rs, dq in rows} for key, rows in by_iface.items()}
    baseline_dq = {key: dq for key, (_rs, dq) in baseline.items()}
    return seeds, baseline_dq, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--score-col", default="pair_iptm",
                    help="confidence column used to pick each seed's best sample "
                         "(default pair_iptm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label_a = args.label_a or Path(args.csv_a).stem.replace("dockq_ost_", "")
    label_b = args.label_b or Path(args.csv_b).stem.replace("dockq_ost_", "")

    A, base_a, col_a = load(args.csv_a, args.score_col)
    B, base_b, col_b = load(args.csv_b, args.score_col)

    # baseline_seed (no mask/perturb) is the same unperturbed reference for both
    # methods; a CSV may carry it for only one. Merge, preferring A's value.
    baseline = {**base_b, **base_a}

    # one group per interface shared by both CSVs, using per-interface shared seeds
    ifaces = []  # (key, common_seeds, values_a, values_b)
    for key in sorted(set(A) & set(B)):
        common = sorted(set(A[key]) & set(B[key]))
        if not common:
            continue
        ifaces.append((key, common,
                       [A[key][s] for s in common],
                       [B[key][s] for s in common]))

    if not ifaces:
        sys.exit("No interface shared between the two CSVs with shared seeds; "
                 "nothing to plot.")

    n_base = sum(1 for key, *_ in ifaces if key in baseline)
    print(f"Interfaces shared: {len(ifaces)}   "
          f"point = one seed (best sample by {col_a})   "
          f"baseline available for {n_base}/{len(ifaces)}")
    print(f"{'interface':28s} {'shared':>6}  {label_a:>10} mean  {label_b:>10} mean  {'baseline':>8}")
    for key, common, va, vb in ifaces:
        aid, c1, c2 = key
        bdq = baseline.get(key)
        bstr = f"{bdq:>8.3f}" if bdq is not None else f"{'-':>8}"
        print(f"  {aid} {c1}|{c2:8s}  {len(common):>4}   "
              f"{statistics.mean(va):>10.3f}    {statistics.mean(vb):>10.3f}  {bstr}   "
              f"(A {min(va):.3f}-{max(va):.3f} / B {min(vb):.3f}-{max(vb):.3f})")

    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    colors = ["#1f77b4", "#d62728"]
    n = len(ifaces)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, n * 1.6), 5.5))

    mean_a = [statistics.mean(va) for _, _, va, _ in ifaces]
    mean_b = [statistics.mean(vb) for _, _, _, vb in ifaces]
    ax.bar(x - w / 2, mean_a, w, color=colors[0], alpha=0.45, label=label_a, zorder=2)
    ax.bar(x + w / 2, mean_b, w, color=colors[1], alpha=0.45, label=label_b, zorder=2)

    for i, (_, _, va, vb) in enumerate(ifaces):
        for center, vals, c in ((x[i] - w / 2, va, colors[0]),
                                (x[i] + w / 2, vb, colors[1])):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), center) + jitter, vals, s=20, color=c,
                       alpha=0.7, edgecolor="white", linewidths=0.4, zorder=3)

    # baseline (no mask/perturb): a horizontal dash spanning each interface's group
    drawn_base = False
    for i, (key, _, _, _) in enumerate(ifaces):
        bdq = baseline.get(key)
        if bdq is None:
            continue
        ax.hlines(bdq, x[i] - w, x[i] + w, color="black", linewidth=2.2,
                  zorder=4, label="baseline (no mask/perturb)" if not drawn_base else None)
        drawn_base = True

    plotted_base = [baseline[key] for key, *_ in ifaces if key in baseline]
    data_max = max([v for _, _, va, vb in ifaces for v in (va + vb)] + plotted_base)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{aid}\n{c1}|{c2}\n(n={len(common)})"
                        for (aid, c1, c2), common, _, _ in ifaces])
    ax.set_ylabel("DockQ")
    ax.set_ylim(0, data_max * 1.25)
    ax.set_title(f"DockQ comparison — best seed by {col_a}", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = args.out or str(RESULTS_DIR / f"seed_compare_{label_a}_vs_{label_b}_{col_a}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
