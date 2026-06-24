"""Plot s_lm deviation (from baseline and from the real-seed cloud) for 5- vs 15-step
sequential repulsion, annotated with the DockQ we have for the scored runs.

Reads:  results/csv_files/steer_deviation_7txw.csv   (from steer_deviation.py)
        results/csv_files/dockq_ost_7txw_seqrepel.csv (A-C ipTM-selected DockQ)
Writes: analysis/figs/steer_deviation.png

Run:    uv run python analysis/grad_steer/plot_deviation.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, REPO_ROOT  # noqa: E402

# steer_deviation.py writes a combined file when run over both step counts, or
# per-step-count files (_5step/_15step) when run one at a time — accept either.
DEV_CSVS = [CSV_DIR / "steer_deviation_7txw.csv",
            CSV_DIR / "steer_deviation_7txw_5step.csv",
            CSV_DIR / "steer_deviation_7txw_15step.csv"]
DOCKQ_CSV = CSV_DIR / "dockq_ost_7txw_seqrepel.csv"
FIG = REPO_ROOT / "analysis" / "figs" / "steer_deviation.png"


def load_dev_rows():
    """Merge whichever deviation CSVs exist; first (steps, run) seen wins."""
    rows, seen = [], set()
    for p in DEV_CSVS:
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            key = (r["steps"], r["run"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_dockq():
    """(schedule, run) -> ipTM-selected A-C DockQ."""
    by = defaultdict(list)
    if not DOCKQ_CSV.exists():
        return {}
    with open(DOCKQ_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["chain1"] == "A" and r["chain2"] == "C":
                by[(r["schedule"], r["run"])].append(r)
    out = {}
    for key, d in by.items():
        sel = max(d, key=lambda r: f(r["pair_iptm"]) or -1)
        out[key] = f(sel["dockq"])
    return out


def main():
    rows = load_dev_rows()
    base = next((r for r in rows if r["run"] == "-1"), None)
    base_cloud_z = f(base["cloud_z"]) if base else None

    data = defaultdict(dict)   # steps -> run -> row
    for r in rows:
        if r["run"] == "-1":
            continue
        data[int(r["steps"])][int(r["run"])] = r

    dockq = load_dockq()
    scored = {(5, 2): ("5step", "run2"), (5, 8): ("5step", "run8"),
              (15, 1): ("15step", "run1"), (15, 7): ("15step", "run7")}

    steps_order = sorted(data)            # [5, 15]
    xpos = {s: i for i, s in enumerate(steps_order)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in (
        (axes[0], "dev_baseline", "Deviation from baseline  ||s-s0||/||s0||"),
        (axes[1], "cloud_z", "Distance outside real-seed cloud  (z, cloud-widths)")):
        for s in steps_order:
            xs, ys = [], []
            for run, r in sorted(data[s].items()):
                y = f(r[metric])
                x = xpos[s] + (run - 4.5) * 0.012
                xs.append(x); ys.append(y)
                tag = scored.get((s, run))
                if tag and tag in dockq and dockq[tag] is not None:
                    ax.annotate(f"r{run}\nDockQ={dockq[tag]:.2f}", (x, y),
                                fontsize=8, ha="left", va="center",
                                xytext=(6, 0), textcoords="offset points",
                                color=("green" if dockq[tag] >= 0.8 else
                                       "orange" if dockq[tag] >= 0.49 else "red"))
            ax.scatter(xs, ys, s=40, zorder=3,
                       color=("tab:blue" if s == 5 else "tab:red"),
                       label=f"{s}-step")
            ax.plot(xpos[s], sum(v for v in ys) / len(ys), "_", ms=40,
                    color="black", zorder=2)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([f"{s}-step" for s in steps_order])
        ax.set_xlim(-0.5, len(steps_order) - 0.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if metric == "cloud_z":
            ax.axhspan(-2, 2, color="green", alpha=0.08, label="real-seed cloud (|z|<2)")
            if base_cloud_z is not None:
                ax.axhline(base_cloud_z, ls="--", color="gray",
                           label=f"baseline (z={base_cloud_z:.1f})")
        ax.legend(fontsize=8)

    fig.suptitle("Does 15-step repulsion push s_lm off the network's input manifold?",
                 fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")

    # console summary
    print("\nmean by schedule:")
    for s in steps_order:
        dv = [f(r["dev_baseline"]) for r in data[s].values()]
        cz = [f(r["cloud_z"]) for r in data[s].values()]
        print(f"  {s:>2}-step: dev_baseline {sum(dv)/len(dv):.3f}   "
              f"cloud_z {sum(cz)/len(cz):.2f}")
    print("\nscored runs (z_match should be ~0):")
    for (s, run), tag in scored.items():
        r = data.get(s, {}).get(run)
        if r:
            print(f"  {s:>2}-step run{run}: dev_baseline={r['dev_baseline']} "
                  f"cloud_z={r['cloud_z']} z_match={r['z_match']} "
                  f"basin={r['basin']} DockQ={dockq.get(tag)}")


if __name__ == "__main__":
    main()
