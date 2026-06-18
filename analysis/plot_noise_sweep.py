"""Idea-1 noise sweep: does a suppressed good basin reappear from a committed `z`?

For each diffusion noise_scale, reduce each seed's many A|C samples to one DockQ
(default: max over samples = "did the good basin appear at all"), then draw one bar
per noise scale at the mean over seeds, with a dot per seed colored by the seed's
baseline class (good / mid / bad). The Idea-1 signal is whether the *bad*-seed dots
climb as noise increases.

Reads CSV_DIR/dockq_7txw_noise{ns}.csv (from eval/score_7txw_noise.py) and, for the
noise=0 reference + seed classification, CSV_DIR/dockq_ost_7txw_reps_all_samples.csv.
Output: analysis/figs/noise_sweep_7txw.png

Usage:
    python analysis/plot_noise_sweep.py                 # reduce=max, include noise=0
    python analysis/plot_noise_sweep.py --reduce topconf
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR

OUT = Path(__file__).resolve().parent / "figs" / "noise_sweep_7txw.png"
REPS_CSV = CSV_DIR / "dockq_ost_7txw_reps_all_samples.csv"
GOOD_T, BAD_T = 0.5, 0.2          # class thresholds on baseline max A|C DockQ
RESCUE_T = 0.23                   # CAPRI "acceptable" — a committed-bad seed counts rescued
CLASS_COLOR = {"good": "#2ca02c", "mid": "#ff7f0e", "bad": "#7f7f7f"}


def load_ac(csv_path):
    """{seed: [(dockq, pair_iptm), ...]} for the A|C interface."""
    out = defaultdict(list)
    if not Path(csv_path).exists():
        return out
    for r in csv.DictReader(open(csv_path)):
        if r.get("chain1") != "A" or r.get("chain2") != "C":
            continue
        try:
            s, dq, ip = int(r["seed"]), float(r["dockq"]), float(r["pair_iptm"])
        except (ValueError, KeyError):
            continue
        if dq != dq:                       # skip NaN dockq (OST failure)
            continue
        out[s].append((dq, ip))
    return out


def reduce_seed(samples, how):
    """samples: [(dockq, pair_iptm)] -> one scalar DockQ."""
    if not samples:
        return np.nan
    dqs = [d for d, _ in samples]
    if how == "max":
        return max(dqs)
    if how == "mean":
        return float(np.mean(dqs))
    if how == "topconf":                   # DockQ of the max-pair_iptm sample
        return max(samples, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--reduce", choices=["max", "mean", "topconf"], default="max")
    ap.add_argument("--no_baseline", action="store_true",
                    help="omit the noise=0 reps reference column")
    args = ap.parse_args()

    # --- classify seeds from the baseline reps run (max A|C DockQ over its samples)
    reps = load_ac(REPS_CSV)
    cls = {}
    for s, samp in reps.items():
        mx = max(d for d, _ in samp)
        cls[s] = "good" if mx >= GOOD_T else ("bad" if mx < BAD_T else "mid")
    n_bad = sum(v == "bad" for v in cls.values())

    # --- columns: optional noise=0 (reps) then each scored noise CSV that exists
    cols = []                              # (label, {seed: reduced_dockq})
    if not args.no_baseline and reps:
        cols.append(("0\n(reps)", {s: reduce_seed(reps[s], args.reduce) for s in reps}))
    for ns in args.noise:
        tag = f"{ns:.1f}"
        data = load_ac(CSV_DIR / f"dockq_7txw_noise{tag}.csv")
        if not data:
            print(f"noise {tag}: no scored CSV yet, skipping column", flush=True)
            continue
        cols.append((tag, {s: reduce_seed(v, args.reduce) for s, v in data.items()}))

    if not cols:
        print("No data to plot yet. Run eval/score_7txw_noise.py after the sweep.")
        return

    # --- plot
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 6))
    rng = np.random.default_rng(0)
    xs = np.arange(len(cols))
    for x, (label, seedvals) in zip(xs, cols):
        vals = np.array([v for v in seedvals.values() if v == v])
        ax.bar(x, vals.mean(), width=0.62, color="0.85", edgecolor="0.5", zorder=1)
        # per-seed dots, colored by class, jittered
        for s, v in seedvals.items():
            if v != v:
                continue
            c = CLASS_COLOR.get(cls.get(s, "bad"), "#7f7f7f")
            ax.scatter(x + rng.uniform(-0.22, 0.22), v, s=22, color=c,
                       alpha=0.8, edgecolor="none", zorder=3)
        # Idea-1 signal: mean over BAD seeds + rescue count
        bad_vals = [v for s, v in seedvals.items() if cls.get(s) == "bad" and v == v]
        if bad_vals:
            ax.scatter(x, np.mean(bad_vals), marker="D", s=55, color="crimson",
                       edgecolor="k", lw=0.6, zorder=4)
            rescued = sum(v >= RESCUE_T for v in bad_vals)
            ax.text(x, 1.02, f"{rescued}/{len(bad_vals)}", ha="center", va="bottom",
                    fontsize=9, color="crimson", transform=ax.get_xaxis_transform())

    ax.axhline(0.23, color="0.6", ls=":", lw=1)
    ax.axhline(0.50, color="0.4", ls="--", lw=1)
    ax.text(len(cols) - 0.5, 0.51, "good (0.5)", fontsize=8, color="0.3", ha="right")
    ax.text(len(cols) - 0.5, 0.24, "acceptable (0.23)", fontsize=8, color="0.3", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_xlabel("diffusion noise_scale (sampler churn $\\lambda$)")
    ax.set_ylabel(f"A|C DockQ per seed ({args.reduce} over samples)")
    ax.set_ylim(0, 1.08)
    ax.set_title("7txw Idea-1: diffuse from committed loop16 z under elevated noise\n"
                 "bar = mean over seeds; dots = seeds (green good / orange mid / grey bad); "
                 "red ◆ = mean over bad seeds")
    # legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLOR[k],
                          label=f"{k} (n={sum(v==k for v in cls.values())})")
               for k in ("good", "mid", "bad")]
    handles.append(plt.Line2D([], [], marker="D", ls="", color="crimson",
                              mec="k", label="mean(bad)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.text(0.5, -0.13, f"red numbers = #bad seeds rescued (DockQ≥{RESCUE_T}, CAPRI acceptable) / {n_bad}",
            ha="center", va="top", fontsize=8, color="crimson", transform=ax.transAxes)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)

    # --- text summary
    print(f"\nreduce={args.reduce}; classes: " +
          ", ".join(f"{k}={sum(v==k for v in cls.values())}" for k in ("good", "mid", "bad")))
    for label, seedvals in cols:
        bad_vals = [v for s, v in seedvals.items() if cls.get(s) == "bad" and v == v]
        lbl = label.replace("\n", " ")
        if bad_vals:
            print(f"  noise {lbl:>8}: mean(all)={np.nanmean(list(seedvals.values())):.3f}  "
                  f"mean(bad)={np.mean(bad_vals):.3f}  "
                  f"max(bad)={max(bad_vals):.3f}  rescued(>={RESCUE_T})={sum(v>=RESCUE_T for v in bad_vals)}/{len(bad_vals)}")


if __name__ == "__main__":
    main()
