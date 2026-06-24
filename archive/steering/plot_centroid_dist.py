"""Plot W1 distance to each basin centroid (G/B1/B2) across runs, per trial,
for the two working reset repulsion runs (s_lm nudge, s_lm z0-var).

Reads results/csv_files/steer_repel_<tag>_M5_K3.csv — the wg/wb1/wb2 columns ARE the
W1 distance from each run's 17-loop distogram to the G/B1/B2 loop-16 centroids — and makes
a small-multiples grid: one subplot per trial, three lines (->G green, ->B1 gray, ->B2
orange) vs run, with the within-G band shaded. Real-G runs (basin==G & wg<0.08) are starred.
One PNG per config. Read-only / no GPU; safe to run while the jobs are still writing.

Run:  /home/ebrahim/esm/.venv/bin/python analysis/grad_steer/plot_centroid_dist.py
"""
import csv
import math
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

FIGDIR = REPO_ROOT / "analysis" / "figs"
BAND = (0.033, 0.079)                      # within-G band (good seeds to own centroid)
REAL_G = 0.08                              # wg below this AND basin==G => real G hit
# (tag, title). tag = full CSV stem after "steer_repel_" and before ".csv".
CONFIGS = [
    ("slm_nudge_carry_w1_rn_leash15_M5_K3", "s_lm nudge (carry+renudge, leash0.15)"),
    ("slm_nudge_carry_w1_kb_leash15_M5_K3", "s_lm nudge (carry+keepbase, leash0.15)"),
]


def load(tag):
    """tag -> {trial: [(run, wg, wb1, wb2, basin), ...] sorted by run}."""
    path = CSV_DIR / f"steer_repel_{tag}.csv"
    by_trial = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_trial[int(r["trial"])].append(
                (int(r["run"]), float(r["wg"]), float(r["wb1"]), float(r["wb2"]), r["basin"]))
    for t in by_trial:
        by_trial[t].sort()
    return dict(sorted(by_trial.items()))


def plot_config(tag, title):
    by_trial = load(tag)
    trials = list(by_trial)
    ncol = 5
    nrow = max(1, math.ceil(len(trials) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axflat = axes.flatten()
    n_realG = 0
    for ax, t in zip(axflat, trials):
        rows = by_trial[t]
        runs = [r[0] for r in rows]
        ax.axhspan(*BAND, color="green", alpha=0.12)
        ax.axhline(1.0, color="lightgray", ls=":", lw=0.8)         # B1 baseline ~1.0
        ax.plot(runs, [r[2] for r in rows], "-o", color="gray", ms=3, lw=1.2, label="->B1")
        ax.plot(runs, [r[3] for r in rows], "-o", color="tab:orange", ms=3, lw=1.2, label="->B2")
        ax.plot(runs, [r[1] for r in rows], "-o", color="tab:green", ms=3, lw=1.7, label="->G")
        gx = [r[0] for r in rows if r[4] == "G" and r[1] < REAL_G]
        gy = [r[1] for r in rows if r[4] == "G" and r[1] < REAL_G]
        if gx:
            ax.plot(gx, gy, "*", color="darkgreen", ms=13, zorder=5)
        n_realG += len(gx)
        ax.set_title(f"trial {t}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axflat[len(trials):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("run")
    for ax in axes[:, 0]:
        ax.set_ylabel("W1 to centroid")
    axflat[0].legend(fontsize=8, loc="center right")
    fig.suptitle(f"{title} - W1 distance to G/B1/B2 centroids across runs "
                 f"(* = real G, wg<{REAL_G}; {n_realG} total over {len(trials)} trials)",
                 fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"centroid_dist_{tag.replace('_M5_K3', '')}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(trials)} trials, {n_realG} real-G runs)")


def main():
    # CLI: pass one or more full CSV-stem tags to override CONFIGS (title = tag).
    configs = [(t, t) for t in sys.argv[1:]] if len(sys.argv) > 1 else CONFIGS
    for tag, title in configs:
        path = CSV_DIR / f"steer_repel_{tag}.csv"
        if not path.exists():
            print(f"skip {tag}: no {path.name}")
            continue
        plot_config(tag, title)


if __name__ == "__main__":
    main()
