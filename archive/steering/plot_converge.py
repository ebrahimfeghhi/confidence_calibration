"""Plot per-step convergence curves from steer_converge.py runs.

Reads every results/csv_files/steer_converge_<lever>_K<k>_lr<lr>.csv and makes a
two-panel figure:
  left  : short-horizon loss  w1(P_K, P_G)   vs optimization step  (the optimized loss)
  right : full-horizon transfer  w1(P_full, P_G)  vs step           (does it reach G?)
with the within-G band (0.033-0.079) shaded on the right panel.

Run:  uv run python analysis/grad_steer/plot_converge.py
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, REPO_ROOT  # noqa: E402

FIG = REPO_ROOT / "analysis" / "figs" / "steer_converge.png"
BAND = (0.033, 0.079)


def parse_name(p):
    # steer_converge_<lever>_K<k>_lr<lr>.csv
    stem = Path(p).stem.replace("steer_converge_", "")
    lever, krest = stem.split("_K", 1)
    k, lr = krest.split("_lr", 1)
    return lever, int(k), lr


def load(p):
    steps, short, fstep, full = [], [], [], []
    for r in csv.DictReader(open(p)):
        steps.append(int(r["step"]))
        short.append(float(r["short_loss"]))
        if r["full_w1"] not in ("", None):
            fstep.append(int(r["step"])); full.append(float(r["full_w1"]))
    return steps, short, fstep, full


def main():
    files = sorted(glob.glob(str(CSV_DIR / "steer_converge_*.csv")))
    if not files:
        raise SystemExit("no steer_converge_*.csv found")
    # color by lever, dash by (K, lr)
    color = {"slm": "tab:blue", "z0": "tab:red"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for p in files:
        lever, k, lr = parse_name(p)
        steps, short, fstep, full = load(p)
        label = f"{lever} K={k} lr={lr}"
        ls = "-" if lever == "slm" else "--"
        lw = 2.2 if (lever == "z0" and lr == "0.1") else 1.6
        c = color.get(lever, "gray")
        axL.plot(steps, short, ls, color=c, lw=lw, label=label, alpha=0.85)
        axR.plot(fstep, full, ls, color=c, lw=lw, marker="o", ms=4, label=label, alpha=0.85)

    axL.set_title("Short-horizon loss  W1(P_K, P_G)  (the optimized objective)")
    axL.set_xlabel("optimization step"); axL.set_ylabel("W1 to G centroid")
    axL.grid(alpha=0.3); axL.legend(fontsize=8)

    axR.axhspan(*BAND, color="green", alpha=0.12, label=f"within-G band {BAND}")
    axR.axhline(1.0, color="gray", ls=":", lw=1, label="B1 baseline ~1.0")
    axR.set_title("Full-horizon transfer  W1(P_full(17 loops), P_G)")
    axR.set_xlabel("optimization step"); axR.set_ylabel("W1 to G centroid")
    axR.grid(alpha=0.3); axR.legend(fontsize=8)

    fig.suptitle("Supervised attract-to-G steering: convergence of s_lm vs z0 across horizon K",
                 fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}  ({len(files)} configs: {[parse_name(f)[:2] for f in files]})")


if __name__ == "__main__":
    main()
