"""DockQ and chain-pair ipTM across recycling loops, per seed.

For each (seed, loop) cell, pick the single best diffusion sample by A|C chain-pair
ipTM (confidence selection, per the sample-selection-by-confidence finding), then
plot that sample's DockQ and pair-ipTM vs loop. 4 good seeds vs 4 bad seeds.

Reads CSV_DIR/dockq_7txw_loopLL.csv (from eval/score_7txw_loops.py).
Output: analysis/figs/dockq_iptm_across_loops_7txw.png
"""
import csv
import sys
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

OUT = Path(__file__).resolve().parent / "figs" / "dockq_iptm_across_loops_7txw.png"
GOOD = [10, 14, 33, 39]
BAD = [12, 0, 32, 42]          # 2 deep failures + 2 best-of-bad
SEEDS = GOOD + BAD
N_LOOPS = 17


def load_loop(L):
    """Per seed: the (dockq, pair_iptm) of the max-pair_iptm A|C sample. {seed: (dq, iptm)}."""
    f = CSV_DIR / f"dockq_7txw_loop{L:02d}.csv"
    if not f.exists():
        return {}
    best = {}                                  # seed -> (pair_iptm, dockq)
    for r in csv.DictReader(open(f)):
        if r["chain1"] != "A" or r["chain2"] != "C":
            continue
        try:
            s = int(r["seed"]); iptm = float(r["pair_iptm"]); dq = float(r["dockq"])
        except (ValueError, KeyError):
            continue
        if np.isnan(iptm):
            continue
        if s not in best or iptm > best[s][0]:
            best[s] = (iptm, dq)
    return {s: (dq, iptm) for s, (iptm, dq) in best.items()}


def main():
    dq = np.full((len(SEEDS), N_LOOPS), np.nan)
    ip = np.full((len(SEEDS), N_LOOPS), np.nan)
    for L in range(N_LOOPS):
        sel = load_loop(L)
        for i, s in enumerate(SEEDS):
            if s in sel:
                dq[i, L], ip[i, L] = sel[s]

    loops = np.arange(N_LOOPS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    reds = plt.cm.Reds(np.linspace(0.55, 0.95, len(GOOD)))
    blues = plt.cm.Blues(np.linspace(0.5, 0.9, len(BAD)))

    for ax, M, title, ylab in [
        (axes[0], dq, "A|C DockQ across recycling loops", "DockQ (best pair-ipTM sample)"),
        (axes[1], ip, "A|C chain-pair ipTM across recycling loops", "pair ipTM (selected sample)"),
    ]:
        for i, s in enumerate(SEEDS):
            good = s in GOOD
            col = reds[GOOD.index(s)] if good else blues[BAD.index(s)]
            ax.plot(loops, M[i], "-o", color=col, ms=4, lw=1.8,
                    label=f"seed {s} ({'good' if good else 'bad'})")
        ax.set_xlabel("pairformer recycling loop")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_xticks(loops); ax.grid(alpha=0.25)
    axes[0].axhline(0.23, color="0.6", ls=":", lw=1); axes[0].axhline(0.5, color="0.4", ls="--", lw=1)
    axes[0].text(0, 0.51, "acceptable (0.23) / good (0.5)", fontsize=8, color="0.3")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="center right")

    fig.suptitle("7txw — diffuse-from-each-loop: does structure quality track the "
                 "trunk's good/bad commitment? (best sample by pair-ipTM)", y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    # quick text summary
    print("\nloop : " + " ".join(f"{L:5d}" for L in loops))
    for i, s in enumerate(SEEDS):
        tag = "G" if s in GOOD else "B"
        print(f"s{s:<3}{tag} DockQ " + " ".join(
            (f"{dq[i,L]:.2f}" if not np.isnan(dq[i, L]) else "  - ") for L in loops))


if __name__ == "__main__":
    main()
