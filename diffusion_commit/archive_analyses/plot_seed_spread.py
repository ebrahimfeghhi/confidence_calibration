#!/usr/bin/env python
"""Per-seed view of an interface: DockQ spread vs mean DockQ, coloured by mean pair ipTM.

WHAT THE AXES MEAN. One point per seed, over that seed's diffusion samples.
  x  spread  = max DockQ - min DockQ within the seed. How much the five poses disagree.
  y  mean    = that seed's average DockQ. How good the seed is on average.
  c  ipTM    = the seed's mean pair ipTM -- what selection actually sees.

WHY THIS PLOT. The arms of the fold+commit experiment compare poses WITHIN a seed, so only seeds whose
poses straddle DockQ 0.23 can test selection at all; a seed with near-zero spread has a flat donor axis
and is uninformative whatever its quality. The dashed wedge marks where straddling is possible: a seed
can only cross 0.23 if its spread is at least |mean - 0.23| (roughly -- exactly, if min < 0.23 <= max).
Colour then answers the selection question directly: if ipTM saw interface quality, the top of the plot
would be bright and the bottom dark.

Usage: plot_seed_spread.py --iface 9x1h_EF [--source pxm|results] [--out <png>]
"""
import argparse
import collections
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
AP = 0.23


def load_pxm(iface):
    """The PXM-2025 pool: 50 seeds x 5 samples per interface."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv")):
        if f"{r['pdb']}_{r['chain1']}{r['chain2']}" != iface:
            continue
        try:
            by[r["seed"]].append((float(r["dockq"]), float(r["pair_iptm"])))
        except (TypeError, ValueError):
            continue
    return by


def load_results(iface):
    """Our own fold+commit runs -- donor poses only, which are the free-run equivalent."""
    by, seen = collections.defaultdict(list), set()
    for r in csv.DictReader(open(f"{EXP}/artifacts/commit11/results.csv")):
        if r["iface"] != iface or r["arm"] != "donor" or not r["iptm"]:
            continue
        k = (r["seed"], r["sample"], r["dockq"])
        if k in seen:                 # exact duplicate rows predate score()'s dedup guard
            continue
        seen.add(k)
        by[r["seed"]].append((float(r["dockq"]), float(r["iptm"])))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--source", choices=("pxm", "results"), default="pxm")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    by = (load_pxm if A.source == "pxm" else load_results)(A.iface)
    by = {s: v for s, v in by.items() if len(v) >= 2}
    if not by:
        raise SystemExit(f"no data for {A.iface} in source={A.source}")

    x, y, c, strad = [], [], [], []
    for s, v in by.items():
        dq = [p[0] for p in v]
        x.append(max(dq) - min(dq))
        y.append(st.mean(dq))
        c.append(st.mean(p[1] for p in v))
        strad.append(min(dq) < AP <= max(dq))

    fig, ax = plt.subplots(figsize=(8.2, 6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=90, edgecolor="black", linewidth=0.6, zorder=3)
    # ring the straddling seeds -- the only ones the arms can learn anything from
    sx = [a for a, f in zip(x, strad) if f]
    sy = [b for b, f in zip(y, strad) if f]
    ax.scatter(sx, sy, s=260, facecolors="none", edgecolors="crimson", linewidth=1.8, zorder=2,
               label=f"straddles {AP} ({len(sx)}/{len(x)} seeds)")
    ax.axhline(AP, color="gray", ls="--", lw=1, zorder=1)
    ax.text(ax.get_xlim()[1], AP, f" DockQ {AP}", va="bottom", ha="right", color="gray", fontsize=9)

    ax.set_xlabel("within-seed DockQ spread   (max - min across the 5 diffusion samples)")
    ax.set_ylabel("within-seed mean DockQ")
    ax.set_title(f"{A.iface} — per-seed quality vs spread, coloured by mean pair ipTM\n"
                 f"{len(x)} seeds, source={A.source}", fontsize=11)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("mean pair ipTM  (what selection sees)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    out = A.out or f"{EXP}/artifacts/seed_spread_{A.iface}_{A.source}.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")

    # the numbers behind the picture, so the plot is never the only record
    print(f"  seeds {len(x)}   straddling {sum(strad)}   "
          f"mean DockQ {st.mean(y):.3f}   mean spread {st.mean(x):.3f}")
    if len(x) > 2:
        mx, my = st.mean(c), st.mean(y)
        r = (sum((a-mx)*(b-my) for a, b in zip(c, y)) / len(x)) / (st.pstdev(c)*st.pstdev(y))
        print(f"  corr(mean ipTM, mean DockQ) across seeds r = {r:+.3f}")


if __name__ == "__main__":
    main()
