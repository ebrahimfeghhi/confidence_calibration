#!/usr/bin/env python
"""Does ipTM track how GOOD a pose is, or how COMMITTED the seed that made it is?

THE QUESTION. On the 85 PXM selection failures the most-confident seed was the LESS spread-out one
99% of the time -- its five diffusion samples agreed with each other, whatever they agreed on. That
is a paired comparison on failures only. This is the population version: regress ipTM on quality and
on commitment together, over every ab-ag interface, and read the two standardized slopes.

    b1 > 0, b2 < 0  is the prediction:  better -> more confident, more spread -> less confident.
    |b2| ~ |b1|     would mean commitment is worth as much to ipTM as correctness.

TWO LEVELS, deliberately (the user asked for both, and they answer different questions):

  POSE level  -- one row per pose.  ipTM_pose ~ dockq_POSE + sd_seed
      ipTM is near-constant within a seed while DockQ is not, so the pose-level b1 is carried almost
      entirely by variation that ipTM cannot see. If pose quality is what ipTM reads, b1 survives
      here; if b1 collapses at the pose level while b2 does not, ipTM is reading the seed, not the
      pose. That collapse IS the result, not an artefact of the redundancy.

  SEED level  -- one row per seed.  ipTM_seed ~ mean_dockq_seed + sd_dockq_seed
      Matched to how ipTM actually varies. No redundancy, but n drops 5x.

THREE SCOPES per level, because pooling across interfaces confounds difficulty (an easy interface has
both high DockQ and high ipTM, which inflates b1 without any within-interface relationship):
  pooled            -- raw z-scores over everything; the confounded number, reported for contrast
  within-interface  -- each variable centered on its own interface mean first; the honest pooled fit
  per-interface     -- a separate fit per interface; reports the median slope and the sign split

SD IS COMPUTED ON THE SEED'S 5 SAMPLES, ddof=1, and seeds without exactly 5 scored samples are
dropped (counted in the output) so the spread statistic means the same thing on every row.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/iptm_dockq_spread_regression.py
"""
import collections
import csv

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
DQ = f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv"
ROLES = f"{EXP}/artifacts/pxm2025/chain_roles.csv"
SEL = f"{EXP}/artifacts/pxm2025/pxm2025_selection_protocol.csv"
NSAMP = 5


def zs(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def fit(y, X):
    """Standardized OLS with intercept. Returns (betas, r2, n)."""
    Z = np.column_stack([zs(c) for c in X.T])
    A = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(A, zs(y), rcond=None)
    resid = zs(y) - A @ b
    ss = np.sum(zs(y) ** 2)
    return b[1:], (1 - np.sum(resid ** 2) / ss if ss > 0 else np.nan), len(y)


def main():
    # ---- ab-ag interfaces only, keyed by pdb + unordered chain pair
    abag = set()
    for r in csv.DictReader(open(ROLES)):
        if r["status"] == "ok":
            abag.add(r["pdb"] + "_" + "".join(sorted((r["chain_1"], r["chain_2"]))))

    # ---- gather poses
    poses = collections.defaultdict(list)   # (iface, seed) -> [(sample, dockq, iptm)]
    skipped_noniface = 0
    for r in csv.DictReader(open(DQ)):
        key = r["pdb"] + "_" + "".join(sorted((r["chain1"], r["chain2"])))
        if key not in abag:
            skipped_noniface += 1
            continue
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        poses[(key, int(r["seed"]))].append(
            (int(r["sample"]), float(r["dockq"]), float(r["pair_iptm"])))

    # ---- keep only seeds with exactly NSAMP scored samples
    rows_pose, rows_seed = [], []
    dropped = 0
    for (iface, seed), v in poses.items():
        if len(v) != NSAMP:
            dropped += 1
            continue
        d = np.array([x[1] for x in v])
        p = np.array([x[2] for x in v])
        sd = float(d.std(ddof=1))
        for _, dq, ip in v:
            rows_pose.append((iface, seed, dq, sd, ip))
        rows_seed.append((iface, seed, float(d.mean()), sd, float(p.mean())))

    print(f"ab-ag interfaces with roles resolved: {len(abag)}")
    print(f"seeds kept (exactly {NSAMP} scored samples): {len(rows_seed)}"
          f"   dropped for wrong sample count: {dropped}")
    print(f"poses kept: {len(rows_pose)}"
          f"   rows skipped as non-ab-ag interfaces: {skipped_noniface}")
    print(f"distinct interfaces in the fit: {len({r[0] for r in rows_seed})}\n")

    PANELS = []
    for label, rows in (("POSE LEVEL  (y = pose ipTM,  x1 = POSE DockQ,  x2 = seed SD DockQ)",
                         rows_pose),
                        ("SEED LEVEL   (y = seed ipTM,  x1 = seed MEAN DockQ, x2 = seed SD DockQ)",
                         rows_seed)):
        iface = np.array([r[0] for r in rows], dtype=object)
        x1 = np.array([r[2] for r in rows])
        x2 = np.array([r[3] for r in rows])
        y = np.array([r[4] for r in rows])

        print("=" * 78)
        print(label)
        print("=" * 78)

        b, r2, n = fit(y, np.column_stack([x1, x2]))
        print(f"  pooled            b1(quality) {b[0]:+.3f}   b2(spread) {b[1]:+.3f}"
              f"   R2 {r2:.3f}   n={n}")

        # within-interface centering: remove each interface's own mean from every variable
        cx1, cx2, cy = x1.copy(), x2.copy(), y.copy()
        for f in set(iface):
            m = iface == f
            cx1[m] -= cx1[m].mean(); cx2[m] -= cx2[m].mean(); cy[m] -= cy[m].mean()
        b, r2, n = fit(cy, np.column_stack([cx1, cx2]))
        print(f"  within-interface  b1(quality) {b[0]:+.3f}   b2(spread) {b[1]:+.3f}"
              f"   R2 {r2:.3f}   n={n}")

        # per-interface fits
        B1, B2, R2, IF = [], [], [], []
        for f in set(iface):
            m = iface == f
            if m.sum() < 8 or x1[m].std() == 0 or x2[m].std() == 0 or y[m].std() == 0:
                continue
            bb, rr, _ = fit(y[m], np.column_stack([x1[m], x2[m]]))
            B1.append(bb[0]); B2.append(bb[1]); R2.append(rr); IF.append(f)
        B1, B2, R2 = np.array(B1), np.array(B2), np.array(R2)
        PANELS.append((label.split()[0], B1, B2, R2, np.array(IF, dtype=object)))
        if len(B1):
            print(f"  per-interface     b1 median {np.median(B1):+.3f} "
                  f"({100*np.mean(B1 > 0):.0f}% positive)"
                  f"   b2 median {np.median(B2):+.3f} "
                  f"({100*np.mean(B2 < 0):.0f}% negative)"
                  f"   n={len(B1)} interfaces")
        print()

    plot(PANELS)


def decoy_interfaces():
    """The CONFIDENT-DECOY set: ipTM selection picks an unacceptable pose when an acceptable one
    was available in the same 50-seed pool.

        sel50_dockq < 0.23  AND  oracle_dockq >= 0.23

    sel50 is the top-pair_iptm pose over all 250; oracle is the best DockQ in the same pool. So the
    good pose was GENERATED and confidence passed it over -- a selection failure, not a coverage
    failure. Interfaces whose oracle is itself unacceptable are excluded: nothing was there to pick.
    """
    keys = set()
    for r in csv.DictReader(open(SEL)):
        try:
            s, o = float(r["sel50_dockq"]), float(r["oracle_dockq"])
        except (ValueError, KeyError):
            continue
        if s < 0.23 <= o:
            pdb, ch = r["interface"].split("_")
            keys.add(pdb + "_" + "".join(sorted(ch)))
    return keys


def plot(panels):
    """Per-interface b1 vs b2, coloured by that interface's R2.

    One dot per interface, from its OWN joint fit -- so the dot's position says how that interface
    trades quality against commitment, and its colour says how much of its ipTM either explains.
    The quadrant that matters is lower-right (b1>0, b2<0): confidence rises with quality AND falls
    with spread, which is the predicted behaviour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dec = decoy_interfaces()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.9 * len(panels), 6.0))
    axes = np.atleast_1d(axes)
    for ax, (name, B1, B2, R2, IF) in zip(axes, panels):
        ax.axhline(0, color="0.75", lw=1, zorder=0)
        ax.axvline(0, color="0.75", lw=1, zorder=0)
        isd = np.array([f in dec for f in IF])
        s = ax.scatter(B1[~isd], B2[~isd], c=R2[~isd], cmap="viridis", vmin=0, vmax=1,
                       marker="o", s=34, edgecolor="white", linewidth=0.4, zorder=3,
                       label=f"ipTM selects an acceptable pose  (n={int((~isd).sum())})")
        ax.scatter(B1[isd], B2[isd], c=R2[isd], cmap="viridis", vmin=0, vmax=1,
                   marker="^", s=88, edgecolor="black", linewidth=0.7, zorder=5,
                   label=f"CONFIDENT DECOY  (n={int(isd.sum())})")
        for m, col, lab in ((~isd, "crimson", "median, rest"), (isd, "black", "median, decoys")):
            if m.sum():
                ax.scatter([np.median(B1[m])], [np.median(B2[m])], marker="X", s=200, color=col,
                           edgecolor="white", linewidth=1.2, zorder=6,
                           label=f"{lab} ({np.median(B1[m]):+.2f}, {np.median(B2[m]):+.2f})")
        q = int(np.sum((B1 > 0) & (B2 < 0)))
        ax.set_title(f"{name} level   n={len(B1)} interfaces"
                     f"   ({int(isd.sum())} confident decoys)\n"
                     f"{q}/{len(B1)} ({100*q/len(B1):.0f}%) in the predicted quadrant")
        ax.set_xlabel("b1  —  standardized slope on DockQ (quality)")
        ax.set_ylabel("b2  —  standardized slope on seed DockQ SD (spread)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    cb = fig.colorbar(s, ax=list(axes), fraction=0.035, pad=0.02)
    cb.set_label("R² of that interface's joint fit")
    out = f"{EXP}/artifacts/iptm_b1_b2_by_interface.png"
    fig.suptitle("Per-interface joint fit:  ipTM ~ DockQ + seed DockQ spread", y=0.99)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
