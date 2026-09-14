#!/usr/bin/env python
"""How much does CDR H3 LOCAL RMSD vary across seeds within a complex?

Local RMSD = the H3 Calpha superposed directly onto the reference H3 Calpha, so it measures loop
CONFORMATION with placement divided out.

Three outputs:
  1. Variance decomposition. Total H3 variance split into between-complex vs within-complex, and
     within a complex, seed-to-seed vs sample-to-sample. The second split matters because a seed's 5
     diffusion samples share ONE trunk state -- if H3 is decided in the trunk they should agree, and
     the seed-level term should dominate.
  2. Box-whisker of local H3 RMSD per complex, sorted by median -- the direct picture of the spread.
  3. Small-multiple scatter of H3-local vs DockQ for 20 complexes spanning the range, to show whether
     the within-complex variation actually tracks docking quality.

The 20 are chosen to SPAN median DockQ (evenly spaced quantiles), not hand-picked, so the panel is
not a curated success story.

Reads artifacts/cdr_h3_rmsd_all.csv joined to the free-run DockQ; seeds 101-150 only, which is the
range where free-run DockQ exists, giving every complex the same 50-seed depth.

Run with the Protenix venv python (it has matplotlib):
  /home/ebrahim/Protenix/.venv/bin/python scripts/h3_variation.py
"""
import csv
import collections
import glob
import json
import os
import re
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
OUT = f"{EXP}/artifacts"


def load():
    ifm = collections.defaultdict(list)
    for r in csv.DictReader(open(IFACE_CSV)):
        ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))
    h3 = {}
    for r in csv.DictReader(open(f"{OUT}/cdr_h3_rmsd_all.csv")):
        s = int(r["seed"])
        if 101 <= s <= 150:
            h3[(r["pdb"], s, int(r["sample"]))] = (float(r["h3_local"]), r["chain"])
    dq = collections.defaultdict(dict)
    for f in glob.glob(f"{OUT}/abag_all_ost/*.json"):
        m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
        if not m:
            continue
        pdb, s, k = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            j = json.load(open(f))
        except Exception:
            continue
        qs, ifs = j.get("dockq") or [], j.get("dockq_interfaces") or []
        for c1, c2 in ifm.get(pdb, []):
            for i, t in enumerate(ifs):
                if i < len(qs) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                    dq[(pdb, s, k)][c1] = max(dq[(pdb, s, k)].get(c1, -1), qs[i])
                    break
    rows = []
    for key, (loc, ch) in h3.items():
        d = dq.get(key)
        if not d:
            continue
        rows.append((key[0], key[1], key[2], loc, d.get(ch, max(d.values()))))
    return rows


def decompose(rows):
    """between-complex vs within-complex, then seed-level vs sample-level within a complex."""
    by_c = collections.defaultdict(list)
    for p, s, k, loc, q in rows:
        by_c[p].append((s, loc))
    allv = [loc for _, _, _, loc, _ in rows]
    grand = st.mean(allv)
    # between = variance of complex means; within = mean of within-complex variances
    cmeans = {p: st.mean(v for _, v in vs) for p, vs in by_c.items()}
    between = st.pvariance(list(cmeans.values()))
    within = st.mean([st.pvariance([v for _, v in vs]) for vs in by_c.values() if len(vs) > 1])
    tot = between + within
    print(f"  total spread (SD)            : {st.pstdev(allv):.3f} A")
    print(f"  BETWEEN-complex variance     : {between:.4f}  ({between/tot:.0%})")
    print(f"  WITHIN-complex variance      : {within:.4f}  ({within/tot:.0%})")

    # within a complex: seed-level vs sample-level
    seed_var, samp_var = [], []
    for p, vs in by_c.items():
        by_s = collections.defaultdict(list)
        for s, v in vs:
            by_s[s].append(v)
        smeans = [st.mean(v) for v in by_s.values() if v]
        wsamp = [st.pvariance(v) for v in by_s.values() if len(v) > 1]
        if len(smeans) > 1 and wsamp:
            seed_var.append(st.pvariance(smeans))
            samp_var.append(st.mean(wsamp))
    sv, kv = st.mean(seed_var), st.mean(samp_var)
    print(f"\n  within a complex, split by source (n={len(seed_var)} complexes):")
    print(f"    SEED-to-seed variance      : {sv:.4f}  ({sv/(sv+kv):.0%})")
    print(f"    SAMPLE-to-sample (same seed): {kv:.4f}  ({kv/(sv+kv):.0%})")
    print(f"    -> seed-level dominates by {sv/kv:.1f}x" if kv > 0 else "")
    return by_c


def per_complex_table(rows):
    by_c = collections.defaultdict(list)
    dqc = collections.defaultdict(list)
    for p, s, k, loc, q in rows:
        by_c[p].append(loc)
        dqc[p].append(q)
    out = []
    for p, v in by_c.items():
        v = sorted(v)
        q1, q3 = np.percentile(v, [25, 75])
        out.append((p, len(v), st.median(v), q1, q3, q3 - q1, min(v), max(v),
                    st.median(dqc[p])))
    return sorted(out, key=lambda r: r[2])


def main():
    rows = load()
    print(f"n = {len(rows)} poses, {len(set(r[0] for r in rows))} complexes, seeds 101-150\n")
    print("VARIANCE DECOMPOSITION of CDR H3 local RMSD")
    decompose(rows)

    tbl = per_complex_table(rows)
    print(f"\nPER-COMPLEX SPREAD (IQR of local H3 RMSD across the 250 poses):")
    iqrs = [r[5] for r in tbl]
    print(f"  median IQR {st.median(iqrs):.2f} A | mean {st.mean(iqrs):.2f} | "
          f"min {min(iqrs):.2f} ({tbl[int(np.argmin(iqrs))][0]}) | max {max(iqrs):.2f} "
          f"({tbl[int(np.argmax(iqrs))][0]})")
    print("\n  tightest 5:", ", ".join(f"{r[0]} {r[5]:.2f}" for r in sorted(tbl, key=lambda r: r[5])[:5]))
    print("  widest 5  :", ", ".join(f"{r[0]} {r[5]:.2f}" for r in sorted(tbl, key=lambda r: -r[5])[:5]))

    # ---------- plot 1: box-whisker, every complex ----------
    data = collections.defaultdict(list)
    for p, s, k, loc, q in rows:
        data[p].append(loc)
    order = [r[0] for r in tbl]
    med_dq = {r[0]: r[8] for r in tbl}
    fig, ax = plt.subplots(figsize=(max(14, len(order) * 0.16), 6.5))
    bp = ax.boxplot([data[p] for p in order], showfliers=False, patch_artist=True,
                    widths=0.65, medianprops=dict(color="black", lw=1.1))
    for patch, p in zip(bp["boxes"], order):
        q = med_dq[p]
        patch.set_facecolor("#2a7f62" if q >= 0.8 else "#e0a33e" if q >= 0.23 else "#b3543f")
        patch.set_alpha(0.85); patch.set_edgecolor("#333"); patch.set_linewidth(0.5)
    ax.axhline(1.0, color="#444", ls="--", lw=1, zorder=0)
    ax.text(0.6, 1.06, "1 Å threshold", fontsize=8, color="#444")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=5.5)
    ax.set_ylabel("CDR H3 local RMSD (Å)")
    ax.set_ylim(0, min(8, max(max(v) for v in data.values())))
    ax.set_title("CDR H3 local RMSD across 50 seeds x 5 samples, per complex "
                 "(sorted by median; box = IQR, whiskers = 1.5 IQR)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2a7f62", label="median DockQ ≥ 0.8"),
                       Patch(facecolor="#e0a33e", label="0.23 ≤ median DockQ < 0.8"),
                       Patch(facecolor="#b3543f", label="median DockQ < 0.23")],
              fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/h3_box_all.png", dpi=160); plt.close(fig)

    # ---------- plot 2: 20 complexes spanning median DockQ ----------
    by_dq = sorted(tbl, key=lambda r: r[8])
    idx = np.linspace(0, len(by_dq) - 1, 20).astype(int)
    sel = [by_dq[i][0] for i in idx]
    pts = collections.defaultdict(list)
    for p, s, k, loc, q in rows:
        pts[p].append((loc, q))
    fig, axes = plt.subplots(4, 5, figsize=(17, 12), sharex=False, sharey=True)
    for ax, p in zip(axes.ravel(), sel):
        xs = [a for a, b in pts[p]]; ys = [b for a, b in pts[p]]
        ax.scatter(xs, ys, s=7, alpha=0.45, c="#2b6cb0", edgecolors="none")
        ax.axhline(0.23, color="#b3543f", ls=":", lw=0.9)
        ax.axhline(0.8, color="#2a7f62", ls=":", lw=0.9)
        ax.axvline(1.0, color="#444", ls="--", lw=0.8)
        r = np.corrcoef(xs, ys)[0, 1] if len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")
        ax.set_title(f"{p}  (r={r:+.2f}, n={len(xs)})", fontsize=9)
        ax.set_xlim(0, min(6, max(xs) * 1.05 + 0.1)); ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("H3 local RMSD (Å)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("DockQ", fontsize=8)
    fig.suptitle("DockQ vs CDR H3 local RMSD, 20 complexes spanning median DockQ "
                 "(each point = one pose; dashed = 1 Å, dotted = DockQ 0.23 / 0.8)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/h3_scatter_20.png", dpi=150); plt.close(fig)

    # within-complex correlation summary
    rs = []
    for p, v in pts.items():
        xs = [a for a, b in v]; ys = [b for a, b in v]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            rs.append(np.corrcoef(xs, ys)[0, 1])
    print(f"\nWITHIN-COMPLEX correlation r(H3 local, DockQ), n={len(rs)} complexes:")
    print(f"  median r {np.median(rs):+.2f} | mean {np.mean(rs):+.2f} | "
          f"negative in {sum(1 for r in rs if r < 0)}/{len(rs)}")
    print(f"  |r|>0.5 in {sum(1 for r in rs if abs(r) > 0.5)}/{len(rs)} complexes")
    print(f"\nwrote {OUT}/h3_box_all.png and {OUT}/h3_scatter_20.png")


if __name__ == "__main__":
    main()
