#!/usr/bin/env python
"""Per interface and arm: what does ipTM select, before and after self-templating?

THE QUESTION THIS ANSWERS. Selection is the whole problem -- a good pose exists and ipTM ships a bad
one. So the number that matters is not the best DockQ an arm produced, it is the DockQ of the pose
ipTM would actually CHOOSE. This reports that at three points:

  pool      the free pool's most confident pose, and its DockQ. This is the confident-decoy
            definition and the baseline any method has to beat.
  donors    the most confident of the 5 poses this arm templated ON, and its DockQ.
  after     the most confident pose the arm PRODUCED, and its DockQ.

pool -> after is the end-to-end claim. donors -> after says whether templating raised confidence in
the basin it was given.

POOLED ACROSS SEEDS, not per seed. An arm's 5 seeds x 5 samples are 25 candidate poses and a user
would pick the most confident of all of them; per-seed selection cannot see a between-seed reordering,
which is exactly where confident decoys live.

CROSS-ARM ipTM IS NOT COMPARABLE and this script does not compare it. Self-templating raises ipTM
whether or not the donors were right -- on 9h01_AB the arm templated on five DockQ-0.005 poses
reported ipTM 0.933. Read ipTM WITHIN an arm (did it go up) and DockQ ACROSS arms (did it help).

    arms_report.py [--out arms.csv] [--min-seeds 1]
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
from fmt import table  # noqa: E402
from make_queue import ipmatched  # noqa: E402
from separation import pool_classes  # noqa: E402

RESULTS = f"{paths.DATA_DIR}/top5/results.csv"
ACC = 0.23


def donor_indices(dq, ip, k=5):
    """{arm: [pose indices]} -- recomputed exactly as make_queue.py chose them."""
    o = list(np.argsort(-dq)[:k])
    c = list(np.argsort(-ip)[:k])
    out = {"oracle": o, "conf": c}
    m = ipmatched(dq, ip, o, c, k)
    if m is not None:
        out["ipmatch"] = m
    return out


def plot_per_interface(done, path, klass):
    """One row per interface: what ipTM ships, before templating and after, per arm.

    A dumbbell rather than bars: the quantity of interest is the MOVE from the pool's confident pick
    to the arm's confident pick, and a bar chart hides which interfaces moved and which did not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = collections.defaultdict(dict)
    for r in done:
        if klass in ("all", r["klass"]):
            per[r["interface"]][r["arm"]] = r
    if not per:
        return
    COL = {"oracle": "#1b7f4b", "conf": "#d95f02", "ipmatch": "#6d3f9e"}
    order = sorted(per, key=lambda i: (per[i].get("oracle", {}).get("after_dockq_at_max_iptm") or 0))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(order) + 1.4)))
    for y, iface in enumerate(order):
        v = per[iface]
        base = next(iter(v.values()))["pool_dockq_at_max_iptm"]
        for arm in ("oracle", "conf", "ipmatch"):
            if arm not in v:
                continue
            aft = v[arm]["after_dockq_at_max_iptm"]
            ax.plot([base, aft], [y, y], lw=1.1, c=COL[arm], alpha=.5, zorder=1)
            ax.scatter([aft], [y], s=30, c=COL[arm], zorder=3,
                       label=arm if y == 0 else None)
        ax.scatter([base], [y], s=26, facecolor="none", edgecolor="#555", zorder=2,
                   label="free pool" if y == 0 else None)
    ax.axvline(ACC, ls="--", lw=1, c="k", alpha=.5)
    ax.text(ACC, len(order) + .2, " acceptable", fontsize=8, va="bottom")
    ax.set(yticks=range(len(order)), xlabel="DockQ of the most confident pose (what ipTM ships)",
           xlim=(-0.02, 1.0), title=f"What confidence selects, per {klass} interface (n={len(order)})")
    ax.set_yticklabels(order, fontsize=7)
    ax.grid(axis="x", alpha=.25)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arms.csv")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--per-interface", action="store_true",
                    help="one row per interface instead of the class summary")
    ap.add_argument("--plot", default=None, help="write a per-interface figure here")
    ap.add_argument("--klass", default="decoy")
    ap.add_argument("--pooled", action="store_true",
                    help="selection pooled across ALL templated poses -- the deployable number")
    A = ap.parse_args()

    cls = pool_classes()

    # produced poses: (interface, arm) -> [(iptm, dockq, seed), ...]
    made = collections.defaultdict(list)
    if os.path.exists(A.results):
        with open(A.results) as fh:
            for r in csv.DictReader(fh):
                made[(r["iface"], r["arm"])].append(
                    (float(r["iptm"]), float(r["dockq"]), r["seed"]))

    rows = []
    for f in sorted(glob.glob("matrices/*.npz")):
        iface = os.path.basename(f)[:-4]
        z = np.load(f)
        dq, ip = z["dockq_gt"].astype(float), z["pair_iptm"].astype(float)
        if len(dq) < 5:
            continue
        pool_i = int(ip.argmax())
        dons = donor_indices(dq, ip)
        for arm, idx in dons.items():
            got = made.get((iface, arm), [])
            nseed = len({s for _, _, s in got})
            if nseed < A.min_seeds:
                continue
            j = int(np.argmax(ip[idx]))          # most confident DONOR
            row = dict(
                interface=iface, klass=cls.get(iface, "?"), arm=arm,
                n_poses_made=len(got), n_seeds=nseed,
                pool_max_iptm=round(float(ip[pool_i]), 4),
                pool_dockq_at_max_iptm=round(float(dq[pool_i]), 4),
                pool_best_dockq=round(float(dq.max()), 4),
                donor_max_iptm=round(float(ip[idx][j]), 4),
                donor_dockq_at_max_iptm=round(float(dq[idx][j]), 4),
                donor_best_dockq=round(float(dq[idx].max()), 4))
            if got:
                k = int(np.argmax([a for a, _, _ in got]))
                row.update(
                    after_max_iptm=round(got[k][0], 4),
                    after_dockq_at_max_iptm=round(got[k][1], 4),
                    after_best_dockq=round(max(b for _, b, _ in got), 4))
            else:
                row.update(after_max_iptm="", after_dockq_at_max_iptm="", after_best_dockq="")
            rows.append(row)

    if not rows:
        raise SystemExit("no interfaces with results yet")
    with open(A.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r["after_max_iptm"] != ""]
    by = collections.defaultdict(list)
    for r in done:
        by[(r["klass"], r["arm"])].append(r)
    out = []
    for k in sorted(by, key=lambda k: ({"decoy": 0, "solved": 1, "nocov": 2}.get(k[0], 3), k[1])):
        v = by[k]
        m = lambda f: round(float(np.mean([x[f] for x in v])), 3)  # noqa: E731
        out.append([k[0], k[1], len(v),
                    m("pool_max_iptm"), m("pool_dockq_at_max_iptm"),
                    m("after_max_iptm"), m("after_dockq_at_max_iptm"),
                    sum(1 for x in v if x["pool_dockq_at_max_iptm"] >= ACC),
                    sum(1 for x in v if x["after_dockq_at_max_iptm"] >= ACC)])
    print(table(["class", "arm", "interfaces (n)",
                 "pool max ipTM", "DockQ it picks",
                 "after max ipTM", "DockQ it picks",
                 "pool picks OK", "after picks OK"], out, align="l l r r r r r r r"))
    # POOLED ACROSS EVERY TEMPLATED POSE. Within-arm selection reported 8 of 24 acceptable; pooling
    # the oracle and conf arms together gives 5 of 32, because on three interfaces the CONF arm's pose
    # is more confident than the oracle arm's and selection ships that instead. Pooled is the honest
    # number: in use you do not know which arm produced which pose.
    pooled = collections.defaultdict(list)
    for (iface, arm), v in made.items():
        pooled[iface].extend(v)
    if A.pooled:
        prows = []
        for r in rows:
            if r["arm"] != "oracle" or (A.klass not in ("all", r["klass"])):
                continue
            P = pooled.get(r["interface"])
            if not P:
                continue
            k = int(np.argmax([a for a, _, _ in P]))
            prows.append([r["interface"], r["klass"], r["pool_dockq_at_max_iptm"],
                          round(P[k][1], 3), round(P[k][0], 3), len(P)])
        prows.sort(key=lambda x: -x[3])
        print(table(["interface", "class", "free pool baseline",
                     "POOLED templated", "its ipTM", "poses pooled"], prows, align="l l r r r r"))
        a = np.array([x[3] for x in prows])
        b = np.array([x[2] for x in prows])
        print(f"\n  n={len(prows)}   mean {b.mean():.3f} -> {a.mean():.3f}   "
              f"acceptable {int((b >= ACC).sum())}/{len(b)} -> {int((a >= ACC).sum())}/{len(a)}")
    if A.per_interface:
        per = collections.defaultdict(dict)
        for r in done:
            per[r["interface"]][r["arm"]] = r
        sel = {i: v for i, v in per.items()
               if (A.klass == "all" or (v and next(iter(v.values()))["klass"] == A.klass))}
        rows2 = []
        for iface, v in sorted(sel.items(),
                               key=lambda kv: -(kv[1].get("oracle", {}).get("after_dockq_at_max_iptm", 0)
                                                or 0)):
            any_r = next(iter(v.values()))
            rows2.append([iface, any_r["klass"],
                          any_r["pool_dockq_at_max_iptm"], any_r["pool_best_dockq"]]
                         + [(v[a]["after_dockq_at_max_iptm"] if a in v else "-")
                            for a in ("oracle", "conf", "ipmatch")]
                         + [(v[a]["after_max_iptm"] if a in v else "-")
                            for a in ("oracle", "conf")])
        print(table(["interface", "class", "pool picks", "pool best",
                     "oracle picks", "conf picks", "ipmatch picks",
                     "oracle ipTM", "conf ipTM"], rows2, align="l l r r r r r r r"))
        print(f"\n  n = {len(rows2)} interfaces with results")
    if A.plot:
        plot_per_interface(done, A.plot, A.klass)
    print(f"\n  'DockQ it picks' = the DockQ of the most confident pose -- what a user would ship.")
    print(f"  wrote {A.out}  ({len(rows)} interface-arm rows, {len(done)} with results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
