#!/usr/bin/env python
"""Budget-matched at 75 seeds: pooled-ipTM pick from a FREE pool vs a SELF-TEMPLATED pool.

  FREE-75    50 free seeds (PXM pool, 300000-300049) + 25 more free seeds (arm `free75`, 300050-300074)
  HYBRID-75  25 free donors (300000-300024) + 25 gen-1 templated (hybrid25) + 25 gen-2 (arm `gen2`)

Both are 75 seeds x 5 poses, so the comparison prices self-templating against simply sampling more --
which the earlier 50-seed comparison did NOT do. Pooled-ipTM pick means: take the single highest-ipTM
pose in the pool and report its DockQ. That is what a user actually gets.

ONLY INTERFACES WHERE BOTH POOLS ARE COMPLETE are reported, and the per-arm seed counts are printed, so a
half-finished interface is never mistaken for a result. Each pool's oracle is shown too: if the templated
pool's oracle drops, self-templating is destroying good poses rather than failing to surface them.

Usage: gen2_report.py [--min-seeds 75] [--csv path]
"""
import argparse
import collections
import csv
import statistics as st

AP = 0.23
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def load():
    free = collections.defaultdict(lambda: collections.defaultdict(list))   # pxm pool, 50 seeds
    for r in csv.DictReader(open(f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv")):
        try:
            free[f"{r['pdb']}_{r['chain1']}{r['chain2']}"][r["seed"]].append(
                (float(r["dockq"]), float(r["pair_iptm"])))
        except (TypeError, ValueError):
            continue
    gen1 = collections.defaultdict(lambda: collections.defaultdict(list))
    cls = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/hybrid25/results_all.csv")):
        try:
            gen1[r["interface"]][r["new_seed"]].append((float(r["dockq"]), float(r["iptm"])))
            cls[r["interface"]] = r["cls"]
        except (TypeError, ValueError):
            continue
    ours = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    seen = set()
    for r in csv.DictReader(open(f"{EXP}/artifacts/commit11/results.csv")):
        if r["arm"] not in ("gen2", "free75") or not r["iptm"]:
            continue
        k = (r["iface"], r["seed"], r["arm"], r["sample"], r["dockq"])
        if k in seen:
            continue
        seen.add(k)
        try:
            ours[r["iface"]][r["arm"]][r["seed"]].append((float(r["dockq"]), float(r["iptm"])))
        except ValueError:
            continue
    return free, gen1, cls, ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=75)
    A = ap.parse_args()
    free, gen1, cls, ours = load()

    rows, partial = [], 0
    for iface in sorted(gen1):
        F, G1 = free.get(iface), gen1[iface]
        G2 = ours[iface]["gen2"]
        F25 = ours[iface]["free75"]
        if not F:
            continue
        fs = sorted(F)
        donors = [s for s in fs if s < "300025"][:25]
        pool_free = [x for s in fs[:50] for x in F[s]] + [x for s in sorted(F25) for x in F25[s]]
        n_free = min(len(fs), 50) + len(F25)
        pool_hyb = ([x for s in donors for x in F[s]]
                    + [x for s in sorted(G1)[:25] for x in G1[s]]
                    + [x for s in sorted(G2) for x in G2[s]])
        n_hyb = len(donors) + min(len(G1), 25) + len(G2)
        if n_free < A.min_seeds or n_hyb < A.min_seeds:
            partial += 1
            continue
        rows.append(dict(iface=iface, cls=cls.get(iface, "?"), nf=n_free, nh=n_hyb,
                         self=max(pool_free, key=lambda x: x[1])[0],
                         selh=max(pool_hyb, key=lambda x: x[1])[0],
                         orcf=max(x[0] for x in pool_free),
                         orch=max(x[0] for x in pool_hyb)))
    print(f"  complete at {A.min_seeds} seeds per arm: {len(rows)} interfaces   "
          f"(still filling: {partial})")
    if not rows:
        return
    print(f"\n  {'class':<8} {'n':>4} {'sel FREE-75':>12} {'sel HYBRID-75':>14} {'delta':>8} | "
          f"{'orc FREE':>9} {'orc HYB':>8} | better/worse")
    for c in ["solved", "decoy", "nocov", "?"]:
        R = [r for r in rows if r["cls"] == c]
        if not R:
            continue
        f = [r["self"] for r in R]; h = [r["selh"] for r in R]
        w = sum(1 for r in R if r["selh"] > r["self"] + 1e-9)
        l = sum(1 for r in R if r["selh"] < r["self"] - 1e-9)
        print(f"  {c:<8} {len(R):>4} {st.mean(f):>12.3f} {st.mean(h):>14.3f} "
              f"{st.mean(h)-st.mean(f):>+8.3f} | {st.mean(r['orcf'] for r in R):>9.3f} "
              f"{st.mean(r['orch'] for r in R):>8.3f} | {w}/{l}")
    f = [r["self"] for r in rows]; h = [r["selh"] for r in rows]
    w = sum(1 for r in rows if r["selh"] > r["self"] + 1e-9)
    l = sum(1 for r in rows if r["selh"] < r["self"] - 1e-9)
    d = [r["selh"] - r["self"] for r in rows]
    print(f"  {'ALL':<8} {len(rows):>4} {st.mean(f):>12.3f} {st.mean(h):>14.3f} "
          f"{st.mean(h)-st.mean(f):>+8.3f} | {st.mean(r['orcf'] for r in rows):>9.3f} "
          f"{st.mean(r['orch'] for r in rows):>8.3f} | {w}/{l}")
    print(f"\n  acceptable picks: FREE-75 {sum(1 for x in f if x>=AP)}/{len(rows)}  ->  "
          f"HYBRID-75 {sum(1 for x in h if x>=AP)}/{len(rows)}")
    print(f"  paired delta: mean {st.mean(d):+.4f}   median {st.median(d):+.4f}   "
          f"({w} up / {l} down / {len(rows)-w-l} tied)")
    print("  NOTE: the median is the number to watch -- the 50-seed version had mean +0.010 with a "
          "median of exactly 0.000, i.e. a few large rescues rather than a broad shift.")


if __name__ == "__main__":
    main()
