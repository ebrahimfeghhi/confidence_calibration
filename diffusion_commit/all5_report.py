#!/usr/bin/env python
"""THE END-GOAL TABLE: per interface, the pooled-ipTM pick after 10 recycles vs after 11.

For one interface, pool every pose of every seed and take the single highest-ipTM pose -- that is what a
user gets. `donor` is the pool after 10 recycles + diffusion; `all5_k1` is the pool after one further
recycle resuming that seed's trunk with all five of its own poses as the sole template.

MATCHED POOLS ONLY. A seed counts only if BOTH arms have their full complement of poses, because an
unmatched pool would differ in depth as well as in method and the comparison would confound the two. The
seed count is reported per interface so a partial interface is never mistaken for a finished one.

The oracle of each pool is reported too: if the 11-recycle pool's oracle drops, the arm is destroying good
poses rather than merely failing to surface them, which is a different failure and a different fix.

Usage: all5_report.py [--csv path] [--min-seeds N]
"""
import argparse
import collections
import csv
import statistics as st

AP = 0.23
CSV = "/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/commit11/results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="only report interfaces with at least this many matched seeds")
    A = ap.parse_args()

    seen = set()
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(A.csv)):
        if r["arm"] not in ("donor", "all5_k1") or not r["iptm"]:
            continue
        k = (r["iface"], r["seed"], r["arm"], r["sample"], r["dockq"])
        if k in seen:                      # exact duplicates predate score()'s dedup guard
            continue
        seen.add(k)
        try:
            by[r["iface"]][(r["seed"], r["arm"])].append((float(r["dockq"]), float(r["iptm"])))
        except ValueError:
            continue

    rows = []
    for iface, d in sorted(by.items()):
        seeds = [s for s in {k[0] for k in d}
                 if len(d.get((s, "donor"), [])) >= 5 and len(d.get((s, "all5_k1"), [])) >= 5]
        if len(seeds) < A.min_seeds:
            continue
        P10 = [x for s in seeds for x in d[(s, "donor")]]
        P11 = [x for s in seeds for x in d[(s, "all5_k1")]]
        rows.append(dict(iface=iface, n=len(seeds),
                         sel10=max(P10, key=lambda x: x[1])[0],
                         sel11=max(P11, key=lambda x: x[1])[0],
                         orc10=max(x[0] for x in P10),
                         orc11=max(x[0] for x in P11)))
    if not rows:
        print("  no interface has matched pools yet"); return

    print(f"  {'interface':<12} {'seeds':>5} | {'sel@10':>7} {'sel@11':>7} {'delta':>7} | "
          f"{'orc@10':>7} {'orc@11':>7}")
    print("  " + "-" * 66)
    for r in sorted(rows, key=lambda r: -(r["sel11"] - r["sel10"])):
        flag = ""
        if r["sel11"] >= AP > r["sel10"]:
            flag = "  RESCUED"
        elif r["sel10"] >= AP > r["sel11"]:
            flag = "  LOST"
        print(f"  {r['iface']:<12} {r['n']:>5} | {r['sel10']:>7.3f} {r['sel11']:>7.3f} "
              f"{r['sel11']-r['sel10']:>+7.3f} | {r['orc10']:>7.3f} {r['orc11']:>7.3f}{flag}")

    s10 = [r["sel10"] for r in rows]; s11 = [r["sel11"] for r in rows]
    w = sum(1 for r in rows if r["sel11"] > r["sel10"] + 1e-9)
    l = sum(1 for r in rows if r["sel11"] < r["sel10"] - 1e-9)
    resc = sum(1 for r in rows if r["sel11"] >= AP > r["sel10"])
    lost = sum(1 for r in rows if r["sel10"] >= AP > r["sel11"])
    print(f"\n  n={len(rows)} interfaces, {sum(r['n'] for r in rows)} matched seeds")
    print(f"  mean pooled-ipTM DockQ   recycle10 {st.mean(s10):.3f}  ->  recycle11 {st.mean(s11):.3f}   "
          f"delta {st.mean(s11)-st.mean(s10):+.3f}")
    print(f"  acceptable picks         recycle10 {sum(1 for x in s10 if x>=AP)}/{len(rows)}  ->  "
          f"recycle11 {sum(1 for x in s11 if x>=AP)}/{len(rows)}")
    print(f"  better/worse/tied {w}/{l}/{len(rows)-w-l}    crossed 0.23: rescued {resc}, lost {lost}")
    print(f"  mean oracle in pool      recycle10 {st.mean([r['orc10'] for r in rows]):.3f}  ->  "
          f"recycle11 {st.mean([r['orc11'] for r in rows]):.3f}")
    print("  (interfaces below 50 seeds are still running -- their pools are shallower, so a delta "
          "there is not yet the final number)")


if __name__ == "__main__":
    main()
