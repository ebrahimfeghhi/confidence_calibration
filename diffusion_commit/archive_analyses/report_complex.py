#!/usr/bin/env python
"""Summarise one interface's fold+commit results: does committing to a pose let confidence rank poses?

THE TEST. Within a seed, all five donor poses come from ONE trunk state, so ipTM -- a property of that
state -- cannot rank them. Each p-arm re-runs one recycle from that same trunk with pose k as its only
template, giving pose k its own trunk state. If the arms' confidence tracks the DockQ of the pose they
were shown, that is a selector.

WHAT DECIDES WHETHER A SEED CAN ANSWER. Only seeds whose donors STRADDLE DockQ 0.23 -- at least one
acceptable pose and at least one decoy -- can test selection. A seed whose five poses are all good (or
all bad) has a flat donor axis: any arm ordering is consistent with both hypotheses. Those seeds are
counted and then excluded, and the exclusion is reported, because averaging them in dilutes the effect
toward zero and would read as a null.

Usage: report_complex.py --iface 9x1h_CD [--csv path]
"""
import argparse
import collections
import csv
import statistics as st

AP = 0.23


def load(csv_path, iface):
    """-> {seed: {arm: {sample: (dockq, iptm)}}}, dropping exact duplicate rows."""
    seen, out = set(), collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(csv_path)):
        if r["iface"] != iface:
            continue
        key = (r["seed"], r["arm"], r["sample"], r["dockq"])
        if key in seen:        # a repaired arm re-scored before score() had its dedup guard
            continue
        seen.add(key)
        try:
            out[r["seed"]][r["arm"]][r["sample"]] = (
                float(r["dockq"]), float(r["iptm"]) if r["iptm"] else None)
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--csv", default="/home/ebrahim/tts_abag/confident_decoy_experiments"
                                    "/artifacts/commit11/results.csv")
    A = ap.parse_args()
    D = load(A.csv, A.iface)
    if not D:
        print(f"{A.iface}: no rows yet")
        return

    complete, straddle = [], []
    for seed, arms in D.items():
        don = arms.get("donor") or {}
        if len(don) < 5 or not all(f"p{i}" in arms for i in range(5)):
            continue
        complete.append(seed)
        dq = [v[0] for v in don.values()]
        if min(dq) < AP <= max(dq):
            straddle.append(seed)

    print(f"===== {A.iface} =====")
    print(f"  seeds with all 7 stages: {len(complete)}   of which straddle {AP}: {len(straddle)}")
    if not complete:
        return

    # Baseline: what plain ipTM selection does across the donor pools of the complete seeds.
    poses = [v for s in complete for v in D[s]["donor"].values() if v[1] is not None]
    if poses:
        sel = max(poses, key=lambda x: x[1])
        orc = max(poses, key=lambda x: x[0])
        print(f"  BASELINE pooled ipTM pick: DockQ {sel[0]:.3f}   oracle {orc[0]:.3f}   "
              f"gap {orc[0]-sel[0]:.3f}")

    if not straddle:
        print("  no straddling seeds -> this interface CANNOT test selection (see module docstring)")
        return

    # The test, per straddling seed: rank the five p-arms by their own mean ipTM, then ask what the
    # DockQ of the donor pose shown to the winning arm was. Compare against ranking the donor poses by
    # their own ipTM directly, which is what a user gets today.
    win_commit, win_plain, orac = [], [], []
    print(f"\n  {'seed':>6} {'best donor':>10} {'plain ipTM':>11} {'committed':>10}")
    for s in sorted(straddle, key=lambda x: int(x) if x.isdigit() else 0):
        don = D[s]["donor"]
        best = max(v[0] for v in don.values())
        plain = max(don.values(), key=lambda v: v[1] if v[1] is not None else -1)[0]
        scores = {}
        for i in range(5):
            ip = [v[1] for v in D[s][f"p{i}"].values() if v[1] is not None]
            if ip:
                scores[i] = st.mean(ip)
        if not scores:
            continue
        k = max(scores, key=scores.get)
        shown = don.get(str(k))
        if shown is None:
            continue
        win_commit.append(shown[0]); win_plain.append(plain); orac.append(best)
        print(f"  {s:>6} {best:>10.3f} {plain:>11.3f} {shown[0]:>10.3f}")

    if win_commit:
        n = len(win_commit)
        print(f"\n  n={n} straddling seeds")
        print(f"  mean selected DockQ -- plain ipTM {st.mean(win_plain):.3f}   "
              f"COMMITTED {st.mean(win_commit):.3f}   oracle {st.mean(orac):.3f}")
        pw = sum(1 for x in win_plain if x >= AP)
        cw = sum(1 for x in win_commit if x >= AP)
        print(f"  picks an acceptable pose -- plain {pw}/{n}   COMMITTED {cw}/{n}")
        print("  (n is small per interface; pool across interfaces before claiming an effect)")


if __name__ == "__main__":
    main()
