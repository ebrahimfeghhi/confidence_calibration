#!/usr/bin/env python
"""In-memory self-templating: the continuation vs the poses it templated on, within the same fold.

THE COMPARISON IS WITHIN ONE FOLD, which is what makes it clean. The hook decodes at recycle 10 and
those poses ARE the free baseline -- same seed, same run, same basin draw. Comparing against a
separately generated free pool would reintroduce the bf16 basin lottery: the pool's fold of 9h01 seed
300001 found a 0.776 pose where a fresh fold of the same seed found nothing above 0.008, so a
"shortfall" could be donor luck rather than the mechanism.

NO ipTM AT RECYCLE 10. The confidence head does not run at the mid-loop decode, so the at10 arm has
DockQ only. That rules out comparing SELECTED poses (selection needs ipTM on both sides), so the
honest comparisons are:

    best     did the continuation produce a better pose than it was given?   (oracle to oracle)
    mean     did the pose set improve overall?
    selected for the continuation only -- what its own ipTM would ship

CONDITIONED ON THE DONOR SET. A continuation cannot amplify a good pose that was never there, so the
rescue rate is reported split by whether at10 already contained an acceptable pose. Pooling those two
cases hides whether the mechanism works or the donors were lucky.

    inmem_compare.py --at10 <csv> --arms 11:<csv> 12:<csv> ... [--thresh 0.23]
"""
import argparse
import collections
import csv
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

from fmt import table   # noqa: E402


def load(path, want_arm=None):
    """{(iface, seed): [{'dq','ip'}]} -- ip is None when the csv has no ipTM (the at10 arm)."""
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if want_arm and r["arm"] != want_arm:
            continue
        ip = r["iptm"].strip()
        out[(r["iface"], r["seed"])].append({"dq": float(r["dockq"]),
                                             "ip": float(ip) if ip else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at10", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="N:path/to/nN.csv")
    ap.add_argument("--thresh", type=float, default=0.23)
    A = ap.parse_args()
    T = A.thresh

    base = load(A.at10, want_arm="at10")
    print(f"  at10 (donor) pose sets: {len(base)}\n")

    summary = []
    for spec in A.arms:
        n, path = spec.split(":", 1)
        try:
            arm = load(path)
        except FileNotFoundError:
            print(f"  ncycle {n}: {path} not present yet")
            continue
        keys = sorted(set(base) & set(arm))
        if not keys:
            print(f"  ncycle {n}: no overlap with at10 yet")
            continue
        rows = []
        for k in keys:
            b, a = base[k], arm[k]
            bb, ab = max(p["dq"] for p in b), max(p["dq"] for p in a)
            ips = [p for p in a if p["ip"] is not None]
            sel = max(ips, key=lambda p: p["ip"])["dq"] if ips else None
            rows.append([k[0], k[1], bb, ab, ab - bb,
                         st.mean([p["dq"] for p in b]), st.mean([p["dq"] for p in a]), sel])
        rows.sort(key=lambda r: -r[4])
        print(f"=== ncycle {n}  (n={len(rows)} folds)")
        print(table(["interface", "seed", "at10 best", "cont best", "d best",
                     "at10 mean", "cont mean", "cont selected"], rows[:12],
                    align="l l r r r r r r",
                    dec={"at10 best": 3, "cont best": 3, "d best": 3, "at10 mean": 3,
                         "cont mean": 3, "cont selected": 3}))
        # split by whether the donor set had anything worth amplifying
        had = [r for r in rows if r[2] >= T]
        hadnt = [r for r in rows if r[2] < T]
        blk = []
        for lab, sub in (("donor HAD a good pose", had), ("donor had NONE", hadnt), ("all", rows)):
            if not sub:
                continue
            sels = [r[7] for r in sub if r[7] is not None]
            blk.append([lab, len(sub), st.mean([r[2] for r in sub]), st.mean([r[3] for r in sub]),
                        st.mean([r[4] for r in sub]),
                        sum(1 for r in sub if r[4] > 0.05), sum(1 for r in sub if r[4] < -0.05),
                        f"{sum(1 for x in sels if x >= T)}/{len(sels)}" if sels else "-"])
        print(table(["donor set", "folds", "at10 best", "cont best", "mean d",
                     "cont better", "cont worse", f"cont selected ok(>={T})"], blk,
                    align="l r r r r r r c",
                    dec={"at10 best": 3, "cont best": 3, "mean d": 3}))
        print()
        summary.append([n, len(rows), st.mean([r[4] for r in rows]),
                        sum(1 for r in rows if r[4] > 0.05), sum(1 for r in rows if r[4] < -0.05)])
    if len(summary) > 1:
        print("=== across recycle counts")
        print(table(["ncycle", "folds", "mean d best", "better", "worse"], summary,
                    align="r r r r r", dec={"mean d best": 3}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
