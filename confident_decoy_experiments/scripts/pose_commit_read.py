#!/usr/bin/env python
"""Read back the pose-commit arms: did committing to each pose separate their confidences?

ONE ARM PER POSE, all started from the same cached 10-recycle trunk state and given one templated
recycle, so the only difference between arms is which pose was shown. The baseline ("none") is an
11th recycle with no template, which separates "one more recycle" from "a templated recycle".

WHAT TO LOOK FOR. In the original seed the five poses span DockQ 0.044-0.634 while ipTM spans 0.034 --
confidence is blind within a seed. If the per-arm ipTMs now TRACK the donor pose's DockQ, that is a
selector. If every arm rises to the same high value regardless of donor quality, that is the
confidence-attractor result, reproduced within a single seed and therefore free of the seed-to-seed
confounds that muddied the hybrid25 version.

DockQ is read from each pose's own PXMeter json and ipTM is chain_pair_iptm indexed through that
pose's ref_to_model_chain_mapping -- reference chain letters are NOT model chain letters, and
substituting one for the other once put 45% of this project's interfaces on the wrong chains.

Usage: pose_commit_read.py --dir artifacts/posecommit/9x1h_300014 [--iface EF]
"""
import argparse
import glob
import json
import os

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def arm(d, c1, c2):
    rows = []
    for jf in sorted(glob.glob(f"{d}/pxm_json/*.json")):
        k = os.path.basename(jf).split(".")[0]
        try:
            J = json.load(open(jf))
        except Exception:
            continue
        v = (J.get("interface") or {}).get(f"{c1},{c2}") or (J.get("interface") or {}).get(f"{c2},{c1}")
        if not v or v.get("dockq") is None:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        ip = np.nan
        cf = glob.glob(f"{d}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
        if cf and c1 in r2m and c2 in r2m:
            try:
                conf = json.load(open(cf[0])).get("chain_pair_iptm")
                i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                if conf and 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    ip = float(conf[i][j])
            except Exception:
                pass
        rows.append((int(k), float(v["dockq"]), ip))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iface", default="EF")
    ap.add_argument("--pdb", default="9x1h")
    ap.add_argument("--seed", type=int, default=300014)
    A = ap.parse_args()
    c1, c2 = A.iface[0], A.iface[1]

    # the donor poses' own DockQ, from the original free run
    import csv
    donor = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/pxm2025/dockq_pxm2025.csv")):
        if (r["pdb"] == A.pdb and int(r["seed"]) == A.seed and r["dockq"]
                and "".join(sorted((r["chain1"], r["chain2"]))) == "".join(sorted((c1, c2)))):
            donor[int(r["sample"])] = (float(r["dockq"]),
                                       float(r["pair_iptm"]) if r["pair_iptm"] else np.nan)

    print(f"\n{A.pdb}_{A.iface} seed {A.seed}: donor poses (original free run)")
    for k in sorted(donor):
        print(f"   pose {k}  DockQ {donor[k][0]:.3f}  ipTM {donor[k][1]:.3f}")
    if donor:
        d = [v[0] for v in donor.values()]; p = [v[1] for v in donor.values()]
        print(f"   -> DockQ span {max(d)-min(d):.3f}   ipTM span {np.nanmax(p)-np.nanmin(p):.3f}")

    print(f"\nper-arm results (one templated 11th recycle each, same starting trunk state)")
    print(f"{'arm':<6} {'donor DockQ':>11} {'n':>2} {'DockQ mean':>11} {'max':>7} {'spread':>7} "
          f"{'ipTM mean':>10} {'max':>7}")
    out = []
    for tag in ["none"] + [f"p{i}" for i in range(10)]:
        d = f"{A.dir}/{tag}"
        if not os.path.isdir(d):
            continue
        rows = arm(d, c1, c2)
        if not rows:
            print(f"{tag:<6} {'':>11} {'0':>2}   (no scored poses)")
            continue
        dq = np.array([r[1] for r in rows]); ip = np.array([r[2] for r in rows], dtype=float)
        dn = donor.get(int(tag[1:]), (np.nan,))[0] if tag.startswith("p") else np.nan
        print(f"{tag:<6} {dn:>11.3f} {len(rows):>2} {dq.mean():>11.3f} {dq.max():>7.3f} "
              f"{dq.max()-dq.min():>7.3f} {np.nanmean(ip):>10.3f} {np.nanmax(ip):>7.3f}")
        out.append(dict(arm=tag, donor_dockq=None if np.isnan(dn) else float(dn),
                        n=len(rows), dockq_mean=float(dq.mean()), dockq_max=float(dq.max()),
                        dockq_spread=float(dq.max() - dq.min()),
                        iptm_mean=float(np.nanmean(ip)), iptm_max=float(np.nanmax(ip))))

    pose_arms = [o for o in out if o["donor_dockq"] is not None and not np.isnan(o["iptm_mean"])]
    if len(pose_arms) >= 3:
        x = np.array([o["donor_dockq"] for o in pose_arms])
        y = np.array([o["iptm_mean"] for o in pose_arms])
        print(f"\nDOES CONFIDENCE NOW TRACK POSE QUALITY?")
        print(f"   corr(donor DockQ, arm ipTM) = {np.corrcoef(x, y)[0,1]:+.3f} over {len(x)} arms")
        print(f"   arm ipTM span {y.max()-y.min():.3f}   (original within-seed ipTM span was 0.034)")
        best_by_iptm = pose_arms[int(np.argmax(y))]
        print(f"   highest-ipTM arm is p{best_by_iptm['arm'][1:]} "
              f"(donor DockQ {best_by_iptm['donor_dockq']:.3f}); "
              f"best donor was DockQ {x.max():.3f}")
    json.dump(out, open(f"{A.dir}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
