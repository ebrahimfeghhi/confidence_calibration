#!/usr/bin/env python
"""Read back one tmplctl arm: DockQ and interface pair ipTM over its 5 samples.

WHY ITS OWN READER. The comparison being made is against hybrid25's recorded numbers for the same
unit (donor 300021 -> run seed 300046: DockQ 0.061, ipTM 0.824), so this must compute DockQ and ipTM
the same way hybrid25_all_interfaces.py does or the comparison is meaningless:

  - DockQ comes from the pose's own PXMeter JSON, keyed on REFERENCE chains (A,C).
  - ipTM is chain_pair_iptm, indexed by MODEL asym order, so each pose's own
    ref_to_model_chain_mapping is required to line the two up. Reference letters are NOT model
    letters -- substituting one for the other once put 45% of this project's interfaces on the wrong
    chains.
  - It is the per-interface PAIR ipTM, never the global iptm, which blends intra-antibody confidence
    and hides the interface signal.

Usage: tmplctl_read.py --tag run1 [--field iptm]
"""
import argparse
import glob
import json
import os

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
S = f"{EXP}/artifacts/tmplctl"
C1, C2 = "A", "C"


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def read(tag):
    rows = []
    for jf in sorted(glob.glob(f"{S}/{tag}/pxm_json/*.json")):
        samp = os.path.basename(jf).split(".")[0]
        try:
            J = json.load(open(jf))
        except Exception:
            continue
        v = (J.get("interface") or {}).get(f"{C1},{C2}") or \
            (J.get("interface") or {}).get(f"{C2},{C1}")
        if not v or v.get("dockq") is None:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        ip = np.nan
        cf = glob.glob(f"{S}/{tag}/out/**/*summary_confidence_sample_{samp}.json", recursive=True)
        if cf and C1 in r2m and C2 in r2m:
            try:
                conf = json.load(open(cf[0])).get("chain_pair_iptm")
                i, j = chain_idx(r2m[C1]), chain_idx(r2m[C2])
                if conf and 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    ip = float(conf[i][j])
            except Exception:
                pass
        rows.append((int(samp), float(v["dockq"]), ip))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--summary", action="store_true",
                    help="one row per condition, plus whether its confidence-head dump exists")
    ap.add_argument("--field", default="", choices=["", "iptm", "dockq"],
                    help="print just this scalar (max over samples), for shell gating")
    A = ap.parse_args()

    if A.summary:
        LAB = {"run0": "free, donor seed 300021", "run0b": "free, seed 300046",
               "run1": "5 self-tmpl, seed 300046",
               "run1b": "5 self-tmpl, seed 300021",
               "run2": "4 real + 5 self", "run3": "5 self, full MSA, no drop"}
        print(f"\n{'arm':<6} {'condition':<28} {'n':>2} {'DockQ mean':>11} {'DockQ max':>10} "
              f"{'ipTM mean':>10} {'ipTM max':>9} {'dump':>6}")
        for tag in ("run0", "run0b", "run1", "run1b", "run2", "run3"):
            r = read(tag)
            dump = len(glob.glob(f"{S}/{tag}/full/*_reps.npz"))
            if not r:
                print(f"{tag:<6} {LAB[tag]:<28} {'-':>2}")
                continue
            d = np.array([x[1] for x in r])
            p = np.array([x[2] for x in r], dtype=float)
            q = p[np.isfinite(p)]
            print(f"{tag:<6} {LAB[tag]:<28} {len(r):>2} {d.mean():>11.3f} {d.max():>10.3f} "
                  f"{(q.mean() if q.size else np.nan):>10.3f} "
                  f"{(q.max() if q.size else np.nan):>9.3f} {dump:>6}")
        print("\nhybrid25 reference (donor 300021 -> seed 300046): DockQ 0.061  ipTM 0.824")
        print("Arm A free, donor seed 300021: ipTM 0.315   |  Arm A free, seed 300046: ipTM 0.245")
        return

    if not A.tag:
        raise SystemExit("need --tag or --summary")
    rows = read(A.tag)
    if A.field:
        if not rows:
            raise SystemExit(1)
        col = 2 if A.field == "iptm" else 1
        v = [r[col] for r in rows if np.isfinite(r[col])]
        if not v:
            raise SystemExit(1)
        print(f"{max(v):.4f}")
        return

    if not rows:
        print(f"{A.tag}: NO SCORED SAMPLES")
        return
    d = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows], dtype=float)
    print(f"{A.tag}: n={len(rows)}")
    for s, dq, ip in rows:
        print(f"   sample {s}  DockQ {dq:.3f}  ipTM " + (f"{ip:.3f}" if np.isfinite(ip) else "  n/a"))
    print(f"   DockQ mean {d.mean():.3f}  max {d.max():.3f}  range {d.max()-d.min():.3f}")
    if np.isfinite(p).any():
        q = p[np.isfinite(p)]
        print(f"   ipTM  mean {q.mean():.3f}  max {q.max():.3f}  range {q.max()-q.min():.3f}")
    print(f"   hybrid25 reference for this unit: DockQ mean 0.061 max 0.069 | "
          f"ipTM mean 0.824 max 0.824")


if __name__ == "__main__":
    main()
