#!/usr/bin/env python
"""Compare the self-templating generations on 9x1h: free -> gen 1 -> gen 2.

Each generation templates on the best pose of the one before, with the cross-chain mask off and
per-chain centring disabled so real inter-chain geometry survives.

WHAT TO WATCH. The interesting failure mode is not a crash but silent drift: DockQ falling while
ipTM keeps climbing, i.e. the donor wandering off the native basin while the model grows more certain
about the place it has wandered to. That is what generational self-templating did on ESMFold2
(selected DockQ 0.587 -> 0.511 over generations, n=35). So DockQ and ipTM are reported together and
per generation, never a single summary number.

Reports the DONOR's own DockQ alongside, because a generation that fails to beat its own donor has
not refined anything -- it has just reproduced the input.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/read_gen.py
"""
import glob
import json
import os
import re

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
GENS = [
    ("free (good seeds)", None, None, None),
    ("gen 1  (donor 0.683)", f"{EXP}/artifacts/tmplbest", "9x1h_tmplbest", 0.683),
    ("gen 2  (donor 0.945)", f"{EXP}/artifacts/tmplgen2", "9x1h_tmplbest", 0.945),
]


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def read(root, jobname):
    dq, ip = [], []
    for f in sorted(glob.glob(f"{root}/pxm_json/*.json")):
        m = re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            J = json.load(open(f))
        except Exception:
            continue
        v = (J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
        if not v or v.get("dockq") is None:
            continue
        dq.append(float(v["dockq"]))
        r2m = J.get("ref_to_model_chain_mapping", {})
        # gen dirs use the job name for both the seed dir and the file prefix
        cf = (f"{root}/out/{jobname}/seed_{m.group(1)}/predictions/"
              f"{jobname}_summary_confidence_sample_{m.group(2)}.json")
        p = float("nan")
        if os.path.exists(cf) and all(c in r2m for c in ("C", "D")):
            M = json.load(open(cf)).get("chain_pair_iptm")
            i, j = chain_idx(r2m["C"]), chain_idx(r2m["D"])
            if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                p = float(M[i][j])
        ip.append(p)
    return np.array(dq), np.array(ip)


def main():
    print(f"\n9x1h C-D, recursive self-templating")
    print(f"{'generation':<22} {'n':>4} {'donor':>6} {'DockQ mean':>11} {'min':>7} {'max':>7} "
          f"{'acc':>7} {'ipTM mean':>10} {'ipTM max':>9}")
    print(f"{'free, confident decoy':<22} {5:>4} {'-':>6} {0.019:>11.3f} {'-':>7} {'-':>7} "
          f"{'0/5':>7} {0.700:>10.3f} {0.700:>9.3f}")
    print(f"{'free, good seed 300038':<22} {5:>4} {'-':>6} {0.201:>11.3f} {0.021:>7.3f} "
          f"{0.615:>7.3f} {'2/5':>7} {0.120:>10.3f} {0.120:>9.3f}")
    for name, root, jobname, donor in GENS:
        if root is None:
            continue
        if not os.path.isdir(f"{root}/pxm_json"):
            print(f"{name:<22} {'-- not run --':>20}")
            continue
        d, p = read(root, jobname)
        if not len(d):
            print(f"{name:<22} {'-- no scores --':>20}")
            continue
        print(f"{name:<22} {len(d):>4} {donor:>6.3f} {d.mean():>11.3f} {d.min():>7.3f} "
              f"{d.max():>7.3f} {str(int((d>=0.23).sum()))+'/'+str(len(d)):>7} "
              f"{np.nanmean(p):>10.3f} {np.nanmax(p):>9.3f}")
        if donor is not None:
            beat = int((d > donor).sum())
            print(f"{'':<22} poses beating their own donor: {beat}/{len(d)}"
                  f"   mean vs donor: {d.mean()-donor:+.3f}")
    print()


if __name__ == "__main__":
    main()
