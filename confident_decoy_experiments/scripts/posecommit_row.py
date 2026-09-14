#!/usr/bin/env python
"""Emit one CSV row per scored pose of a pose-commit arm. Called by ops/posecommit_worker.sh.

DockQ comes from the pose's own PXMeter json, keyed on REFERENCE chain letters. ipTM is
chain_pair_iptm, which is indexed by MODEL asym order, so each pose's own ref_to_model_chain_mapping
is required to line the two up -- reference letters are NOT model letters, and substituting one for
the other once put 45% of this project's interface measurements on the wrong chains.

Prints to stdout so the worker can append; writes nothing itself.

Usage: posecommit_row.py --dir <armdir> --pdb 9s36 --iface 9s36_AB --seed 300014 \
                         --arm p3_off_c1 --msa off --ncyc 1 --donor 3
"""
import argparse
import glob
import json
import os


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    ap = argparse.ArgumentParser()
    for a in ("--dir", "--pdb", "--iface", "--seed", "--arm", "--msa", "--ncyc", "--donor"):
        ap.add_argument(a, required=True)
    A = ap.parse_args()
    c1, c2 = A.iface.split("_")[1][0], A.iface.split("_")[1][1]

    for jf in sorted(glob.glob(f"{A.dir}/pxm_json/*.json")):
        k = os.path.basename(jf).split(".")[0]
        try:
            J = json.load(open(jf))
        except Exception:
            continue
        v = (J.get("interface") or {}).get(f"{c1},{c2}") or \
            (J.get("interface") or {}).get(f"{c2},{c1}")
        if not v or v.get("dockq") is None:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        ip = ""
        cf = glob.glob(f"{A.dir}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
        if cf and c1 in r2m and c2 in r2m:
            try:
                conf = json.load(open(cf[0])).get("chain_pair_iptm")
                i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                if conf and 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    ip = f"{float(conf[i][j]):.4f}"
            except Exception:
                pass
        print(f"{A.pdb},{A.iface},{A.seed},{A.arm},{A.msa},{A.ncyc},{A.donor},{k},"
              f"{float(v['dockq']):.4f},{ip}")


if __name__ == "__main__":
    main()
