#!/usr/bin/env python
"""Read the two no-dump replicates of 9x1h seed 300014 and say which explanation survives.

Reference points, all on chain pair CD (identical conclusions on EF/AF/BD):
    sweep                 DockQ max 0.643, mean 0.325, SD 0.268, pair_iptm 0.131
    rerun with DUMP_FULL  DockQ max 0.018, SD 0.000,             pair_iptm 0.506

PXMeter's interface keys are REFERENCE chain labels and chain_pair_iptm is indexed by MODEL asym
order, so ref_to_model_chain_mapping is required to line them up -- on 9x1h it is not the identity.
Same extraction as scripts/score_pxm2025.py.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/read_probe14.py
"""
import glob
import json
import os
import re

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
OUT = f"{EXP}/artifacts/probe14"
PAIR = ("C", "D")
SWEEP = dict(mx=0.643, mean=0.325, sd=0.268, iptm=0.131)
DUMPRUN = dict(mx=0.018, mean=0.018, sd=0.000, iptm=0.506)


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def read(rep):
    dq, ip = [], []
    for f in sorted(glob.glob(f"{OUT}/pxm_json/rep{rep}_*.json")):
        smp = re.search(r"_(\d+)\.json$", os.path.basename(f))
        if not smp:
            continue
        try:
            J = json.load(open(f))
        except json.JSONDecodeError:
            continue
        v = (J.get("interface") or {}).get(",".join(PAIR)) or \
            (J.get("interface") or {}).get(",".join(PAIR[::-1]))
        if not v or v.get("dockq") is None:
            continue
        dq.append(float(v["dockq"]))
        r2m = J.get("ref_to_model_chain_mapping", {})
        cf = (f"{OUT}/rep{rep}/9x1h/seed_300014/predictions/"
              f"9x1h_summary_confidence_sample_{smp.group(1)}.json")
        if os.path.exists(cf) and all(c in r2m for c in PAIR):
            M = json.load(open(cf)).get("chain_pair_iptm")
            i, j = chain_idx(r2m[PAIR[0]]), chain_idx(r2m[PAIR[1]])
            if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                ip.append(float(M[i][j]))
    return np.array(dq), np.array(ip)


def main():
    print("\n9x1h seed 300014, chain pair CD -- no-dump replicates vs the two reference runs")
    print(f"{'run':<26} {'DockQ max':>10} {'DockQ mean':>11} {'DockQ SD':>9} {'ipTM mean':>10} {'n':>3}")
    print(f"{'sweep (50-seed process)':<26} {SWEEP['mx']:>10.3f} {SWEEP['mean']:>11.3f} "
          f"{SWEEP['sd']:>9.3f} {SWEEP['iptm']:>10.3f} {5:>3}")
    print(f"{'rerun WITH dump':<26} {DUMPRUN['mx']:>10.3f} {DUMPRUN['mean']:>11.3f} "
          f"{DUMPRUN['sd']:>9.3f} {DUMPRUN['iptm']:>10.3f} {5:>3}")
    res = {}
    for rep in (1, 2):
        dq, ip = read(rep)
        if not len(dq):
            print(f"{'replicate ' + str(rep) + ' (no dump)':<26} {'-- no scores --':>10}")
            continue
        res[rep] = (dq, ip)
        print(f"{'replicate ' + str(rep) + ' (no dump)':<26} {dq.max():>10.3f} {dq.mean():>11.3f} "
              f"{dq.std(ddof=1):>9.3f} {ip.mean() if len(ip) else np.nan:>10.3f} {len(dq):>3}")

    if len(res) < 2:
        print("\nboth replicates needed for a verdict\n")
        return
    d1, d2 = res[1][0], res[2][0]
    good1, good2 = d1.max() >= 0.23, d2.max() >= 0.23
    print("\nVERDICT")
    if good1 or good2:
        if good1 and good2:
            print("  Both no-dump replicates recovered an acceptable pose, while the dumped run did"
                  " not.\n  => the DUMP is implicated after all; the code reading was wrong.")
        else:
            print("  The replicates DISAGREE with each other (one good, one not).\n"
                  "  => basin-commitment LOTTERY under CUDA nondeterminism. The dump is not special;"
                  "\n     re-roll seed 300014 until it lands in the good basin, then dump that run.")
    else:
        spread = abs(d1.mean() - d2.mean())
        print("  Neither no-dump replicate recovered the good pose, and the dumped run did not"
              " either.\n  => the DUMP IS EXONERATED. The collapse is reproducible without it.")
        print(f"  Replicate means differ by {spread:.4f}, so the run is near-deterministic given the"
              " invocation.")
        print("  => the remaining difference from the sweep is SEED POSITION IN THE PROCESS:\n"
              "     the sweep ran 9x1h's 50 seeds in one invocation (300014 was 15th), these ran it"
              " 1st,\n     so RNG state carried across seeds changes the MSA subsample and hence the"
              " basin.")
        print("  NEXT: reproduce the sweep's ordering (--seeds 300000,...,300014) and dump at the"
              " end,\n     or accept a re-rolled good-basin run as the object of study.")
    print()


if __name__ == "__main__":
    main()
