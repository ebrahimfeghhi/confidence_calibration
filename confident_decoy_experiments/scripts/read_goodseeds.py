#!/usr/bin/env python
"""Which of 9x1h's five known-good seeds landed in the good basin on re-run, and does the seed
number carry the basin across invocations?

THE DISCRIMINATING NUMBER is the acceptable-pose count out of 25. Within these five seeds the sweep
scored 8/25 = 32%; across all 250 sweep poses the rate was 8/250 = 3.2%. So
    ~8 of 25 -> the seed number transfers; "good seed" is a durable property
    ~1 of 25 -> it does not; the sweep's good seeds are just where the lottery happened to land
Under the pessimistic rate P(zero hits) is about 0.44, so zero is NOT evidence for either side -- it
is a re-roll. This script says so explicitly rather than letting a null read as a finding.

Any seed that DID land good has a usable good-basin pair rep in artifacts/goodseeds/reps, which is
the thing we actually came for -- we already hold the confident decoy's (seed 300018) and have no
good-basin counterpart.

PXMeter interface keys are REFERENCE chain labels and chain_pair_iptm is indexed by MODEL asym order;
ref_to_model_chain_mapping connects them and is not the identity on 9x1h. Same extraction as
scripts/score_pxm2025.py.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/read_goodseeds.py
"""
import collections
import csv
import glob
import json
import os
import re

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
OUT = f"{EXP}/artifacts/goodseeds"
PAIR = ("C", "D")
ACC = 0.23
SEEDS = (300014, 300015, 300038, 300002, 300024)


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def sweep():
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] == "9x1h" and r["dockq"]
                and "".join(sorted((r["chain1"], r["chain2"]))) == "".join(sorted(PAIR))):
            d[int(r["seed"])].append((float(r["dockq"]),
                                      float(r["pair_iptm"]) if r["pair_iptm"] else np.nan))
    return d


def rerun():
    d = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{OUT}/pxm_json/*.json")):
        m = re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        seed, smp = int(m.group(1)), m.group(2)
        try:
            J = json.load(open(f))
        except json.JSONDecodeError:
            continue
        v = (J.get("interface") or {}).get(",".join(PAIR)) or \
            (J.get("interface") or {}).get(",".join(PAIR[::-1]))
        if not v or v.get("dockq") is None:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        cf = (f"{OUT}/out/9x1h/seed_{seed}/predictions/"
              f"9x1h_summary_confidence_sample_{smp}.json")
        pi = np.nan
        if os.path.exists(cf) and all(c in r2m for c in PAIR):
            M = json.load(open(cf)).get("chain_pair_iptm")
            i, j = chain_idx(r2m[PAIR[0]]), chain_idx(r2m[PAIR[1]])
            if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                pi = float(M[i][j])
        d[seed].append((float(v["dockq"]), pi))
    return d


def main():
    sw, rr = sweep(), rerun()
    if not rr:
        print("no rerun scores found -- nothing to report")
        return
    print(f"\n9x1h chain pair {PAIR[0]}{PAIR[1]} -- five known-good seeds, re-run with DUMP_FULL")
    print(f"{'seed':>8} | {'sweep max':>9} {'sweep acc':>9} {'sweep ipTM':>10} "
          f"| {'rerun max':>9} {'rerun acc':>9} {'rerun ipTM':>10} {'n':>3}")
    tot_acc = tot_n = 0
    hits = []
    for seed in SEEDS:
        s, r = sw.get(seed, []), rr.get(seed, [])
        if not r:
            continue
        sd = np.array([x[0] for x in s]) if s else np.array([np.nan])
        si = np.array([x[1] for x in s]) if s else np.array([np.nan])
        rd = np.array([x[0] for x in r])
        ri = np.array([x[1] for x in r])
        nacc = int((rd >= ACC).sum())
        tot_acc += nacc
        tot_n += len(rd)
        if nacc:
            hits.append(seed)
        print(f"{seed:>8} | {np.nanmax(sd):>9.3f} {int((sd >= ACC).sum()):>9} "
              f"{np.nanmean(si):>10.3f} | {rd.max():>9.3f} {nacc:>9} "
              f"{np.nanmean(ri):>10.3f} {len(rd):>3}")
    print(f"\nacceptable poses on re-run: {tot_acc}/{tot_n}"
          f"   (sweep on these same seeds: 8/25 = 32%;  sweep overall: 8/250 = 3.2%)")

    print("\nTRANSFER VERDICT")
    if tot_n == 0:
        print("  no scored poses")
    elif tot_acc == 0:
        print("  ZERO hits. Under the no-transfer rate (3.2%) the expected count is"
              f" {0.032*tot_n:.1f} and P(zero) is about {0.968**tot_n:.2f}, so this is NOT evidence"
              " against transfer --\n  it is a re-roll. Under the transfer rate (32%) P(zero) would"
              f" be {0.68**tot_n:.4f}, which IS strong evidence against transfer.")
    else:
        rate = tot_acc / tot_n
        print(f"  {tot_acc}/{tot_n} = {100*rate:.0f}% acceptable.")
        print("  -> consistent with TRANSFER (sweep rate on these seeds was 32%)" if rate > 0.15
              else "  -> closer to the global 3.2% rate, i.e. NO transfer; these seeds are not"
                   " special on re-run")
        print(f"  GOOD-BASIN DUMPS available for seed(s): {hits}")
        print(f"     artifacts/goodseeds/reps/9x1h_seed<SEED>_reps.npz  (z_trunk, s_trunk, s_inputs)")
    print()


if __name__ == "__main__":
    main()
