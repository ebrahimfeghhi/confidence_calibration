#!/usr/bin/env python
"""Did this seed produce an acceptable 9x1h pose? Exit 0 if yes, 1 if no.

Used by ops/hunt_9x1h.sh as each seed finishes, to decide whether to keep that seed's 3.3 GB trunk
dump or delete it and move on. Prints one line either way so the driver log records every seed's
outcome, not just the winner.

PXMeter's interface keys are REFERENCE chain labels and the metric key is lowercase "dockq";
chain_pair_iptm is indexed by MODEL asym order and needs ref_to_model_chain_mapping, which is not the
identity on 9x1h. Same extraction as scripts/score_pxm2025.py.

Usage: hunt_check.py <seed>
"""
import glob
import json
import os
import re
import sys

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
OUT = f"{EXP}/artifacts/hunt"
PAIR = ("C", "D")
ACC = 0.23


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    seed = sys.argv[1]
    best, best_ip, n = -1.0, float("nan"), 0
    for f in sorted(glob.glob(f"{OUT}/pxm_json/{seed}_*.json")):
        smp = re.search(r"_(\d+)\.json$", os.path.basename(f))
        try:
            J = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        v = (J.get("interface") or {}).get(",".join(PAIR)) or \
            (J.get("interface") or {}).get(",".join(PAIR[::-1]))
        if not v or v.get("dockq") is None:
            continue
        n += 1
        d = float(v["dockq"])
        if d > best:
            best = d
            r2m = J.get("ref_to_model_chain_mapping", {})
            cf = (f"{OUT}/out/9x1h/seed_{seed}/predictions/"
                  f"9x1h_summary_confidence_sample_{smp.group(1)}.json")
            best_ip = float("nan")
            if os.path.exists(cf) and all(c in r2m for c in PAIR):
                M = json.load(open(cf)).get("chain_pair_iptm")
                i, j = chain_idx(r2m[PAIR[0]]), chain_idx(r2m[PAIR[1]])
                if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                    best_ip = float(M[i][j])
    if n == 0:
        print(f"    seed {seed}: NO SCORES ({n} poses read) -- treating as miss")
        sys.exit(1)
    hit = best >= ACC
    print(f"    seed {seed}: max DockQ {best:.3f}  ipTM {best_ip:.3f}  ({n} poses)"
          f"  -> {'HIT' if hit else 'miss'}")
    sys.exit(0 if hit else 1)


if __name__ == "__main__":
    main()
