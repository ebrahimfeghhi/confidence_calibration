#!/usr/bin/env python
"""Pull the three DockQ COMPONENTS out of the PXM-2025 pose jsons.

DockQ = (fnat + f(iRMSD) + f(LRMSD)) / 3, so the aggregate can hide which part is failing. The
components are already in artifacts/pxm2025/pxm_json/<pdb>_<seed>_<sample>.json under
interface[<pair>]["dockq_info"], but scripts/score_pxm2025.py only ever wrote dockq/lddt/bb_lddt/
pair_iptm into dockq_pxm2025.csv.

WHY A SEPARATE FILE rather than adding columns to score_pxm2025.py: that script runs on a live loop
(ops/pxm2025_scorewatch.sh) and APPENDS to dockq_pxm2025.csv, so changing its column set mid-sweep
would write rows that disagree with the existing header. This reads the same jsons after the fact and
joins on (pdb, chain pair, seed, sample), so nothing in flight is disturbed.

The interface key inside the json is a "C1,C2" string whose order need not match dockq_pxm2025.csv's
chain1/chain2, so the pair is emitted as it appears in the json AND keyed on the sorted pair for
joining.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/dump_dockq_components_pxm.py
Outputs artifacts/pxm2025/dockq_components_pxm2025.csv:
  pdb,chain1,chain2,seed,sample,fnat,fnonnat,irmsd,lrmsd
"""
import csv
import glob
import json
import os
import re
import sys

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
OUT = f"{P}/dockq_components_pxm2025.csv"


def main():
    files = sorted(glob.glob(f"{P}/pxm_json/*_*_*.json"))
    print(f"{len(files)} pose jsons", file=sys.stderr)
    n, bad = 0, 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2", "seed", "sample",
                    "fnat", "fnonnat", "irmsd", "lrmsd"])
        for f in files:
            m = re.search(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", os.path.basename(f))
            if not m:
                continue
            pdb, seed, smp = m.group(1), int(m.group(2)), int(m.group(3))
            try:
                d = json.load(open(f))
            except Exception:
                bad += 1
                continue
            for pair, v in (d.get("interface") or {}).items():
                di = v.get("dockq_info") or {}
                if "iRMSD" not in di or "LRMSD" not in di:
                    continue
                c = pair.split(",")
                if len(c) != 2:
                    continue
                w.writerow([pdb, c[0], c[1], seed, smp, di.get("fnat"),
                            di.get("fnonnat"), di["iRMSD"], di["LRMSD"]])
                n += 1
    print(f"wrote {n} interface-poses to {OUT}" + (f"  ({bad} unreadable jsons)" if bad else ""))


if __name__ == "__main__":
    main()
