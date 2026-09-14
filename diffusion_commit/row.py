#!/usr/bin/env python
"""Score every pose of one arm and print one CSV row each. Called by fold_and_commit.sh.

SCORING GOES THROUGH eval/ -- NOT THROUGH A LOCAL COPY. This file used to hand-parse PXMeter json,
which is how 14 near-copies of the same scoring code came to exist and drift apart. `--scorer ost|pxm`
picks one of the two canonical scorers; both return the identical shape

    {"<refC1>,<refC2>": {"dockq": float, "model_chains": (m1, m2), "pair_iptm": float}}

so the CSV is the same either way and the two datasets stay comparable.

CHAIN LETTERS. The interface is keyed on REFERENCE chain names; `pair_iptm` is indexed by MODEL asym
order and the scorers do that translation internally from each pose's own mapping. Reference letters
are NOT model letters -- confusing the two once put 45% of this project's interface measurements on the
wrong chains.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag/eval")
import ost_dockq
import pxm_dockq

ap = argparse.ArgumentParser()
for a in ("--dir", "--iface", "--seed", "--arm", "--donor", "--ref"):
    ap.add_argument(a, required=True)
ap.add_argument("--scorer", choices=("ost", "pxm"), default="pxm")
A = ap.parse_args()

chains = A.iface.split("_")[1]
if len(chains) != 2:
    # Raise rather than guess: a silently mis-split interface would score the wrong pair and still
    # produce a full, plausible-looking CSV.
    raise SystemExit(f"--iface {A.iface}: expected <pdb>_<C1><C2> with single-letter chains")
c1, c2 = chains[0], chains[1]

mod = ost_dockq if A.scorer == "ost" else pxm_dockq
jdir = os.path.join(A.dir, f"{A.scorer}_json")
os.makedirs(jdir, exist_ok=True)

for cif in sorted(glob.glob(f"{A.dir}/out/**/*_sample_*.cif", recursive=True)):
    k = cif.split("sample_")[-1].split(".")[0]
    conf = glob.glob(f"{A.dir}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
    res = mod.score(cif, A.ref, os.path.join(jdir, f"{k}.json"),
                    conf_json=conf[0] if conf else None)
    v = res.get(f"{c1},{c2}") or res.get(f"{c2},{c1}")
    if not v or v.get("dockq") is None:
        continue          # no such interface in this pose -- a real condition, not an error
    ip = v.get("pair_iptm")
    print(f"{A.iface},{A.seed},{A.arm},{A.donor},{k},{float(v['dockq']):.4f},"
          f"{'' if ip is None else f'{float(ip):.4f}'}")
