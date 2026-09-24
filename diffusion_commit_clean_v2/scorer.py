#!/usr/bin/env python
"""Score one fold's poses against the reference and print CSV rows on stdout.

Run as a subprocess by folder.py, once per interface. Writes nothing itself -- the caller captures
stdout and appends it to results.csv. A separate process because DockQ scoring needs a different venv
than folding, and because a pose the scorer chokes on then kills a subprocess rather than the worker
holding the model.

    scorer.py --dir <fold dir> --iface 8w5d_AB --seed 430000 --arm A06 --donor <pose> --ref <cif>
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
# which chains of the donor were pinned as the template. A fold is identified by
# (pose, seed, pinned chains): the same pose pinned to a different antigen copy is a different fold.
ap.add_argument("--chains", default="")
A = ap.parse_args()

# in:  --iface "<pdb>_<C1><C2>", REFERENCE chain letters (not model letters)
# out: the two chain names to score
chains = A.iface.split("_")[1]
if len(chains) != 2:
    raise SystemExit(f"--iface {A.iface}: expected <pdb>_<C1><C2> with single-letter chains")
c1, c2 = chains[0], chains[1]

mod = ost_dockq if A.scorer == "ost" else pxm_dockq
jdir = os.path.join(A.dir, f"{A.scorer}_json")
os.makedirs(jdir, exist_ok=True)

# in:  every *_sample_*.cif under --dir, with its confidence json
# out: one csv line per pose -- iface,seed,arm,donor,chains,sample,dockq,iptm
# Both scorers return {"<refC1>,<refC2>": {"dockq", "model_chains", "pair_iptm"}} and translate
# reference letters to model letters internally, so the csv is identical either way.
for cif in sorted(glob.glob(f"{A.dir}/**/*_sample_*.cif", recursive=True)):
    k = cif.split("sample_")[-1].split(".")[0]
    conf = glob.glob(f"{A.dir}/**/*summary_confidence_sample_{k}.json", recursive=True)
    res = mod.score(cif, A.ref, os.path.join(jdir, f"{k}.json"),
                    conf_json=conf[0] if conf else None)
    v = res.get(f"{c1},{c2}") or res.get(f"{c2},{c1}")
    if not v or v.get("dockq") is None:
        continue          # this pose has no such interface -- a real condition, not an error
    ip = v.get("pair_iptm")
    print(f"{A.iface},{A.seed},{A.arm},{A.donor},{A.chains},{k},{float(v['dockq']):.4f},"
          f"{'' if ip is None else f'{float(ip):.4f}'}")
