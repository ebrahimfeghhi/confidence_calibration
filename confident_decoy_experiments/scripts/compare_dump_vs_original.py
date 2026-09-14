#!/usr/bin/env python
"""Did instrumenting the run change the run?

ops/dump_9x1h_two_seeds.sh re-runs 9x1h seeds 300014 (best) and 300018 (most confident) with
DUMP_FULL=1 so the trunk state can be re-fed to the diffusion and confidence heads offline. If the
dump perturbed anything -- a stray in-place op on z, memory pressure changing a kernel choice -- the
saved reps would describe a run that never happened. This puts the rerun's poses next to the sweep's
poses for the SAME two seeds.

EXACT AGREEMENT IS NOT THE BAR. CUDA nondeterminism makes Protenix's end-to-end output irreproducible
run-to-run at fixed seed (established earlier in this project). So the comparison is distributional:
the rerun should land inside the sweep's own sample-to-sample spread. The failure signatures worth
watching for are qualitative, not decimal --
    - seed 300018 no longer scoring ~0.70 pair_iptm  (the decoy's confidence is the whole phenomenon)
    - seed 300014 losing its ~0.6 DockQ pose entirely (the good basin becoming unreachable)
Either would mean the dumped reps do not describe the runs we have been analysing.

READING PXMETER OUTPUT: the interface keys are REFERENCE chain labels, the metric key is lowercase
"dockq", and chain_pair_iptm is indexed by MODEL asym order -- ref_to_model_chain_mapping is what
connects the two. On 9x1h that mapping is NOT the identity (ref C -> model D, ref D -> model E,
ref E -> model C), so skipping it reads the wrong interface's confidence. The extraction below is
copied from scripts/score_pxm2025.py, which is how the sweep's own numbers were produced; the first
version of this script hand-rolled it and silently found nothing.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/compare_dump_vs_original.py
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
OUT = f"{EXP}/artifacts/dump9x1h"
SEEDS = (300014, 300018)
LABEL = {300014: "BEST", 300018: "CONFIDENT"}


def chain_idx(name):
    """Model chain label -> index into chain_pair_iptm (from scripts/score_pxm2025.py)."""
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def original():
    """Sweep DockQ + pair_iptm: (seed, pair) -> {sample: (dockq, iptm)}."""
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] != "9x1h" or not r["dockq"]:
            continue
        seed = int(r["seed"])
        if seed not in SEEDS:
            continue
        pair = "".join(sorted((r["chain1"], r["chain2"])))
        d[(seed, pair)][int(r["sample"])] = (
            float(r["dockq"]), float(r["pair_iptm"]) if r["pair_iptm"] else np.nan)
    return d


def rerun():
    """Instrumented rerun, same keying, read exactly as the sweep reads it."""
    d = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{OUT}/pxm_json/*.json")):
        m = re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        seed, smp = int(m.group(1)), int(m.group(2))
        try:
            J = json.load(open(f))
        except json.JSONDecodeError:
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        # NOTE: one level shallower than the sweep's layout (the sweep dumps to <out>/<pdb>, so it
        # gets <pdb>/<pdb>/seed_*; this run dumped to <out> directly).
        cf = (f"{OUT}/out/9x1h/seed_{seed}/predictions/"
              f"9x1h_summary_confidence_sample_{smp}.json")
        conf = None
        if os.path.exists(cf):
            try:
                conf = json.load(open(cf)).get("chain_pair_iptm")
            except Exception:
                conf = None
        for k, v in (J.get("interface") or {}).items():
            if v.get("dockq") is None:
                continue
            c1, c2 = k.split(",")
            pi = np.nan
            if conf and c1 in r2m and c2 in r2m:
                i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                if 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    pi = float(conf[i][j])
            d[(seed, "".join(sorted((c1, c2))))][smp] = (float(v["dockq"]), pi)
    return d


def main():
    orig, new = original(), rerun()
    if not new:
        print("no rerun scores found -- nothing to compare")
        return
    pairs = sorted({p for _, p in orig} & {p for _, p in new})
    if not pairs:
        print(f"WARNING: no shared chain pairs. sweep={sorted({p for _, p in orig})} "
              f"rerun={sorted({p for _, p in new})}")
        return
    print(f"\ncomparing {len(pairs)} chain pairs: {', '.join(pairs)}")

    for metric, col in (("DockQ", 0), ("pair_iptm", 1)):
        print(f"\n{metric} -- sweep vs instrumented rerun "
              f"(n=5 samples per cell in both)")
        print(f"{'seed':>8} {'role':<10} {'pair':<5} {'sweep max':>10} {'rerun max':>10} "
              f"{'sweep mean':>11} {'rerun mean':>11} {'sweep SD':>9} {'rerun SD':>9}")
        for seed in SEEDS:
            for pair in pairs:
                o, r = orig.get((seed, pair), {}), new.get((seed, pair), {})
                if not o or not r:
                    continue
                ov = np.array([v[col] for v in o.values()], dtype=float)
                rv = np.array([v[col] for v in r.values()], dtype=float)
                print(f"{seed:>8} {LABEL[seed]:<10} {pair:<5} "
                      f"{np.nanmax(ov):>10.3f} {np.nanmax(rv):>10.3f} "
                      f"{np.nanmean(ov):>11.3f} {np.nanmean(rv):>11.3f} "
                      f"{np.nanstd(ov, ddof=1):>9.3f} {np.nanstd(rv, ddof=1):>9.3f}")

    print("\nVERDICT (exact reproduction is NOT expected under CUDA nondeterminism)")
    bad = 0
    for seed in SEEDS:
        for pair in pairs:
            o, r = orig.get((seed, pair), {}), new.get((seed, pair), {})
            if not o or not r:
                continue
            od = np.array([v[0] for v in o.values()]); rd = np.array([v[0] for v in r.values()])
            oi = np.array([v[1] for v in o.values()]); ri = np.array([v[1] for v in r.values()])
            msgs = []
            if od.max() >= 0.23 > rd.max():
                msgs.append(f"LOST the acceptable pose ({od.max():.3f} -> {rd.max():.3f})")
            if np.isfinite(np.nanmean(oi)) and abs(np.nanmean(ri) - np.nanmean(oi)) > 0.15:
                msgs.append(f"confidence moved {np.nanmean(oi):.3f} -> {np.nanmean(ri):.3f}")
            bad += len(msgs)
            print(f"   {seed} {LABEL[seed]:<10} {pair}: "
                  + ("; ".join(msgs) if msgs else "consistent with the sweep"))
    print(f"\n{'DUMP IS TRUSTWORTHY' if bad == 0 else f'{bad} DISCREPANCIES -- investigate'}\n")


if __name__ == "__main__":
    main()
