#!/usr/bin/env python
"""Did self-template rescoring turn ipTM into a POSE-level signal?

Each of 9x1h's 250 poses was handed back as its own template (full inter-chain geometry, cross-chain
mask off, per-chain centering off) and re-run at N_cycle=1 and N_cycle=3. This reads the resulting
ipTM and asks whether it ranks the acceptable poses above the rest -- which the ORIGINAL ipTM does
not: it is ~100% a seed property on this interface, scoring the five samples of a seed alike however
different their structures are, and it ranks the 5 good seeds at the BOTTOM (0.12-0.13 against a 0.39
mean).

THE HEADLINE NUMBER is the rank of the best acceptable pose out of 250, under each score. Original
ipTM puts it 6th at best and typically far worse; a usable selector needs it near the top. Also
reported: AUC (probability a random acceptable pose outranks a random unacceptable one, 0.5 = chance),
and the within-seed vs between-seed variance split, which is the direct test of whether the seed
confound was actually removed.

DockQ comes from the ORIGINAL run (artifacts/pxm2025/dockq_pxm2025.csv) because it is a property of
the pose, and the pose is what was templated -- the rescoring run's own output is a fresh structure
and is not what we are grading.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/read_rescore.py
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
PAIRS = (("C", "D"), ("E", "F"))


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    idx = {r["name"]: r for r in csv.DictReader(open(f"{EXP}/artifacts/rescore/rs/index.csv"))}
    dq, oip = {}, {}
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] != "9x1h" or not r["dockq"]:
            continue
        k = (int(r["seed"]), int(r["sample"]), "".join(sorted((r["chain1"], r["chain2"]))))
        dq[k] = float(r["dockq"])
        if r["pair_iptm"]:
            oip[k] = float(r["pair_iptm"])

    for tag in ("n1", "n3"):
        files = glob.glob(f"{EXP}/artifacts/rescore_out/{tag}/**/*summary_confidence*.json",
                          recursive=True)
        if not files:
            print(f"=== {tag}: no outputs yet\n")
            continue
        rows = []
        for f in files:
            m = re.search(r"(9x1h_rs_(\d+)_(\d+))_summary_confidence", os.path.basename(f))
            if not m or m.group(1) not in idx:
                continue
            seed, smp = int(m.group(2)), int(m.group(3))
            M = json.load(open(f)).get("chain_pair_iptm")
            if not M:
                continue
            e = idx[m.group(1)]
            # the rescoring template used THESE model chains; ipTM is indexed by asym order
            for c1, c2 in PAIRS:
                pass
            # heavy|antigen and light|antigen of the protomer that was templated
            for lab, a, b in (("HA", e["heavy"], e["antigen"]), ("LA", e["light"], e["antigen"])):
                i, j = chain_idx(a), chain_idx(b)
                if not (0 <= i < len(M) and 0 <= j < len(M)):
                    continue
                key = (seed, smp, "".join(sorted(("C", "D"))) if lab == "HA" else "".join(sorted(("A", "F"))))
                rows.append(dict(seed=seed, sample=smp, lab=lab, ip=float(M[i][j]),
                                 dockq=dq.get((seed, smp, "CD"))))
        rows = [r for r in rows if r["lab"] == "HA" and r["dockq"] is not None]
        if not rows:
            print(f"=== {tag}: no joinable rows\n")
            continue
        ip = np.array([r["ip"] for r in rows])
        d = np.array([r["dockq"] for r in rows])
        good = d >= 0.23
        order = np.argsort(-ip)
        rank = int(np.where(good[order])[0][0]) + 1 if good.any() else None
        auc = (float(np.mean([np.mean(ip[good][:, None] > ip[~good][None, :])]))
               if good.any() and (~good).any() else float("nan"))
        by = collections.defaultdict(list)
        for r in rows:
            by[r["seed"]].append(r["ip"])
        win = np.mean([np.std(v) for v in by.values() if len(v) > 1])
        btw = np.std([np.mean(v) for v in by.values()])
        print(f"=== {tag}  ({len(rows)} poses, {int(good.sum())} acceptable)")
        print(f"    rescored ipTM: mean {ip.mean():.3f}  range {ip.min():.3f}-{ip.max():.3f}")
        print(f"    best acceptable pose ranks {rank}/{len(rows)} by rescored ipTM")
        print(f"    AUC (acceptable vs not) = {auc:.3f}     [0.5 = chance]")
        print(f"    within-seed SD {win:.4f}   between-seed SD {btw:.4f}"
              f"   -> {100*btw**2/(win**2+btw**2):.0f}% of variance is between seeds")
        if good.any():
            print(f"    acceptable poses: ipTM " +
                  ", ".join(f"{x:.3f}" for x in sorted(ip[good], reverse=True)[:6]))
        # the same numbers for the ORIGINAL ipTM, on the same poses, as the baseline
        o = np.array([oip.get((r["seed"], r["sample"], "CD"), np.nan) for r in rows])
        ok = ~np.isnan(o)
        if ok.sum() > 10:
            oo = np.argsort(-o[ok])
            gr = good[ok][oo]
            orank = int(np.where(gr)[0][0]) + 1 if gr.any() else None
            oauc = float(np.mean(o[ok][good[ok]][:, None] > o[ok][~good[ok]][None, :]))
            print(f"    ORIGINAL ipTM on the same poses: best acceptable ranks {orank}/{int(ok.sum())}"
                  f", AUC {oauc:.3f}")
        print()


if __name__ == "__main__":
    main()
