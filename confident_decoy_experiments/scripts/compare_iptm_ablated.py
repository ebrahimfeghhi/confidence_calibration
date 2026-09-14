#!/usr/bin/env python
"""Does zeroing the trunk PAIR REP make ipTM read the pose instead of the seed?

THE PROBLEM. ipTM is a seed-level quantity, not a pose-level one: across 262 PXM-2025 interfaces 87%
of its variance is BETWEEN seeds, and on 9x1h it is ~100% -- the five samples of one seed score the
same to three decimals while their DockQ spans 0.044-0.643. The seed signal enters through MSA
subsampling inside the model (protenix/model/utils.py:sample_indices draws a random size AND subset
per seed), which reaches the confidence head only via z_trunk.

THE ABLATION. ConfidenceHead.forward(use_embedding=False) zeroes z_trunk, leaving the head with
s_inputs plus the pose's binned distance map -- the only per-pose input it has. That map is not the
bottleneck: within-seed it tracks structural difference at r = 0.873 against DockQ. It is also a
TRAINED condition, dropped at rate confidence_embedding_drop_rate during training, not an
off-distribution hack.

DUMP_REPS (protenix/model/protenix.py:950) runs the head twice per pose and saves both
chain_pair_iptm matrices: cpi_norm (standard) and cpi_abl (ablated).

WHAT WOULD COUNT. The ablated score should (a) VARY WITHIN a seed where the normal one is flat, and
(b) rank 9x1h's good poses above the confident decoys of seed 300018. Reported here as the
within-seed spread of each score and the rank the best-DockQ pose receives under each.

CHAIN INDEXING: chain_pair_iptm is indexed by asym_id order, not by reference chain name. The mapping
to the reference pair comes from PXMeter's ref_to_model_chain_mapping for that pose, and model chain
letters are converted to indices the same way scripts/selection_protocol.py does it -- never assumed.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/compare_iptm_ablated.py
"""
import csv
import collections
import glob
import json
import os
import re

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
REPS = f"{EXP}/artifacts/reps_9x1h"
PAIRS = (("C", "D"), ("E", "F"))


def chain_idx(name):
    """model chain letter -> asym index, matching selection_protocol.chain_idx"""
    base = name.split("-")[0]
    v = 0
    for ch in base:
        v = v * 26 + (ord(ch.upper()) - 64)
    return v - 1


def main():
    # reference DockQ / epitope for the ORIGINAL poses, for context only
    epi, dq = {}, {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        if r["pdb"] == "9x1h":
            epi[(int(r["seed"]), int(r["sample"]),
                 tuple(sorted((r["ab_chain"], r["ag_chain"]))))] = float(r["jaccard"])
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] == "9x1h" and r["dockq"]:
            dq[(int(r["seed"]), int(r["sample"]),
                tuple(sorted((r["chain1"], r["chain2"]))))] = float(r["dockq"])

    files = sorted(glob.glob(f"{REPS}/9x1h_*.npz"))
    if not files:
        print(f"no npz yet in {REPS}")
        return
    print(f"{len(files)} seeds dumped\n")

    rows = []
    for f in files:
        # runner/inference.py:523 overwrites DUMP_REPS_TAG with "<name>_seed<seed>", so the
        # filename is 9x1h_seed300014.npz regardless of what we passed in.
        seed = int(re.search(r"seed(\d+)\.npz$", f).group(1))
        z = np.load(f, allow_pickle=True)
        if "cpi_norm" not in z or "cpi_abl" not in z:
            print(f"  seed {seed}: npz lacks cpi_norm/cpi_abl (keys: {list(z.keys())[:8]})")
            continue
        cn, ca = z["cpi_norm"], z["cpi_abl"]          # [N_sample, N_chain, N_chain] expected
        for c1, c2 in PAIRS:
            # PXMeter's per-pose mapping, taken from the ORIGINAL run's json for this seed
            for smp in range(cn.shape[0] if cn.ndim == 3 else 1):
                jf = f"{P}/pxm_json/9x1h_{seed}_{smp}.json"
                if not os.path.exists(jf):
                    continue
                r2m = json.load(open(jf)).get("ref_to_model_chain_mapping", {})
                if c1 not in r2m or c2 not in r2m:
                    continue
                i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                M, A = (cn[smp], ca[smp]) if cn.ndim == 3 else (cn, ca)
                if not (0 <= i < M.shape[0] and 0 <= j < M.shape[0]):
                    continue
                rows.append(dict(seed=seed, sample=smp, pair=f"{c1}{c2}",
                                 norm=float(M[i, j]), abl=float(A[i, j]),
                                 dockq=dq.get((seed, smp, tuple(sorted((c1, c2))))),
                                 epi=epi.get((seed, smp, tuple(sorted((c1, c2)))))))

    if not rows:
        print("no rows built -- check chain mapping / npz shape")
        return

    for pair in sorted({r["pair"] for r in rows}):
        R = [r for r in rows if r["pair"] == pair]
        print(f"=== 9x1h_{pair} ===")
        print(f"{'seed':>8}{'smp':>5}{'DockQ':>8}{'epiJ':>7}{'ipTM norm':>11}{'ipTM ABL':>10}")
        for r in sorted(R, key=lambda x: (x["seed"], x["sample"])):
            d = f'{r["dockq"]:.3f}' if r["dockq"] is not None else "  -  "
            e = f'{r["epi"]:.2f}' if r["epi"] is not None else " -  "
            print(f'{r["seed"]:>8}{r["sample"]:>5}{d:>8}{e:>7}{r["norm"]:>11.3f}{r["abl"]:>10.3f}')
        by = collections.defaultdict(list)
        for r in R:
            by[r["seed"]].append(r)
        wn = np.mean([np.std([x["norm"] for x in v]) for v in by.values() if len(v) > 1])
        wa = np.mean([np.std([x["abl"] for x in v]) for v in by.values() if len(v) > 1])
        bn = np.std([np.mean([x["norm"] for x in v]) for v in by.values()])
        ba = np.std([np.mean([x["abl"] for x in v]) for v in by.values()])
        print(f"\n  within-seed SD  normal {wn:.4f}   ablated {wa:.4f}")
        print(f"  between-seed SD normal {bn:.4f}   ablated {ba:.4f}")
        good = [r for r in R if r["dockq"] is not None and r["dockq"] >= 0.23]
        if good:
            for key in ("norm", "abl"):
                order = sorted(R, key=lambda x: -x[key])
                rank = min(order.index(g) + 1 for g in good)
                print(f"  best rank of an ACCEPTABLE pose by {key:>4}: {rank}/{len(R)}")
        print()


if __name__ == "__main__":
    main()
